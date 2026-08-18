"""Временнáя шкала реконструкции события.

Кадр — один час. По ходу воспроизведения накапливаются термоточки,
очаги загораются в момент первого обнаружения, контур гари растёт,
а дым идёт только от уже обнаруженных очагов.
"""

from __future__ import annotations

import json

from branca.element import MacroElement
from jinja2 import Template

# Полный проход шкалы должен укладываться примерно в это время: длиннее —
# утомительно смотреть и неудобно записывать в ролик.
TARGET_PLAYBACK_SECONDS = 20


def playback_pace(frame_count: int) -> int:
    """Темп воспроизведения, мс на кадр."""
    if frame_count < 2:
        return 450
    return int(min(max(TARGET_PLAYBACK_SECONDS * 1000 / frame_count, 120), 600))


class Timeline(MacroElement):
    """Панель управления временем с ползунком и воспроизведением."""

    _template = Template(
        """
{% macro html(this, kwargs) %}
<div id="timeline-panel" style="
    position:fixed;left:50%;transform:translateX(-50%);bottom:22px;z-index:9999;
    background:rgba(17,20,28,0.93);border-radius:12px;padding:12px 18px 14px;
    box-shadow:0 2px 20px rgba(0,0,0,0.55);font-family:system-ui,sans-serif;
    color:#e8eaf0;border:1px solid rgba(255,255,255,0.08);
    width:min(720px,calc(100vw - 48px))">
  <div style="display:flex;align-items:center;gap:14px">
    <button id="timeline-play" style="
        width:36px;height:36px;flex:0 0 36px;border:0;border-radius:50%;
        background:#e53935;color:#fff;font-size:14px;cursor:pointer;
        display:flex;align-items:center;justify-content:center">&#9654;</button>
    <div style="flex:1;min-width:0">
      <div style="display:flex;justify-content:space-between;align-items:baseline;
                  font-size:12px;color:#9aa3b2;margin-bottom:3px">
        <span id="timeline-phase"></span>
        <span id="timeline-clock" style="font-size:15px;font-weight:700;color:#fff"></span>
        <span id="timeline-counter"></span>
      </div>
      <input id="timeline-slider" type="range" min="0" max="1" value="0" step="1"
             style="width:100%;accent-color:#e53935;cursor:pointer">
    </div>
  </div>
  <div id="timeline-readout" style="
      margin-top:8px;font-size:12px;color:#c8cedb;line-height:1.6;
      border-top:1px solid rgba(255,255,255,0.1);padding-top:8px;
      max-height:132px;overflow-y:auto"></div>
</div>
{% endmacro %}

{% macro script(this, kwargs) %}
(function () {
  var map = {{ this._parent.get_name() }};
  var frames = {{ this.frames }};
  var tracks = {{ this.tracks }};
  var detections = {{ this.detections }};
  var observedFrom = {{ this.observed_until }};
  var STEP_MS = {{ this.step_ms }};

  var slider = document.getElementById('timeline-slider');
  var play = document.getElementById('timeline-play');
  var clock = document.getElementById('timeline-clock');
  var phase = document.getElementById('timeline-phase');
  var counter = document.getElementById('timeline-counter');
  var readout = document.getElementById('timeline-readout');

  slider.max = frames.length - 1;

  // Повторяющиеся контуры пришли как null — восстанавливаем ссылку на
  // предыдущий, чтобы перемотка в любую сторону работала одинаково.
  tracks.forEach(function (track) {
    var last = null;
    track.frames.forEach(function (step) {
      if (!step) { return; }
      if (step.outline === null) { step.outline = last; }
      else { last = step.outline; }
    });
  });

  // Один полигон на очаг: форма переписывается, слой не пересоздаётся.
  var shapes = tracks.map(function () {
    return L.polygon([[0, 0]], {
      color: '#ff5252', weight: 1.5, opacity: 0.85,
      fillColor: '#e53935', fillOpacity: 0.38
    });
  });

  var badges = tracks.map(function (track, i) {
    return L.marker([0, 0], {
      icon: L.divIcon({
        html: "<div style='background:#37474f;color:#fff;border-radius:11px;" +
              "width:22px;height:22px;line-height:22px;text-align:center;" +
              "font:600 12px system-ui;box-shadow:0 1px 4px rgba(0,0,0,.4)'>" +
              (i + 1) + "</div>",
        iconSize: [22, 22], iconAnchor: [11, 11], className: ''
      })
    });
  });

  // Термоточки разложены по кадрам: на каждом шаге добавляется своя порция.
  var buckets = {};
  detections.forEach(function (point) {
    (buckets[point[2]] = buckets[point[2]] || []).push(point);
  });

  var hotspotLayer = L.layerGroup().addTo(map);
  var shown = -1;

  // Шкала мощности излучения: от тлеющей кромки к активному фронту.
  var FRP_STOPS = [
    { limit: 4, color: '#ffe082' },
    { limit: 15, color: '#ffc107' },
    { limit: 30, color: '#ff9800' },
    { limit: 60, color: '#f4511e' },
    { limit: Infinity, color: '#d50000' }
  ];

  function frpColor(power) {
    for (var i = 0; i < FRP_STOPS.length; i++) {
      if (power < FRP_STOPS[i].limit) { return FRP_STOPS[i].color; }
    }
    return '#d50000';
  }

  function marker(point) {
    var power = point[3] || 0;
    // Радиус растёт медленно: иначе мощные точки затирают карту.
    var radius = 2.2 + Math.min(Math.sqrt(power) / 2.6, 4.2);
    return L.circleMarker([point[0], point[1]], {
      radius: radius,
      color: frpColor(power),
      fill: true,
      fillColor: frpColor(power),
      fillOpacity: 0.9,
      weight: 0,
      interactive: false
    });
  }

  function syncHotspots(index) {
    if (index === shown) { return; }

    // Вперёд — досыпаем, назад — пересобираем: перемотка редка, экономить не на чем.
    if (index < shown) {
      hotspotLayer.clearLayers();
      shown = -1;
    }
    for (var f = shown + 1; f <= index; f++) {
      (buckets[f] || []).forEach(function (point) { hotspotLayer.addLayer(marker(point)); });
    }
    shown = index;
  }

  /**
   * Плоский список вершин кромки для дыма.
   *
   * Контур многосвязный: список частей, каждая — список колец. Дыму нужны
   * просто точки, откуда выпускать частицы. Результат кэшируется на кадре:
   * колец бывает под сотню, а вызывается это на каждом шаге шкалы.
   */
  function edgeVertices(step) {
    if (step._edge) { return step._edge; }
    var points = [];
    for (var p = 0; p < step.outline.length; p++) {
      for (var r = 0; r < step.outline[p].length; r++) {
        var ring = step.outline[p][r];
        for (var v = 0; v < ring.length; v++) { points.push(ring[v]); }
      }
    }
    step._edge = points;
    return points;
  }

  var current = 0;
  var playing = false;
  var timer = null;

  function render(index, scrubbed) {
    current = index;
    clock.textContent = frames[index].label;
    phase.textContent = index <= observedFrom ? 'наблюдение' : 'модель вперёд';

    syncHotspots(index);

    var rows = [];
    var active = [];
    var live = 0;

    for (var i = 0; i < tracks.length; i++) {
      var step = tracks[i].frames[index];

      if (!step) {
        map.removeLayer(shapes[i]);
        map.removeLayer(badges[i]);
        continue;
      }

      live++;
      shapes[i].setLatLngs(step.outline).addTo(map);
      badges[i].setLatLng(tracks[i].centroid).addTo(map);

      active.push({
        outline: edgeVertices(step),
        bearing: step.bearing,
        speed_kmh: step.speed_kmh,
        rate: tracks[i].rate
      });

      var mark = step.observed
        ? " <span style='color:#0288d1' title='съёмка в этот час'>&#9679;</span>"
        : '';
      rows.push(
        "<div style='display:flex;justify-content:space-between;gap:12px'>" +
        "<span><b style='color:#fff'>Очаг " + (i + 1) + "</b>" + mark + " &middot; ветер " +
        step.speed_kmh.toFixed(0) + " км/ч, снос на " + step.to_label + "</span>" +
        "<span>" + Math.round(step.area_ha).toLocaleString('ru-RU') + " га</span></div>"
      );
    }

    counter.textContent = live + ' из ' + tracks.length + ' очагов';
    readout.innerHTML = rows.length
      ? rows.join('')
      : "<span style='color:#7c8598'>Очаги ещё не обнаружены</span>";

    if (window.smokeControl) { window.smokeControl.setFrame(active, scrubbed); }
  }

  function stop() {
    playing = false;
    play.innerHTML = '&#9654;';
    if (timer) { clearInterval(timer); timer = null; }
  }

  function start() {
    playing = true;
    play.innerHTML = '&#10073;&#10073;';
    timer = setInterval(function () {
      var next = current + 1;
      var looped = next >= frames.length;
      if (looped) { next = 0; }
      slider.value = next;
      render(next, looped);
    }, STEP_MS);
  }

  play.addEventListener('click', function () { playing ? stop() : start(); });
  slider.addEventListener('input', function () {
    stop();
    render(parseInt(slider.value, 10), true);
  });

  document.addEventListener('keydown', function (event) {
    if (event.target.tagName === 'INPUT' && event.key !== ' ') { return; }
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      stop();
      var next = current + (event.key === 'ArrowRight' ? 1 : -1);
      next = Math.max(0, Math.min(frames.length - 1, next));
      slider.value = next;
      render(next, true);
    } else if (event.key === ' ') {
      event.preventDefault();
      playing ? stop() : start();
    }
  });

  L.DomEvent.disableClickPropagation(document.getElementById('timeline-panel'));
  L.DomEvent.disableScrollPropagation(document.getElementById('timeline-panel'));

  render(0, true);
})();
{% endmacro %}
        """
    )

    def __init__(
        self,
        frames: list[dict],
        tracks: list[dict],
        detections: list[list],
        observed_until: int,
        step_ms: int | None = None,
    ):
        super().__init__()
        self._name = "Timeline"
        self.frames = json.dumps(frames, ensure_ascii=False)
        self.tracks = json.dumps(tracks, ensure_ascii=False)
        self.detections = json.dumps(detections)
        self.observed_until = observed_until
        self.step_ms = step_ms if step_ms is not None else playback_pace(len(frames))


def build_timeline_data(clusters, timeline, rates: list[float]):
    """Готовит подписи кадров и покадровые данные по каждому очагу."""
    labels = timeline.labels()

    tracks = []
    for cluster, rate in zip(clusters, rates, strict=True):
        frames = []
        previous = None

        for frame in cluster.spread["frames"]:
            if frame is None:
                frames.append(None)
                continue

            # Четыре знака — это около 11 м, много точнее пикселя прибора.
            outline = [
                [[[round(lat, 4), round(lon, 4)] for lat, lon in ring] for ring in part]
                for part in frame["outline"]
            ]

            # После догорания контур замирает, и десятки кадров дублируют
            # друг друга. Повтор помечается null и берётся из предыдущего.
            repeated = outline == previous
            previous = outline

            frames.append(
                {
                    "outline": None if repeated else outline,
                    "area_ha": round(frame["area_ha"], 1),
                    "ros_kmh": round(frame["ros_kmh"], 2),
                    "speed_kmh": round(frame["speed_kmh"], 1),
                    "bearing": round(frame["bearing"], 1),
                    "to_label": frame["to_label"],
                    "observed": frame["observed"],
                }
            )

        tracks.append(
            {
                "title": cluster.title,
                "centroid": [round(cluster.centroid[0], 5), round(cluster.centroid[1], 5)],
                "rate": rate,
                "frames": frames,
            }
        )

    return labels, tracks


def build_detections(hotspots, timeline) -> list[list]:
    """Термоточки с номером кадра и мощностью излучения.

    FRP нужен, чтобы окрасить точку: слабое тление и активный фронт
    отличаются на два порядка, и одним цветом их показывать бессмысленно.
    """
    points = []
    for _, row in hotspots.iterrows():
        index = timeline.index_of(row["acquired_at"])
        if index is None:
            continue

        frp = row.get("frp")
        power = round(float(frp), 1) if frp == frp and frp is not None else 0.0
        points.append([round(row["latitude"], 4), round(row["longitude"], 4), index, power])
    return points
