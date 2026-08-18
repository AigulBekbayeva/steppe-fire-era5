"""Анимация дымового шлейфа поверх карты.

Частицы выпускаются из контура очага и сносятся по азимуту и скорости
текущего кадра временнóй шкалы. Уже выпущенные частицы сохраняют свою
скорость: при развороте ветра шлейф изгибается, как в природе.

Слой рисуется на отдельном canvas. Координаты частиц хранятся в градусах
и пересчитываются в пиксели каждый кадр, поэтому дым остаётся привязанным
к местности при перетаскивании и зуме.
"""

from __future__ import annotations

from branca.element import MacroElement
from jinja2 import Template


class SmokeLayer(MacroElement):
    """Дымовые шлейфы от очагов, снос по ветру текущего кадра."""

    _template = Template(
        """
{% macro script(this, kwargs) %}
(function () {
  var map = {{ this._parent.get_name() }};
  var TIME_SCALE = {{ this.time_scale }};
  var LIFETIME = {{ this.lifetime }};

  // Источники приходят покадрово от временнóй шкалы: какие очаги уже
  // обнаружены, где проходит их кромка и каким ветром сносит дым.
  var sources = [];

  var reduceMotion = window.matchMedia
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var container = map.getContainer();
  var canvas = document.createElement('canvas');
  canvas.style.cssText =
    'position:absolute;top:0;left:0;pointer-events:none;z-index:450';
  container.appendChild(canvas);
  var ctx = canvas.getContext('2d');

  function resize() {
    var ratio = window.devicePixelRatio || 1;
    var size = map.getSize();
    canvas.width = size.x * ratio;
    canvas.height = size.y * ratio;
    canvas.style.width = size.x + 'px';
    canvas.style.height = size.y + 'px';
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  resize();
  map.on('resize', resize);

  var particles = [];
  var running = !reduceMotion;

  function spawn(source) {
    if (source.speed_kmh <= 0.5 || !source.outline.length) { return; }

    // Частицы стартуют с кромки очага, а не из его центра.
    var node = source.outline[Math.floor(Math.random() * source.outline.length)];
    var spread = (Math.random() - 0.5) * 26;
    var bearing = (source.bearing + spread) * Math.PI / 180;
    var speed = source.speed_kmh * (0.75 + Math.random() * 0.5);
    var cosLat = Math.cos(node[0] * Math.PI / 180);

    particles.push({
      lat: node[0],
      lon: node[1],
      vlat: Math.cos(bearing) * speed / 110.54,
      vlon: Math.sin(bearing) * speed / (111.32 * cosLat),
      age: 0,
      life: LIFETIME * (0.7 + Math.random() * 0.6),
      seed: Math.random() * 6.28,
      swing: 0.15 + Math.random() * 0.25,
      grow: 0.9 + Math.random() * 0.9
    });
  }

  var carry = [];
  function emit(dt) {
    for (var i = 0; i < sources.length; i++) {
      carry[i] = (carry[i] || 0) + sources[i].rate * dt;
      while (carry[i] >= 1) {
        spawn(sources[i]);
        carry[i] -= 1;
      }
    }
  }

  function metersPerPixel() {
    var lat = map.getCenter().lat * Math.PI / 180;
    return 40075016.686 * Math.cos(lat) / Math.pow(2, map.getZoom() + 8);
  }

  var last = performance.now();

  function frame(now) {
    var dt = Math.min((now - last) / 1000, 0.1);
    last = now;

    if (!running || document.hidden) {
      requestAnimationFrame(frame);
      return;
    }

    if (sources.length) { emit(dt); }

    var hours = dt * TIME_SCALE / 3600;
    var kmPerPixel = metersPerPixel() / 1000;
    var bounds = map.getBounds().pad(0.35);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'source-over';

    var alive = [];
    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.age += dt;
      if (p.age > p.life) { continue; }

      var wobble = Math.sin(p.age * p.swing * 3 + p.seed) * 0.0009;
      p.lat += p.vlat * hours - p.vlon * wobble;
      p.lon += p.vlon * hours + p.vlat * wobble;
      alive.push(p);

      if (!bounds.contains([p.lat, p.lon])) { continue; }

      var progress = p.age / p.life;
      var point = map.latLngToContainerPoint([p.lat, p.lon]);
      var radius = (0.6 + progress * 5.2 * p.grow) / kmPerPixel;
      if (radius < 1.2) { radius = 1.2; }

      var fadeIn = Math.min(progress / 0.12, 1);
      var fadeOut = 1 - Math.max(0, (progress - 0.35) / 0.65);
      var alpha = 0.34 * fadeIn * fadeOut * fadeOut;
      if (alpha <= 0.004) { continue; }

      var gradient = ctx.createRadialGradient(
        point.x, point.y, 0, point.x, point.y, radius
      );
      // Сиреневый: на тёмной подложке серый дым сливается с фоном,
      // а холодный лиловый читается и не спорит с тёплыми точками огня.
      gradient.addColorStop(0, 'rgba(226,222,248,' + alpha + ')');
      gradient.addColorStop(0.55, 'rgba(186,178,232,' + alpha * 0.6 + ')');
      gradient.addColorStop(1, 'rgba(150,142,214,0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, 6.2832);
      ctx.fill();
    }
    particles = alive.length > 2600 ? alive.slice(alive.length - 2600) : alive;

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // Временная шкала сообщает сюда состояние текущего кадра.
  window.smokeControl = {
    setFrame: function (active, scrubbed) {
      sources = active || [];
      carry = [];
      // При ручной перемотке старый шлейф не соответствует новому времени.
      if (scrubbed) { particles = []; }
    },
    setRunning: function (value) {
      running = value && !reduceMotion;
      if (!running) { ctx.clearRect(0, 0, canvas.width, canvas.height); }
      else { last = performance.now(); }
    },
    isRunning: function () { return running; },
    clear: function () { particles = []; }
  };

  var toggle = L.control({ position: 'topright' });
  toggle.onAdd = function () {
    var div = L.DomUtil.create('div', 'leaflet-bar');
    var button = L.DomUtil.create('a', '', div);
    button.href = '#';
    button.title = 'Анимация дыма';
    button.style.cssText =
      'width:auto;padding:0 10px;font:600 12px system-ui;text-decoration:none';
    button.innerHTML = running ? 'Дым: вкл' : 'Дым: выкл';

    L.DomEvent.on(button, 'click', function (event) {
      L.DomEvent.preventDefault(event);
      L.DomEvent.stopPropagation(event);
      window.smokeControl.setRunning(!window.smokeControl.isRunning());
      button.innerHTML = window.smokeControl.isRunning() ? 'Дым: вкл' : 'Дым: выкл';
    });
    L.DomEvent.disableClickPropagation(div);
    return div;
  };
  toggle.addTo(map);
})();
{% endmacro %}
        """
    )

    def __init__(self, time_scale: float = 300.0, lifetime: float = 9.0):
        super().__init__()
        self._name = "SmokeLayer"
        self.time_scale = time_scale
        self.lifetime = lifetime


def emission_rates(clusters) -> list[float]:
    """Плотность струи растёт с мощностью очага, но упирается в потолок."""
    return [round(6.0 + 26.0 * min(c.total_frp / 2500.0, 1.0), 1) for c in clusters]
