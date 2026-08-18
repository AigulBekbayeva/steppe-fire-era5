"""Отрисовка результата: интерактивная карта в один HTML-файл."""

from __future__ import annotations

import math

import folium
from folium.plugins import Fullscreen, MeasureControl

from .smoke import SmokeLayer, emission_rates
from .timeline import Timeline, build_detections, build_timeline_data

DISCLAIMER = (
    "Индикативная оценка по модели травяных пожаров Cheney et al. (1998). "
    "Не учитывает рельеф, барьеры и работу пожарных расчётов. "
    "Гарь по снимкам — объединение снятых пикселей. Зона распространения — "
    "куда огонь мог бы дойти без тушения и барьеров, а не что выгорит. "
    "Не является оперативным прогнозом и не заменяет данные МЧС."
)


def _wind_arrow(lat: float, lon: float, bearing: float, length_km: float = 12.0) -> list:
    """Координаты стрелки, показывающей направление сноса огня."""
    radians = math.radians(bearing)
    dlat = (length_km / 110.54) * math.cos(radians)
    dlon = (length_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(radians)
    return [[lat, lon], [lat + dlat, lon + dlon]]


def cluster_popup(cluster) -> str:
    """HTML-карточка очага."""
    weather = cluster.wind
    spread = cluster.spread
    rows = [
        ("Термоточек", f"{cluster.count}"),
        ("Суммарная FRP", f"{cluster.total_frp:,.0f} МВт".replace(",", " ")),
        ("Протяжённость", f"{cluster.span_km:.1f} км"),
        ("Последняя съёмка", cluster.latest.strftime("%d.%m.%Y %H:%M UTC")),
        ("Обнаружен", spread["ignition_at"].replace("T", " ")[:16] + " UTC"),
        ("Пролётов в данных", f"{len(spread['pass_indices'])}"),
        ("Ветер при обнаружении", f"{weather['speed_kmh']:.0f} км/ч, из {weather['from_label']}"),
        ("Температура", f"{weather['temperature_c']:.0f} °C"),
        ("Влажность воздуха", f"{weather['humidity_pct']:.0f} %"),
        ("Влажность топлива", f"{spread['fuel_moisture_pct']:.1f} %"),
        ("Контур при обнаружении", f"{spread['current_area_ha']:,.0f} га".replace(",", " ")),
        ("Зона распространения", f"{spread['final_area_ha']:,.0f} га".replace(",", " ")),
        ("Пройдёт фронт", f"{spread['total_km']:.1f} км"),
    ]

    body = "".join(
        f"<tr><td style='padding:3px 10px 3px 0;color:#666'>{key}</td>"
        f"<td style='padding:3px 0;font-weight:600'>{value}</td></tr>"
        for key, value in rows
    )

    return (
        f"<div style='font-family:system-ui,sans-serif;font-size:13px;min-width:290px'>"
        f"<div style='font-size:15px;font-weight:700;margin-bottom:8px'>{cluster.title}</div>"
        f"<table>{body}</table>"
        f"<div style='margin-top:8px;color:#999;font-size:11px;line-height:1.4'>"
        f"Оценка индикативная, не оперативный прогноз.</div></div>"
    )


def legend_html(region: str, moment: str, cluster_count: int, hotspot_count: int) -> str:
    """Панель с легендой и оговоркой.

    Мощность излучения и плотность дыма показаны градиентными шкалами:
    точки различаются на два порядка, и дискретная легенда это скрывает.
    """
    frp_stops = ["#ffe082", "#ffc107", "#ff9800", "#f4511e", "#d50000"]
    gradient = ", ".join(frp_stops)

    return f"""
    <div style="position:fixed;top:14px;left:14px;z-index:9999;
                background:rgba(17,20,28,0.92);padding:14px 16px;border-radius:10px;
                box-shadow:0 2px 16px rgba(0,0,0,0.5);font-family:system-ui,sans-serif;
                font-size:12px;max-width:300px;line-height:1.5;color:#e8eaf0;
                border:1px solid rgba(255,255,255,0.08)">
      <div style="font-size:14px;font-weight:700;margin-bottom:2px;color:#fff">
        Распространение пожаров
      </div>
      <div style="color:#9aa3b2;margin-bottom:10px">{region} · {moment}</div>
      <div style="margin-bottom:10px;color:#c8cedb">
        Очагов: <b style="color:#fff">{cluster_count}</b> ·
        термоточек: <b style="color:#fff">{hotspot_count}</b>
      </div>

      <div style="color:#9aa3b2;margin-bottom:4px">Мощность излучения (FRP, МВт)</div>
      <div style="height:9px;border-radius:5px;margin-bottom:2px;
                  background:linear-gradient(90deg, {gradient})"></div>
      <div style="display:flex;justify-content:space-between;color:#7c8598;
                  font-size:10px;margin-bottom:10px">
        <span>4</span><span>15</span><span>30</span><span>60</span><span>100+</span>
      </div>

      <div style="color:#9aa3b2;margin-bottom:4px">Дым, снос по ветру</div>
      <div style="height:9px;border-radius:5px;margin-bottom:10px;
                  background:linear-gradient(90deg,
                    rgba(150,142,214,0), rgba(186,178,232,.7), rgba(226,222,248,1))"></div>

      <div style="display:flex;align-items:center;gap:8px;margin:3px 0;color:#c8cedb">
        <span style="width:16px;height:10px;background:#e53935;opacity:.65;
                     border-radius:2px"></span>
        <span>Гарь по снимкам</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin:3px 0;color:#c8cedb">
        <span style="width:16px;height:0;border-top:2px dashed #ff5252"></span>
        <span>Зона возможного распространения</span>
      </div>

      <div style="margin-top:10px;padding-top:8px;
                  border-top:1px solid rgba(255,255,255,0.1);
                  color:#79818f;font-size:10.5px">{DISCLAIMER}</div>
    </div>
    """


def build_map(
    clusters,
    hotspots,
    region: str,
    moment: str,
    output: str,
    smoke: bool = True,
    timeline: bool = True,
    event=None,
) -> str:
    """Собирает карту и сохраняет в HTML."""
    center = clusters[0].centroid if clusters else (49.8, 73.1)

    fmap = folium.Map(location=list(center), zoom_start=7, tiles=None)
    # Тёмная подложка: на ней светятся точки по мощности и читается дым.
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
        attr="CartoDB, OpenStreetMap",
        name="Тёмная",
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB, OpenStreetMap",
        name="Тёмная с подписями",
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Спутник",
    ).add_to(fmap)

    # Стрелки сноса и карточки очагов — статический слой.
    # Контуры и термоточки ведёт временная шкала.
    base = folium.FeatureGroup(name="Очаги: карточки и ветер", show=True)
    for cluster in clusters:
        folium.PolyLine(
            _wind_arrow(*cluster.centroid, cluster.wind["spread_bearing"]),
            color="#4fc3f7",
            weight=3,
            opacity=0.7,
            tooltip=f"Ветер на первой съёмке: {cluster.wind['speed_kmh']:.0f} км/ч "
            f"из {cluster.wind['from_label']}",
        ).add_to(base)

        # Контур многосвязный: очаг может состоять из нескольких пятен,
        # а folium, в отличие от Leaflet, вложенность не разбирает.
        for part in cluster.spread["final_outline"]:
            for ring in part:
                folium.Polygon(
                    locations=ring,
                    color="#ff5252",
                    weight=1,
                    opacity=0.4,
                    fill=False,
                    dash_array="4,6",
                    tooltip=f"{cluster.title}: зона возможного распространения",
                ).add_to(base)

        folium.Marker(
            location=list(cluster.centroid),
            icon=folium.DivIcon(html="<div></div>", icon_size=(1, 1)),
            popup=folium.Popup(cluster_popup(cluster), max_width=340),
        ).add_to(base)
    base.add_to(fmap)

    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    Fullscreen(position="topright").add_to(fmap)
    MeasureControl(primary_length_unit="kilometers", primary_area_unit="hectares").add_to(fmap)

    if smoke:
        SmokeLayer().add_to(fmap)

    if timeline and clusters and event is not None:
        rates = emission_rates(clusters)
        labels, tracks = build_timeline_data(clusters, event, rates)
        detections = build_detections(hotspots, event)
        observed_until = max(
            (i for c in clusters for i in c.spread["pass_indices"]), default=0
        )
        Timeline(labels, tracks, detections, observed_until).add_to(fmap)

    span = f"{event.stamps[0]:%d.%m %H:%M} — {event.stamps[-1]:%d.%m %H:%M} UTC" if event else moment
    fmap.get_root().html.add_child(
        folium.Element(legend_html(region, span, len(clusters), len(hotspots)))
    )

    fmap.save(output)
    return output
