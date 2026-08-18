"""Оценка распространения огня по степной растительности.

Скорость фронта считается по модели Cheney, Gould & Catchpole (1998) для
травяных пожаров — она разработана на австралийских пастбищах и применима
к степи как ближайший опубликованный аналог. Это индикативная оценка:
модель не учитывает рельеф, барьеры, порывистость ветра и неоднородность
топлива, поэтому результат нельзя использовать для оперативных решений.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

METERS_PER_DEG_LAT = 110_540.0
METERS_PER_DEG_LON = 111_320.0

# Радиус пикселя, м. Термоточка — это площадка, а не координата.
PIXEL_RADIUS_M = {"VIIRS": 187.5, "MODIS": 500.0}


def fine_fuel_moisture(temperature_c: float, humidity_pct: float) -> float:
    """Влагосодержание тонких мёртвых горючих материалов, % (McArthur)."""
    return max(1.0, 9.58 - 0.205 * temperature_c + 0.138 * humidity_pct)


def moisture_coefficient(moisture_pct: float, wind_kmh: float) -> float:
    """Поправка на влажность топлива (Cheney et al., 1998)."""
    if moisture_pct < 12.0:
        return math.exp(-0.108 * moisture_pct)
    if wind_kmh < 10.0:
        return max(0.0, 0.684 - 0.0342 * moisture_pct)
    return max(0.0, 0.547 - 0.0228 * moisture_pct)


def curing_coefficient(curing_pct: float) -> float:
    """Поправка на степень усыхания травостоя (Cruz et al., 2015)."""
    return 1.12 / (1.0 + 59.2 * math.exp(-0.124 * (curing_pct - 50.0)))


def grass_rate_of_spread(
    wind_kmh: float,
    temperature_c: float,
    humidity_pct: float,
    curing_pct: float = 90.0,
) -> dict:
    """Скорость движения головной части фронта, км/ч."""
    moisture = fine_fuel_moisture(temperature_c, humidity_pct)
    phi_m = moisture_coefficient(moisture, wind_kmh)
    phi_c = curing_coefficient(curing_pct)

    # noqa: SIM108 — кусочная запись повторяет формулу из статьи, тернарник её прячет.
    if wind_kmh <= 5.0:  # noqa: SIM108
        base = 0.054 + 0.269 * wind_kmh
    else:
        base = 1.4 + 0.838 * (wind_kmh - 5.0) ** 0.844

    return {
        "ros_kmh": max(0.0, base * phi_m * phi_c),
        "fuel_moisture_pct": moisture,
        "moisture_coeff": phi_m,
        "curing_coeff": phi_c,
    }


def length_to_breadth(wind_kmh: float) -> float:
    """Отношение длины эллипса пожара к ширине (Cheney, 1993)."""
    if wind_kmh < 1.0:
        return 1.0
    return float(np.clip(1.1 * wind_kmh**0.464, 1.0, 8.0))


def to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Локальные метры относительно опорной точки."""
    return (
        (lon - lon0) * METERS_PER_DEG_LON * math.cos(math.radians(lat0)),
        (lat - lat0) * METERS_PER_DEG_LAT,
    )


def to_geographic(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Обратное преобразование в широту и долготу."""
    return (
        lat0 + y / METERS_PER_DEG_LAT,
        lon0 + x / (METERS_PER_DEG_LON * math.cos(math.radians(lat0))),
    )


def footprint(points, lat0: float, lon0: float, instrument: str = "VIIRS"):
    """Контур очага как объединение снятых пикселей.

    Раньше здесь бралась выпуклая оболочка всех термоточек. Это давало
    кратное завышение: оболочка закрашивает всё пространство между
    разрозненными точками, включая заведомо несгоревшее. Для комплекса
    из сотен точек, разбросанных на десятки километров, разница выходила
    в разы.

    Теперь каждая точка превращается в диск размером с пиксель прибора,
    и они объединяются. Площадь получается не больше, чем число снятых
    пикселей на их площадь, — это и есть физический потолок оценки.

    Результат может быть многосвязным: очаг вполне может состоять из
    нескольких пятен.
    """
    radius = PIXEL_RADIUS_M.get(instrument, 187.5)
    discs = [
        Point(*to_local(lat, lon, lat0, lon0)).buffer(radius, quad_segs=3)
        for lat, lon in points
    ]
    return unary_union(discs)


def pixel_ceiling(point_count: int, instrument: str = "VIIRS") -> float:
    """Физический потолок площади: число снятых пикселей на их площадь.

    Оценка гари не может превышать эту величину — если превышает,
    в геометрии ошибка.
    """
    radius = PIXEL_RADIUS_M.get(instrument, 187.5)
    return point_count * math.pi * radius**2 / 10_000.0


def close_front(geometry, gap_m: float = 1200.0):
    """Смыкает разрозненные пиксели в связную кромку.

    Прогноз нельзя вести от каждого пикселя по отдельности: сотня точек
    даёт сотню независимых коридоров, и объединение раздувается на порядок.
    Морфологическое замыкание собирает близкие пиксели в единый фронт,
    каким он и является физически.
    """
    if geometry.is_empty:
        return geometry
    return geometry.buffer(gap_m, quad_segs=4).buffer(-gap_m, quad_segs=4)


def parts_of(geometry):
    """Связные части геометрии — очаг может состоять из нескольких пятен."""
    if geometry.is_empty:
        return []
    return list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]


def project_front(
    geometry,
    bearing_deg: float,
    ros_kmh: float,
    hours: float,
    lb_ratio: float,
):
    """Контур возможного охвата через заданное число часов.

    Головная часть смещается по ветру на ros × hours, фланги расширяются
    во столько раз медленнее, каково отношение длины к ширине эллипса.

    Коридор заметания строится для каждой связной части отдельно. Общая
    выпуклая оболочка, которая была здесь раньше, склеивала разнесённые
    пятна в один сплошной массив и завышала площадь в разы.
    """
    distance_m = ros_kmh * hours * 1000.0
    if distance_m <= 0:
        return geometry, 0.0

    radians = math.radians(bearing_deg)
    dx = math.sin(radians) * distance_m
    dy = math.cos(radians) * distance_m
    flank = distance_m / max(lb_ratio, 1.0)

    swept = []
    for part in parts_of(geometry):
        head = translate(part, xoff=dx, yoff=dy).buffer(flank)
        swept.append(unary_union([part, head]).convex_hull)

    return unary_union(swept), distance_m


# Параметры отрисовки. На расчёт площади не влияют: она считается по точной
# геометрии, а здесь только уменьшается вес карты.
DISPLAY_TOLERANCE_M = 120.0

# Соседние пиксели при отрисовке смыкаются в пятна. Просветы меньше этого
# порога между детекциями — почти наверняка один пожар, а не два.
DISPLAY_CLOSING_M = 300.0


def polygon_to_latlon(geometry, lat0: float, lon0: float) -> list:
    """Координаты контура для Leaflet.

    Возвращается форма многосвязного полигона: список частей, каждая —
    список колец. Leaflet различает уровни вложенности, и такая форма
    однозначно читается как несколько отдельных пятен, а не как одно
    пятно с дырками.
    """
    display = close_front(geometry, DISPLAY_CLOSING_M).simplify(DISPLAY_TOLERANCE_M)

    parts = []
    for part in parts_of(display):
        ring = [list(to_geographic(x, y, lat0, lon0)) for x, y in part.exterior.coords]
        if len(ring) >= 4:
            parts.append([ring])
    return parts


def area_hectares(polygon: Polygon) -> float:
    """Площадь контура в гектарах (геометрия в метрах)."""
    return polygon.area / 10_000.0


def largest_polygon(geometry):
    """Раньше отбрасывала все части кроме крупнейшей — теперь сохраняет все.

    Очаг может состоять из нескольких пятен, и выбрасывание мелких
    занижало площадь так же необоснованно, как оболочка её завышала.
    """
    return geometry


def simulate_track(
    cluster,
    series: list[dict],
    curing_pct: float = 90.0,
    instrument: str = "VIIRS",
    step_hours: float = 1.0,
) -> dict:
    """Ведёт фронт по часам, разворачивая азимут вслед за ветром.

    На каждом шаге контур сметается в направлении ветра этого часа и
    объединяется с уже пройденной площадью. Выпуклая оболочка не берётся:
    при развороте ветра гарь получается изогнутой, а не веерной.
    """
    lat0, lon0 = cluster.centroid
    points = list(zip(cluster.points["latitude"], cluster.points["longitude"], strict=True))
    hull = footprint(points, lat0, lon0, instrument)

    frames = [
        {
            "index": 0,
            "valid_at": series[0]["valid_at"],
            "hours": 0.0,
            "step_km": 0.0,
            "total_km": 0.0,
            "ros_kmh": 0.0,
            "bearing": series[0]["spread_bearing"],
            "speed_kmh": series[0]["speed_kmh"],
            "to_label": series[0]["to_label"],
            "from_label": series[0]["from_label"],
            "area_ha": area_hectares(hull),
            "outline": polygon_to_latlon(hull, lat0, lon0),
        }
    ]

    burned = hull
    head = hull
    total_km = 0.0

    for index, weather in enumerate(series[1:], start=1):
        rate = grass_rate_of_spread(
            weather["speed_kmh"],
            weather["temperature_c"],
            weather["humidity_pct"],
            curing_pct,
        )
        lb_ratio = length_to_breadth(weather["speed_kmh"])

        swept, distance = project_front(
            head, weather["spread_bearing"], rate["ros_kmh"], step_hours, lb_ratio
        )
        # Упрощение держит размер файла в разумных пределах: допуск 150 м
        # много меньше размера пикселя VIIRS, на точность оценки не влияет.
        head = largest_polygon(swept.simplify(150.0).buffer(0))
        burned = largest_polygon(unary_union([burned, head]).simplify(150.0).buffer(0))
        total_km += distance / 1000.0

        frames.append(
            {
                "index": index,
                "valid_at": weather["valid_at"],
                "hours": index * step_hours,
                "step_km": distance / 1000.0,
                "total_km": total_km,
                "ros_kmh": rate["ros_kmh"],
                "bearing": weather["spread_bearing"],
                "speed_kmh": weather["speed_kmh"],
                "to_label": weather["to_label"],
                "from_label": weather["from_label"],
                "area_ha": area_hectares(burned),
                "outline": polygon_to_latlon(burned, lat0, lon0),
            }
        )

    initial = grass_rate_of_spread(
        series[0]["speed_kmh"],
        series[0]["temperature_c"],
        series[0]["humidity_pct"],
        curing_pct,
    )

    return {
        **initial,
        "lb_ratio": length_to_breadth(series[0]["speed_kmh"]),
        "current_area_ha": area_hectares(hull),
        "current_outline": polygon_to_latlon(hull, lat0, lon0),
        "final_area_ha": area_hectares(burned),
        "final_outline": polygon_to_latlon(burned, lat0, lon0),
        "total_km": total_km,
        "frames": frames,
    }


def observed_footprint(points, lat0: float, lon0: float, instrument: str = "VIIRS") -> Polygon:
    """Контур по фактически снятым точкам одного пролёта."""
    coords = list(zip(points["latitude"], points["longitude"], strict=True))
    return footprint(coords, lat0, lon0, instrument)


def simulate_event(
    cluster,
    series: list[dict],
    timeline,
    curing_pct: float = 90.0,
    instrument: str = "VIIRS",
    horizon_hours: int = 12,
) -> dict:
    """Реконструкция очага вдоль общей временнóй шкалы события.

    Две фазы с разной опорой:

    Пока идут пролёты — ведущим является наблюдение. Между съёмками контур
    достраивается моделью, но каждый новый снимок переустанавливает голову
    фронта на фактическую. Иначе за несколько суток модель уезжает на сотни
    километров: она не знает ни про догорание, ни про тушение, ни про то,
    что огонь упёрся в дорогу.

    После последней съёмки — прогноз вперёд на horizon_hours. Дальше очаг
    считается отработавшим: он перестаёт расти и дымить.

    Кадры до обнаружения — None: показывать нечего.
    """
    from .validate import split_passes

    lat0, lon0 = cluster.centroid
    step_hours = float(timeline.step_hours)

    pass_frames: dict[int, Polygon] = {}
    for group in split_passes(cluster.points):
        stamp = group["acquired_at"].mean()
        index = timeline.index_of(stamp)
        if index is None:
            continue
        shape = observed_footprint(group, lat0, lon0, instrument)
        pass_frames[index] = (
            largest_polygon(unary_union([pass_frames[index], shape]))
            if index in pass_frames
            else shape
        )

    if not pass_frames:
        return {}

    ignition = min(pass_frames)
    last_pass = max(pass_frames)
    burnout = last_pass + int(round(horizon_hours / step_hours))

    weather = {}
    for step in series:
        index = timeline.index_of(pd.Timestamp(step["valid_at"]))
        if index is not None:
            weather[index] = step

    frames: list[dict | None] = [None] * len(timeline)

    # Пока идут съёмки, накопленная гарь строится ТОЛЬКО по наблюдениям.
    # Модельная голова между пролётами — временная: она показывает, куда
    # огонь идёт прямо сейчас, но в площадь не засчитывается, иначе за
    # несколько суток накапливаются сотни тысяч гектаров экстраполяции.
    observed = pass_frames[ignition]
    observed_only = pass_frames[ignition]
    head = close_front(observed)
    transient = None
    total_km = 0.0
    sources = set()

    for index in range(ignition, len(timeline)):
        step = weather.get(index)
        active = index <= burnout
        ros = 0.0
        moisture = 0.0

        if step and index > ignition and active:
            rate = grass_rate_of_spread(
                step["speed_kmh"], step["temperature_c"], step["humidity_pct"], curing_pct
            )
            swept, distance = project_front(
                head,
                step["spread_bearing"],
                rate["ros_kmh"],
                step_hours,
                length_to_breadth(step["speed_kmh"]),
            )
            head = largest_polygon(swept.simplify(150.0).buffer(0))
            if index <= last_pass:
                transient = head
            else:
                # Прогнозная фаза: наблюдений больше не будет, модель ведущая.
                observed = largest_polygon(
                    unary_union([observed, head]).simplify(150.0).buffer(0)
                )
                transient = None
            if index > last_pass:
                total_km += distance / 1000.0
            ros = rate["ros_kmh"]
            moisture = rate["fuel_moisture_pct"]
            sources.add(step.get("source", "ERA5"))

        # Свежий снимок — факт. Голова переустанавливается на наблюдённую,
        # накопленная гарь остаётся: выгоревшее не зарастает.
        if index in pass_frames and index > ignition:
            head = close_front(pass_frames[index])
            transient = None
            observed = unary_union([observed, head]).simplify(120.0).buffer(0)
            # Отдельно копим только снятое: это и есть гарь по наблюдениям,
            # без единого метра экстраполяции.
            observed_only = unary_union([observed_only, pass_frames[index]])

        frames[index] = {
            "index": index,
            "valid_at": timeline.stamps[index].isoformat(),
            "hours": (index - ignition) * step_hours,
            "total_km": total_km,
            "ros_kmh": ros,
            "fuel_moisture_pct": moisture,
            "speed_kmh": step["speed_kmh"] if step else 0.0,
            "bearing": step["spread_bearing"] if step else 0.0,
            "to_label": step["to_label"] if step else "—",
            "from_label": step["from_label"] if step else "—",
            "observed": index in pass_frames,
            "phase": "наблюдение" if index <= last_pass else ("прогноз" if active else "отработал"),
            "active": active,
            "area_ha": area_hectares(observed),
            "outline": polygon_to_latlon(
                largest_polygon(unary_union([observed, transient]))
                if transient is not None
                else observed,
                lat0,
                lon0,
            ),
        }

    live = [f for f in frames if f]
    return {
        "pixel_ceiling_ha": pixel_ceiling(len(cluster.points), instrument),
        "ignition_index": ignition,
        "ignition_at": timeline.stamps[ignition].isoformat(),
        "last_pass_index": last_pass,
        "burnout_index": burnout,
        "pass_indices": sorted(pass_frames),
        "sources": sorted(sources) or ["—"],
        "current_area_ha": area_hectares(pass_frames[ignition]),
        "current_outline": polygon_to_latlon(pass_frames[ignition], lat0, lon0),
        "observed_area_ha": area_hectares(observed_only),
        "final_area_ha": live[-1]["area_ha"],
        "final_outline": live[-1]["outline"],
        "total_km": total_km,
        "fuel_moisture_pct": next((f["fuel_moisture_pct"] for f in live if f["ros_kmh"] > 0), 0.0),
        "frames": frames,
    }


def build_forecast(
    cluster,
    weather: dict,
    horizons: tuple[float, ...] = (1.0, 3.0, 6.0),
    curing_pct: float = 90.0,
    instrument: str = "VIIRS",
) -> dict:
    """Полная оценка для одного очага: скорость, контуры, площади."""
    lat0, lon0 = cluster.centroid
    points = list(zip(cluster.points["latitude"], cluster.points["longitude"], strict=True))
    hull = footprint(points, lat0, lon0, instrument)

    rate = grass_rate_of_spread(
        weather["speed_kmh"],
        weather["temperature_c"],
        weather["humidity_pct"],
        curing_pct,
    )
    lb_ratio = length_to_breadth(weather["speed_kmh"])

    projections = []
    for hours in horizons:
        polygon, distance = project_front(
            hull, weather["spread_bearing"], rate["ros_kmh"], hours, lb_ratio
        )
        projections.append(
            {
                "hours": hours,
                "distance_km": distance / 1000.0,
                "area_ha": area_hectares(polygon),
                "outline": polygon_to_latlon(polygon, lat0, lon0),
            }
        )

    return {
        **rate,
        "lb_ratio": lb_ratio,
        "current_area_ha": area_hectares(hull),
        "current_outline": polygon_to_latlon(hull, lat0, lon0),
        "projections": projections,
    }
