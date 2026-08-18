"""Проверка модели по фактическому смещению очага.

Спутник снимает участок несколько раз в сутки. Между пролётами центр
тяжести термоточек смещается — это наблюдаемый факт, с которым можно
сравнить расчётный снос по ветру ERA5.

Важная оговорка: смещение центра тяжести — не то же самое, что скорость
головной части фронта. Центр отстаёт от головы, потому что позади она
остаётся гореть, и потому что часть очага к следующему пролёту уже
догорела и из данных пропала. Поэтому по расстоянию сравнение занижено
систематически, и осмысленно сравнивать прежде всего направление.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clustering import haversine_km
from .spread import grass_rate_of_spread

# Пролёты одного спутника разделены часами. Всё, что ближе, — один пролёт.
PASS_GAP_MINUTES = 90


@dataclass
class PassPair:
    """Пара последовательных пролётов и что между ними произошло."""

    from_time: pd.Timestamp
    to_time: pd.Timestamp
    hours: float
    points_from: int
    points_to: int
    observed_km: float
    observed_bearing: float
    modelled_bearing: float
    modelled_km: float
    wind_kmh: float

    @property
    def bearing_error(self) -> float:
        """Угловая невязка с учётом перехода через 360 градусов."""
        return abs((self.observed_bearing - self.modelled_bearing + 180) % 360 - 180)


def split_passes(points: pd.DataFrame) -> list[pd.DataFrame]:
    """Делит термоточки очага на отдельные пролёты по разрывам во времени."""
    ordered = points.sort_values("acquired_at").reset_index(drop=True)
    gaps = ordered["acquired_at"].diff() > pd.Timedelta(minutes=PASS_GAP_MINUTES)
    return [group for _, group in ordered.groupby(gaps.cumsum())]


def weighted_centroid(points: pd.DataFrame) -> tuple[float, float]:
    """Центр тяжести, взвешенный по мощности излучения."""
    weights = points["frp"].fillna(1.0).clip(lower=0.1)
    return (
        float(np.average(points["latitude"], weights=weights)),
        float(np.average(points["longitude"], weights=weights)),
    )


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Азимут от первой точки ко второй, градусы от севера."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def compare_with_observations(
    cluster,
    series: list[dict],
    curing_pct: float = 90.0,
    min_points: int = 4,
    min_shift_km: float = 1.0,
) -> list[PassPair]:
    """Сопоставляет расчётный снос с наблюдённым между пролётами.

    Пары с малым смещением отбрасываются: на коротком плече направление
    определяется шумом позиционирования пикселя, а не движением фронта.
    """
    passes = [p for p in split_passes(cluster.points) if len(p) >= min_points]
    if len(passes) < 2:
        return []

    lookup = {pd.Timestamp(step["valid_at"]).floor("h"): step for step in series}
    pairs: list[PassPair] = []

    for earlier, later in zip(passes, passes[1:], strict=False):
        lat1, lon1 = weighted_centroid(earlier)
        lat2, lon2 = weighted_centroid(later)

        distance = haversine_km(lat1, lon1, lat2, lon2)
        if distance < min_shift_km:
            continue

        from_time = earlier["acquired_at"].mean()
        to_time = later["acquired_at"].mean()
        hours = (to_time - from_time).total_seconds() / 3600.0
        if hours <= 0:
            continue

        # Ветер осредняется по часам, попавшим в промежуток между пролётами.
        window = [
            step
            for stamp, step in lookup.items()
            if from_time.floor("h") <= stamp <= to_time.ceil("h")
        ]
        if not window:
            continue

        bearings = np.radians([step["spread_bearing"] for step in window])
        mean_bearing = (
            math.degrees(math.atan2(np.sin(bearings).mean(), np.cos(bearings).mean())) + 360.0
        ) % 360.0
        mean_speed = float(np.mean([step["speed_kmh"] for step in window]))

        # Скорость фронта по погоде каждого часа промежутка.
        rates = [
            grass_rate_of_spread(
                step["speed_kmh"], step["temperature_c"], step["humidity_pct"], curing_pct
            )["ros_kmh"]
            for step in window
        ]

        pairs.append(
            PassPair(
                from_time=from_time,
                to_time=to_time,
                hours=hours,
                points_from=len(earlier),
                points_to=len(later),
                observed_km=distance,
                observed_bearing=bearing_between(lat1, lon1, lat2, lon2),
                modelled_bearing=mean_bearing,
                modelled_km=float(np.mean(rates)) * hours,
                wind_kmh=mean_speed,
            )
        )

    return pairs


def summarize(pairs: list[PassPair]) -> dict:
    """Сводка невязок по всем парам пролётов."""
    if not pairs:
        return {"pairs": 0}

    errors = [pair.bearing_error for pair in pairs]
    return {
        "pairs": len(pairs),
        "mean_bearing_error": float(np.mean(errors)),
        "median_bearing_error": float(np.median(errors)),
        "within_45deg": sum(1 for e in errors if e <= 45.0),
        "mean_observed_km": float(np.mean([p.observed_km for p in pairs])),
        "mean_hours": float(np.mean([p.hours for p in pairs])),
    }


def format_report(pairs: list[PassPair], summary: dict) -> str:
    """Текстовый отчёт для консоли."""
    if not pairs:
        return (
            "Проверка невозможна: нужно минимум два пролёта с заметным смещением очага.\n"
            "Расширьте период (--days) или возьмите более крупный очаг."
        )

    lines = [
        "",
        "Проверка по наблюдениям: расчётный снос против фактического смещения",
        "-" * 78,
        f"{'пролёты':<26} {'ч':>4} {'факт, км':>9} {'факт':>6} {'модель':>7} {'невязка':>8}",
    ]

    for pair in pairs:
        window = f"{pair.from_time:%d.%m %H:%M} → {pair.to_time:%H:%M}"
        lines.append(
            f"{window:<26} {pair.hours:>4.1f} {pair.observed_km:>9.1f} "
            f"{pair.observed_bearing:>5.0f}° {pair.modelled_bearing:>6.0f}° "
            f"{pair.bearing_error:>7.0f}°"
        )

    lines += [
        "-" * 78,
        f"Пар пролётов: {summary['pairs']}, "
        f"средняя невязка направления: {summary['mean_bearing_error']:.0f}°, "
        f"медианная: {summary['median_bearing_error']:.0f}°",
        f"В пределах 45°: {summary['within_45deg']} из {summary['pairs']}",
        "",
        "Смещение центра тяжести отстаёт от головной части фронта, поэтому",
        "сравнивать по расстоянию некорректно — значимо только направление.",
    ]
    return "\n".join(lines)
