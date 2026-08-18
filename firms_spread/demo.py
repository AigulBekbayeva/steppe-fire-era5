"""Синтетические данные: позволяют проверить работу без ключа и без сети.

Геометрия очагов повторяет ситуацию 16 августа 2026 года к востоку от
Караганды — вытянутые вдоль фронта скопления, а не круглые пятна.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .firms import normalize

# Ядра очагов: широта, долгота, точек, азимут вытянутости, длина в градусах,
# день первого обнаружения (смещение от 11 августа) и число пролётов.
SEEDS = [
    (49.93, 73.55, 170, 55.0, 0.22, 0, 9),
    (49.62, 73.72, 120, 40.0, 0.16, 1, 7),
    (49.18, 73.35, 110, 70.0, 0.11, 2, 9),
    (50.15, 72.90, 80, 25.0, 0.08, 4, 8),
    (49.45, 74.30, 60, 85.0, 0.09, 5, 6),
]

FIRST_DAY = "2026-08-11"


def synthetic_hotspots(seed: int = 42) -> pd.DataFrame:
    """Правдоподобный набор термоточек за несколько суток.

    Очаги возникают в разные дни и снимаются несколькими пролётами,
    смещаясь по ветру — иначе реконструкцию событий не на чем показать.
    """
    rng = np.random.default_rng(seed)
    anchor = pd.Timestamp(FIRST_DAY, tz="UTC")
    rows = []

    for lat, lon, count, bearing, length, day_offset, sweeps in SEEDS:
        radians = np.radians(bearing)
        per_sweep = max(count // sweeps, 4)

        for sweep in range(sweeps):
            # Пролёты примерно раз в 12 часов, очаг всё это время движется.
            moment = anchor + pd.Timedelta(hours=day_offset * 24 + sweep * 8 + 7)
            # Снос за пролёт держим умеренным: за девять съёмок очаг иначе
            # уезжает на сотню километров, DBSCAN перестаёт связывать его
            # пролёты и разваливает один пожар на несколько.
            drift = sweep * 0.03

            along = rng.normal(0, length / 2.2, per_sweep)
            across = rng.normal(0, length / 9.0, per_sweep)

            lats = lat + drift * np.cos(radians) + along * np.cos(radians) - across * np.sin(radians)
            lons = lon + (
                drift * np.sin(radians) + along * np.sin(radians) + across * np.cos(radians)
            ) / np.cos(np.radians(lat))

            # Мощность падает от головы фронта к тылу и слабеет к концу горения.
            decay = 1.0 - 0.5 * sweep / max(sweeps - 1, 1)
            frp = np.clip(rng.gamma(2.2, 14.0, per_sweep) * decay * (1 + 0.5 * (along > 0)), 0.6, 900)

            for j in range(per_sweep):
                minute = int(rng.choice([12, 24, 36, 48]))
                rows.append(
                    {
                        "latitude": round(float(lats[j]), 5),
                        "longitude": round(float(lons[j]), 5),
                        "bright_ti4": round(float(rng.normal(338, 16)), 1),
                        "acq_date": moment.strftime("%Y-%m-%d"),
                        "acq_time": f"{moment.hour:02d}{minute:02d}",
                        "satellite": "N20",
                        "instrument": "VIIRS",
                        "confidence": rng.choice(["n", "h"], p=[0.35, 0.65]),
                        "frp": round(float(frp[j]), 1),
                        "daynight": "D",
                    }
                )

    # Разрозненные точки: факелы, палы стерни, ложные срабатывания.
    for _ in range(80):
        moment = anchor + pd.Timedelta(hours=int(rng.integers(0, 168)))
        rows.append(
            {
                "latitude": round(float(rng.uniform(47.0, 51.2)), 5),
                "longitude": round(float(rng.uniform(66.5, 78.5)), 5),
                "bright_ti4": round(float(rng.normal(322, 10)), 1),
                "acq_date": moment.strftime("%Y-%m-%d"),
                "acq_time": f"{moment.hour:02d}{int(rng.choice([6, 18, 30])):02d}",
                "satellite": "N20",
                "instrument": "VIIRS",
                "confidence": rng.choice(["l", "n", "h"], p=[0.5, 0.35, 0.15]),
                "frp": round(float(rng.gamma(1.4, 3.5)), 1),
                "daynight": "D",
            }
        )

    return normalize(pd.DataFrame(rows))


def synthetic_wind_series(
    centroid: tuple[float, float],
    start: str = "2026-08-11T07:00:00+00:00",
    hours: int = 12,
) -> list[dict]:
    """Ряд с разворотом ветра и дневным ходом температуры.

    Ветер поворачивает примерно на 60° за 12 часов — типичная ситуация,
    из-за которой фронт уходит не по прямой.
    """
    from datetime import datetime, timedelta

    from .era5 import compass_label

    lat, lon = centroid
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 10)))
    anchor = datetime.fromisoformat(start)

    base_direction = float(rng.uniform(200, 240))
    veer = float(rng.uniform(40, 75))
    base_speed = float(rng.uniform(9, 15))

    series = []
    for step in range(hours + 1):
        share = step / max(hours, 1)
        direction = (base_direction + veer * share) % 360.0
        speed = max(3.0, base_speed + 5.0 * np.sin(share * 3.14) + rng.normal(0, 1.2))
        temperature = 27.0 + 7.0 * np.sin(share * 2.2) + rng.normal(0, 0.6)
        humidity = max(9.0, 24.0 - 8.0 * np.sin(share * 2.2) + rng.normal(0, 1.5))

        series.append(
            {
                "speed_kmh": round(float(speed), 1),
                "from_direction": round(direction, 1),
                "spread_bearing": round((direction + 180.0) % 360.0, 1),
                "from_label": compass_label(direction),
                "to_label": compass_label(direction + 180.0),
                "temperature_c": round(float(temperature), 1),
                "humidity_pct": round(float(humidity), 1),
                "valid_at": (anchor + timedelta(hours=step)).isoformat(),
            }
        )

    return series


def synthetic_wind(centroid: tuple[float, float]) -> dict:
    """Юго-западный ветер — типичная летняя ситуация для региона."""
    from .era5 import compass_label

    lat, lon = centroid
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 10)))
    from_direction = float(rng.uniform(200, 250))

    return {
        "speed_kmh": float(rng.uniform(16, 30)),
        "from_direction": from_direction,
        "spread_bearing": (from_direction + 180.0) % 360.0,
        "from_label": compass_label(from_direction),
        "to_label": compass_label(from_direction + 180.0),
        "temperature_c": float(rng.uniform(28, 35)),
        "humidity_pct": float(rng.uniform(14, 26)),
        "observed_at": "2026-08-16T09:00:00+00:00",
    }
