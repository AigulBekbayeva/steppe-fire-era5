"""Приземный ветер из реанализа ERA5.

Для разбора прошедших пожаров прогноз не годится: нужен реанализ, то есть
восстановленное поле погоды, усвоившее фактические наблюдения. ERA5 —
реанализ ECMWF с шагом 1 час и ячейкой около 31 км.

Берётся через архивный эндпоинт Open-Meteo: те же данные ERA5, но без
регистрации в CDS, без очереди на обработку и без разбора GRIB. Для работы
с исходными файлами ECMWF есть `cdsapi`, здесь он не нужен.

Ограничение: ERA5 публикуется с задержкой около пяти дней.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY = ["wind_speed_10m", "wind_direction_10m", "temperature_2m", "relative_humidity_2m"]

# Задержка публикации реанализа. Свежее этого срока данных ещё нет.
ERA5_LAG_DAYS = 5

COMPASS = [
    "С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ",
    "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ",
]


class WeatherError(RuntimeError):
    """Ошибка обращения к архиву реанализа."""


def compass_label(bearing: float) -> str:
    """Румб по азимуту в градусах."""
    return COMPASS[int((bearing % 360) / 22.5 + 0.5) % 16]


def _as_utc(moment: datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(moment)
    return stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")


def availability_notice(moment: datetime) -> str | None:
    """Предупреждение, если запрошенная дата ещё не покрыта реанализом."""
    edge = pd.Timestamp.now("UTC").tz_localize(None) - timedelta(days=ERA5_LAG_DAYS)
    if _as_utc(moment).tz_localize(None) > edge:
        return (
            f"ERA5 публикуется с задержкой ~{ERA5_LAG_DAYS} дней. "
            f"Для дат позже {edge.date()} данных может не быть — "
            f"возьмите более ранний период."
        )
    return None


def _fetch_hourly(lat: float, lon: float, start: pd.Timestamp, horizon_h: int, timeout: int):
    """Сырой почасовой ряд ERA5 с запасом в сутки по обе стороны."""
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "hourly": ",".join(HOURLY),
        "timezone": "UTC",
        "start_date": (start - timedelta(days=1)).date().isoformat(),
        "end_date": (start + timedelta(hours=horizon_h) + timedelta(days=1)).date().isoformat(),
    }

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as err:
        raise WeatherError(f"Архив ERA5 недоступен: {err}") from err

    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise WeatherError("Архив вернул пустой ряд — вероятно, дата вне покрытия")

    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.sort_values("time").reset_index(drop=True)


def _fetch_recent(lat: float, lon: float, start: pd.Timestamp, horizon_h: int, timeout: int):
    """Оперативная модель — для часов, которые реанализ ещё не покрыл."""
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "hourly": ",".join(HOURLY),
        "timezone": "UTC",
        "past_days": 7,
        "forecast_days": 3,
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as err:
        raise WeatherError(f"Оперативная модель недоступна: {err}") from err

    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise WeatherError("Оперативная модель вернула пустой ряд")

    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.sort_values("time").reset_index(drop=True)


def _row_to_weather(row, source: str = "ERA5") -> dict:
    """Приводит строку ряда к словарю с производными полями.

    wind_direction_10m — направление, ОТКУДА дует ветер (метеорологическая
    конвенция). Направление сноса огня противоположно.
    """
    from_direction = float(row["wind_direction_10m"] or 0.0)

    return {
        "speed_kmh": float(row["wind_speed_10m"] or 0.0),
        "from_direction": from_direction,
        "spread_bearing": (from_direction + 180.0) % 360.0,
        "from_label": compass_label(from_direction),
        "to_label": compass_label(from_direction + 180.0),
        "temperature_c": float(row["temperature_2m"] if row["temperature_2m"] is not None else 25.0),
        "humidity_pct": float(
            row["relative_humidity_2m"] if row["relative_humidity_2m"] is not None else 40.0
        ),
        "valid_at": row["time"].isoformat(),
        "source": source,
    }


def fetch_wind(lat: float, lon: float, moment: datetime, timeout: int = 30) -> dict:
    """Погода ERA5 в ближайший к moment час."""
    start = _as_utc(moment)
    frame = _fetch_hourly(lat, lon, start, 0, timeout)
    index = int((frame["time"] - start).abs().idxmin())
    return _row_to_weather(frame.loc[index])


def fetch_wind_series(
    lat: float,
    lon: float,
    start: datetime,
    hours: int = 12,
    timeout: int = 30,
    fill_gaps: bool = True,
) -> list[dict]:
    """Почасовой ряд от start на hours часов вперёд.

    Реанализ отстаёт от реального времени, поэтому свежие часы в архиве
    пустые. Если fill_gaps включён, они добираются из оперативной модели
    и помечаются в поле source — чтобы в отчёте было видно, где реанализ,
    а где нет.
    """
    anchor = _as_utc(start).floor("h")
    edge = anchor + timedelta(hours=hours)

    try:
        archive = _fetch_hourly(lat, lon, anchor, hours, timeout)
    except WeatherError:
        if not fill_gaps:
            raise
        archive = pd.DataFrame(columns=["time", *HOURLY])

    covered = {}
    for _, row in archive.iterrows():
        if anchor <= row["time"] <= edge and row["wind_direction_10m"] is not None:
            covered[row["time"]] = _row_to_weather(row, "ERA5")

    wanted = pd.date_range(anchor, edge, freq="h", tz="UTC")
    missing = [stamp for stamp in wanted if stamp not in covered]

    if missing and fill_gaps:
        try:
            recent = _fetch_recent(lat, lon, anchor, hours, timeout)
            lookup = {row["time"]: row for _, row in recent.iterrows()}
            for stamp in missing:
                row = lookup.get(stamp)
                if row is not None and row["wind_direction_10m"] is not None:
                    covered[stamp] = _row_to_weather(row, "прогноз")
        except WeatherError:
            pass

    series = [covered[stamp] for stamp in wanted if stamp in covered]
    if len(series) < 2:
        raise WeatherError(
            "Интервал не покрыт ни реанализом, ни оперативной моделью. "
            "Проверьте дату: ERA5 отстаёт от реального времени примерно на "
            f"{ERA5_LAG_DAYS} дней."
        )
    return series


def source_summary(series: list[dict]) -> str:
    """Из чего собран ряд: только реанализ или частично прогноз."""
    kinds = {step.get("source", "ERA5") for step in series}
    if kinds == {"ERA5"}:
        return "ERA5"
    reanalysis = sum(1 for s in series if s.get("source") == "ERA5")
    return f"ERA5 {reanalysis} ч + прогноз {len(series) - reanalysis} ч"
