"""Загрузка термоточек из NASA FIRMS API."""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd
import requests

# Жёсткое ограничение сервиса: за один запрос отдаётся не больше пяти суток.
# Более длинное окно приходится собирать из нескольких запросов.
MAX_DAY_RANGE = 5

BASE = "https://firms.modaps.eosdis.nasa.gov/api"

AREA_URL = BASE + "/area/csv/{key}/{source}/{bbox}/{days}/{start}"
AVAILABILITY_URL = BASE + "/data_availability/csv/{key}/all"

# NRT — оперативные данные за недавний период, SP — архив стандартной
# обработки за более ранние даты. Точные границы покрытия у каждого свои,
# смотрите `run.py --check`.
SOURCES = {
    "viirs_noaa20": "VIIRS_NOAA20_NRT",
    "viirs_noaa21": "VIIRS_NOAA21_NRT",
    "viirs_snpp": "VIIRS_SNPP_NRT",
    "modis": "MODIS_NRT",
    "viirs_noaa20_sp": "VIIRS_NOAA20_SP",
    "viirs_snpp_sp": "VIIRS_SNPP_SP",
    "modis_sp": "MODIS_SP",
}

# Границы областей: запад, юг, восток, север.
REGIONS = {
    "karaganda": (66.0, 46.5, 79.0, 51.5),
    "akmola": (64.0, 49.5, 74.0, 54.0),
    "kostanay": (59.0, 49.0, 68.0, 54.5),
    "east-kz": (76.0, 46.5, 87.5, 51.5),
    "kazakhstan": (46.0, 40.0, 88.0, 56.0),
}


class FirmsError(RuntimeError):
    """Ошибка обращения к FIRMS."""


def _redact(url: str) -> str:
    """Прячет ключ: сообщения об ошибках часто попадают в чужие руки."""
    parts = url.split("/")
    for index, part in enumerate(parts):
        if len(part) == 32 and all(c in "0123456789abcdef" for c in part.lower()):
            parts[index] = "<ключ скрыт>"
    return "/".join(parts)


def check_availability(map_key: str, timeout: int = 30) -> str:
    """Какие продукты и за какие даты доступны по этому ключу.

    Самая частая причина отказа — дата вне покрытия выбранного продукта
    или неверный ключ. Этот запрос отвечает на оба вопроса сразу.
    """
    url = AVAILABILITY_URL.format(key=map_key)
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as err:
        raise FirmsError(f"FIRMS недоступен: {err}") from err

    text = response.text.strip()
    if response.status_code >= 400 or not text:
        raise FirmsError(
            f"HTTP {response.status_code}. Ответ: {text[:400] or '(пусто)'}\n"
            "Скорее всего ключ неверен или ещё не активирован."
        )
    return text


def _fetch_window(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    days: int,
    start: str,
    timeout: int,
) -> pd.DataFrame:
    """Один запрос к FIRMS — не длиннее MAX_DAY_RANGE суток."""
    url = AREA_URL.format(
        key=map_key,
        source=SOURCES.get(source, source),
        bbox=",".join(f"{c:.3f}" for c in bbox),
        days=days,
        start=start,
    )
    response = requests.get(url, timeout=timeout)
    text = response.text.strip()

    # FIRMS объясняет причину отказа в теле ответа, а не только кодом.
    # Без этого текста диагностировать 400 невозможно.
    if response.status_code >= 400:
        raise FirmsError(
            f"HTTP {response.status_code}. Ответ сервиса: {text[:400] or '(пусто)'}\n"
            f"Запрос: {_redact(url)}"
        )

    header = text.split("\n", 1)[0].lower() if text else ""
    if not text or "latitude" not in header:
        raise FirmsError(
            f"{text[:400] or 'FIRMS вернул пустой ответ'}\n"
            f"Запрос: {_redact(url)}"
        )

    return pd.read_csv(io.StringIO(text))


def fetch_hotspots(
    map_key: str,
    source: str,
    bbox: tuple[float, float, float, float],
    days: int,
    start: str,
    timeout: int = 60,
    verbose: bool = True,
) -> pd.DataFrame:
    """Термоточки за период. Длинное окно собирается из нескольких запросов."""
    if days < 1:
        raise FirmsError("Период должен быть не короче суток")

    cursor = date.fromisoformat(start)
    remaining = days
    chunks: list[pd.DataFrame] = []

    while remaining > 0:
        span = min(remaining, MAX_DAY_RANGE)
        if verbose and days > MAX_DAY_RANGE:
            print(f"  запрос {cursor.isoformat()} +{span} дн.")

        chunks.append(
            _fetch_window(map_key, source, bbox, span, cursor.isoformat(), timeout)
        )
        cursor += timedelta(days=span)
        remaining -= span

    combined = pd.concat(chunks, ignore_index=True)
    if combined.empty:
        return normalize(combined)

    # Границы окон могут перекрываться, одна и та же точка приходит дважды.
    keys = [c for c in ("latitude", "longitude", "acq_date", "acq_time", "satellite")
            if c in combined.columns]
    combined = combined.drop_duplicates(subset=keys).reset_index(drop=True)

    return normalize(combined)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит колонки VIIRS и MODIS к общей схеме."""
    df = df.copy()

    # Яркость: у VIIRS в bright_ti4, у MODIS в brightness.
    if "bright_ti4" in df.columns:
        df["brightness_k"] = df["bright_ti4"]
    elif "brightness" in df.columns:
        df["brightness_k"] = df["brightness"]
    else:
        df["brightness_k"] = pd.NA

    df["frp"] = pd.to_numeric(df.get("frp"), errors="coerce")
    df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
    df["acquired_at"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + df["acq_time"].str[:2] + ":" + df["acq_time"].str[2:],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
        utc=True,
    )
    return df.dropna(subset=["latitude", "longitude", "acquired_at"]).reset_index(drop=True)


def filter_confidence(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Фильтр достоверности. VIIRS отдаёт l/n/h, MODIS — проценты."""
    if "confidence" not in df.columns or not keep:
        return df

    conf = df["confidence"]
    if not pd.api.types.is_numeric_dtype(conf):
        mask = conf.astype(str).str.lower().isin(keep)
    else:
        bands = {"l": (0, 30), "n": (30, 80), "h": (80, 101)}
        mask = pd.Series(False, index=df.index)
        for level in keep:
            low, high = bands[level]
            mask |= conf.between(low, high, inclusive="left")

    return df[mask].reset_index(drop=True)
