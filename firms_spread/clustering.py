"""Выделение очагов: группировка близких термоточек в кластеры."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0088


@dataclass
class Cluster:
    """Очаг — связная группа термоточек."""

    label: int
    points: pd.DataFrame
    centroid: tuple[float, float]  # широта, долгота
    count: int
    total_frp: float
    max_frp: float
    span_km: float
    latest: pd.Timestamp
    wind: dict = field(default_factory=dict)
    spread: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return f"Очаг #{self.label + 1}"


def cluster_hotspots(
    df: pd.DataFrame,
    eps_km: float = 3.0,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Размечает термоточки метками кластеров. Шум получает метку -1.

    DBSCAN выбран потому, что число очагов заранее неизвестно, а форма
    у пожара произвольная — вытянутая вдоль фронта, а не круглая.
    """
    if df.empty:
        return df.assign(cluster=pd.Series(dtype=int))

    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    model = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    return df.assign(cluster=model.fit_predict(coords))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Расстояние между точками по большому кругу, км."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def build_clusters(df: pd.DataFrame, min_points: int = 5) -> list[Cluster]:
    """Собирает объекты очагов, отсортированные по суммарной мощности."""
    clusters: list[Cluster] = []

    for label, group in df[df["cluster"] >= 0].groupby("cluster"):
        if len(group) < min_points:
            continue

        # Центр взвешивается по мощности излучения: смещается к активной части.
        weights = group["frp"].fillna(1.0).clip(lower=0.1)
        lat = float(np.average(group["latitude"], weights=weights))
        lon = float(np.average(group["longitude"], weights=weights))

        span = haversine_km(
            group["latitude"].min(),
            group["longitude"].min(),
            group["latitude"].max(),
            group["longitude"].max(),
        )

        clusters.append(
            Cluster(
                label=int(label),
                points=group.reset_index(drop=True),
                centroid=(lat, lon),
                count=len(group),
                total_frp=float(group["frp"].fillna(0).sum()),
                max_frp=float(group["frp"].fillna(0).max()),
                span_km=span,
                latest=group["acquired_at"].max(),
            )
        )

    clusters.sort(key=lambda c: c.total_frp, reverse=True)
    for index, cluster in enumerate(clusters):
        cluster.label = index

    return clusters
