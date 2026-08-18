"""Тесты сверки расчётного сноса с фактическим смещением очага."""

from __future__ import annotations

import pandas as pd
import pytest

from firms_spread import demo
from firms_spread.clustering import build_clusters, cluster_hotspots
from firms_spread.era5 import availability_notice
from firms_spread.validate import (
    PassPair,
    bearing_between,
    compare_with_observations,
    split_passes,
    summarize,
    weighted_centroid,
)


@pytest.fixture(scope="module")
def cluster():
    hotspots = cluster_hotspots(demo.synthetic_hotspots(), eps_km=3.0, min_samples=5)
    return build_clusters(hotspots)[0]


class TestBearingBetween:
    @pytest.mark.parametrize(
        "lat2,lon2,expected",
        [(50.9, 73.5, 0), (49.9, 74.5, 90), (48.9, 73.5, 180), (49.9, 72.5, 270)],
    )
    def test_cardinal_directions(self, lat2, lon2, expected):
        assert bearing_between(49.9, 73.5, lat2, lon2) == pytest.approx(expected, abs=1.0)

    def test_northeast(self):
        assert bearing_between(49.9, 73.5, 50.4, 74.28) == pytest.approx(45, abs=3.0)

    def test_always_in_range(self):
        for lat in (48.0, 50.0, 52.0):
            for lon in (70.0, 73.0, 76.0):
                assert 0 <= bearing_between(49.9, 73.5, lat, lon) < 360


class TestSplitPasses:
    def test_finds_separate_overpasses(self, cluster):
        """Число пролётов должно совпасть с числом моментов съёмки в данных."""
        moments = cluster.points["acquired_at"].dt.floor("h").nunique()
        assert len(split_passes(cluster.points)) == moments
        assert len(split_passes(cluster.points)) > 1

    def test_passes_are_chronological(self, cluster):
        passes = split_passes(cluster.points)
        times = [p["acquired_at"].mean() for p in passes]
        assert times == sorted(times)

    def test_no_points_lost(self, cluster):
        passes = split_passes(cluster.points)
        assert sum(len(p) for p in passes) == len(cluster.points)

    def test_single_pass_stays_whole(self):
        frame = pd.DataFrame(
            {
                "latitude": [49.9, 49.91, 49.92],
                "longitude": [73.5, 73.51, 73.52],
                "frp": [10.0, 12.0, 14.0],
                "acquired_at": pd.to_datetime(
                    ["2026-08-16 07:20", "2026-08-16 07:24", "2026-08-16 07:26"], utc=True
                ),
            }
        )
        assert len(split_passes(frame)) == 1


class TestWeightedCentroid:
    def test_pulled_toward_high_power(self):
        frame = pd.DataFrame(
            {
                "latitude": [49.0, 50.0],
                "longitude": [73.0, 73.0],
                "frp": [1.0, 99.0],
                "acquired_at": pd.to_datetime(["2026-08-16 07:20"] * 2, utc=True),
            }
        )
        lat, _ = weighted_centroid(frame)
        assert lat > 49.9

    def test_missing_power_does_not_break(self):
        frame = pd.DataFrame(
            {
                "latitude": [49.0, 50.0],
                "longitude": [73.0, 73.0],
                "frp": [None, None],
                "acquired_at": pd.to_datetime(["2026-08-16 07:20"] * 2, utc=True),
            }
        )
        lat, _ = weighted_centroid(frame)
        assert lat == pytest.approx(49.5)


class TestPassPair:
    def build(self, observed, modelled):
        stamp = pd.Timestamp("2026-08-16 07:00", tz="UTC")
        return PassPair(
            from_time=stamp,
            to_time=stamp + pd.Timedelta(hours=5),
            hours=5.0,
            points_from=10,
            points_to=12,
            observed_km=6.0,
            observed_bearing=observed,
            modelled_bearing=modelled,
            modelled_km=8.0,
            wind_kmh=20.0,
        )

    def test_exact_match_is_zero(self):
        assert self.build(45.0, 45.0).bearing_error == 0.0

    def test_wraps_around_north(self):
        """350° и 10° различаются на 20°, а не на 340°."""
        assert self.build(350.0, 10.0).bearing_error == pytest.approx(20.0)

    def test_opposite_is_180(self):
        assert self.build(0.0, 180.0).bearing_error == pytest.approx(180.0)

    def test_error_never_exceeds_180(self):
        for observed in range(0, 360, 15):
            for modelled in range(0, 360, 15):
                assert 0 <= self.build(observed, modelled).bearing_error <= 180.0


class TestCompareWithObservations:
    def test_produces_pairs(self, cluster):
        series = demo.synthetic_wind_series(
            cluster.centroid,
            start=cluster.points["acquired_at"].min().isoformat(),
            hours=14,
        )
        pairs = compare_with_observations(cluster, series)
        assert len(pairs) >= 1

    def test_pairs_have_positive_duration(self, cluster):
        series = demo.synthetic_wind_series(
            cluster.centroid,
            start=cluster.points["acquired_at"].min().isoformat(),
            hours=14,
        )
        for pair in compare_with_observations(cluster, series):
            assert pair.hours > 0
            assert pair.observed_km >= 1.0

    def test_short_series_yields_nothing(self, cluster):
        """Без покрытия ветром сравнивать не с чем — но и падать нельзя."""
        series = demo.synthetic_wind_series(
            cluster.centroid, start="2020-01-01T00:00:00+00:00", hours=2
        )
        assert compare_with_observations(cluster, series) == []


class TestSummarize:
    def test_empty_input(self):
        assert summarize([])["pairs"] == 0

    def test_counts_within_threshold(self):
        stamp = pd.Timestamp("2026-08-16 07:00", tz="UTC")

        def pair(observed, modelled):
            return PassPair(
                from_time=stamp,
                to_time=stamp + pd.Timedelta(hours=4),
                hours=4.0,
                points_from=8,
                points_to=9,
                observed_km=5.0,
                observed_bearing=observed,
                modelled_bearing=modelled,
                modelled_km=7.0,
                wind_kmh=18.0,
            )

        result = summarize([pair(40, 50), pair(10, 100), pair(200, 210)])
        assert result["pairs"] == 3
        assert result["within_45deg"] == 2
        assert result["mean_bearing_error"] == pytest.approx(36.67, abs=0.1)


class TestEra5Availability:
    def test_recent_date_warns(self):
        assert availability_notice(pd.Timestamp.now("UTC").to_pydatetime()) is not None

    def test_old_date_is_silent(self):
        old = (pd.Timestamp.now("UTC") - pd.Timedelta(days=60)).to_pydatetime()
        assert availability_notice(old) is None
