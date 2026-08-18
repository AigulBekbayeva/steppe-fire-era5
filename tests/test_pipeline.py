"""Тесты разбора данных FIRMS, кластеризации и обработки погоды."""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd
import pytest

from firms_spread.clustering import build_clusters, cluster_hotspots, haversine_km
from firms_spread.era5 import compass_label
from firms_spread.firms import (
    MAX_DAY_RANGE,
    FirmsError,
    _redact,
    fetch_hotspots,
    filter_confidence,
    normalize,
)

VIIRS_CSV = """country_id,latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
KAZ,49.85,73.10,340.1,0.4,0.36,2026-08-16,0720,N20,VIIRS,h,2.0NRT,295.0,12.4,D
KAZ,49.90,73.20,320.0,0.4,0.36,2026-08-16,0720,N20,VIIRS,l,2.0NRT,290.0,3.1,D
KAZ,49.95,73.30,355.0,0.4,0.36,2026-08-15,1330,N20,VIIRS,n,2.0NRT,301.0,88.0,D"""

MODIS_CSV = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight
49.85,73.10,330.5,1.1,1.0,2026-08-16,0810,Aqua,MODIS,95,6.1NRT,290.0,22.0,D
49.90,73.20,315.0,1.1,1.0,2026-08-16,0810,Aqua,MODIS,12,6.1NRT,288.0,4.0,D
49.95,73.30,344.0,1.1,1.0,2026-08-16,0810,Aqua,MODIS,55,6.1NRT,295.0,60.0,D"""


def read(csv: str) -> pd.DataFrame:
    return normalize(pd.read_csv(io.StringIO(csv)))


class TestNormalize:
    def test_viirs_brightness_mapped(self):
        df = read(VIIRS_CSV)
        assert df["brightness_k"].iloc[0] == pytest.approx(340.1)

    def test_modis_brightness_mapped(self):
        df = read(MODIS_CSV)
        assert df["brightness_k"].iloc[0] == pytest.approx(330.5)

    def test_timestamp_parsed_as_utc(self):
        df = read(VIIRS_CSV)
        stamp = df["acquired_at"].iloc[0]
        assert stamp.hour == 7 and stamp.minute == 20
        assert str(stamp.tz) == "UTC"

    def test_time_is_zero_padded(self):
        """acq_time приходит как 720 и должен читаться как 07:20, а не 72:0."""
        df = read(VIIRS_CSV)
        assert df["acq_time"].iloc[0] == "0720"

    def test_rows_without_coordinates_dropped(self):
        broken = VIIRS_CSV + "\nKAZ,,,,,,2026-08-16,0720,N20,VIIRS,h,2.0NRT,,,D"
        assert len(normalize(pd.read_csv(io.StringIO(broken)))) == 3


class TestConfidenceFilter:
    def test_letter_scale(self):
        df = filter_confidence(read(VIIRS_CSV), ["n", "h"])
        assert len(df) == 2
        assert "l" not in set(df["confidence"])

    def test_numeric_scale(self):
        """MODIS отдаёт проценты — фильтр должен работать и с ними."""
        df = filter_confidence(read(MODIS_CSV), ["n", "h"])
        assert len(df) == 2
        assert 12 not in set(df["confidence"])

    def test_empty_selection_returns_all(self):
        assert len(filter_confidence(read(VIIRS_CSV), [])) == 3

    def test_single_level(self):
        assert len(filter_confidence(read(VIIRS_CSV), ["h"])) == 1


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km(49.9, 73.5, 49.9, 73.5) == pytest.approx(0.0)

    def test_one_degree_of_latitude(self):
        assert haversine_km(49.0, 73.0, 50.0, 73.0) == pytest.approx(111.2, abs=0.5)

    def test_symmetric(self):
        forward = haversine_km(49.9, 73.5, 50.2, 74.1)
        backward = haversine_km(50.2, 74.1, 49.9, 73.5)
        assert forward == pytest.approx(backward)


class TestClustering:
    def build(self):
        """Две плотные группы в 40 км друг от друга плюс одиночная точка."""
        rows = []
        for lat, lon in [(49.90, 73.50), (49.60, 73.90)]:
            for i in range(8):
                rows.append(
                    {
                        "latitude": lat + i * 0.004,
                        "longitude": lon + i * 0.004,
                        "bright_ti4": 340.0,
                        "acq_date": "2026-08-16",
                        "acq_time": "0720",
                        "satellite": "N20",
                        "instrument": "VIIRS",
                        "confidence": "h",
                        "frp": 20.0 + i,
                        "daynight": "D",
                    }
                )
        rows.append(
            {
                "latitude": 48.0,
                "longitude": 70.0,
                "bright_ti4": 320.0,
                "acq_date": "2026-08-16",
                "acq_time": "0720",
                "satellite": "N20",
                "instrument": "VIIRS",
                "confidence": "h",
                "frp": 2.0,
                "daynight": "D",
            }
        )
        return normalize(pd.DataFrame(rows))

    def test_finds_both_groups(self):
        labelled = cluster_hotspots(self.build(), eps_km=3.0, min_samples=5)
        assert len(set(labelled[labelled["cluster"] >= 0]["cluster"])) == 2

    def test_isolated_point_is_noise(self):
        """Одиночные срабатывания — факелы и палы — не должны стать очагом."""
        labelled = cluster_hotspots(self.build(), eps_km=3.0, min_samples=5)
        lonely = labelled[labelled["latitude"] == 48.0]
        assert lonely["cluster"].iloc[0] == -1

    def test_clusters_sorted_by_power(self):
        labelled = cluster_hotspots(self.build(), eps_km=3.0, min_samples=5)
        clusters = build_clusters(labelled)
        powers = [c.total_frp for c in clusters]
        assert powers == sorted(powers, reverse=True)

    def test_labels_are_sequential(self):
        labelled = cluster_hotspots(self.build(), eps_km=3.0, min_samples=5)
        clusters = build_clusters(labelled)
        assert [c.label for c in clusters] == list(range(len(clusters)))

    def test_centroid_inside_cluster_bounds(self):
        labelled = cluster_hotspots(self.build(), eps_km=3.0, min_samples=5)
        for cluster in build_clusters(labelled):
            lat, lon = cluster.centroid
            assert cluster.points["latitude"].min() <= lat <= cluster.points["latitude"].max()
            assert cluster.points["longitude"].min() <= lon <= cluster.points["longitude"].max()

    def test_empty_input_survives(self):
        empty = pd.DataFrame(columns=["latitude", "longitude", "frp", "acquired_at"])
        assert build_clusters(cluster_hotspots(empty)) == []


class TestCompass:
    @pytest.mark.parametrize(
        "bearing,expected",
        [(0, "С"), (90, "В"), (180, "Ю"), (270, "З"), (45, "СВ"), (360, "С"), (-90, "З")],
    )
    def test_labels(self, bearing, expected):
        assert compass_label(bearing) == expected


class TestErrorReporting:
    """Диагностика отказов FIRMS: без тела ответа причину не найти."""

    def test_key_is_redacted(self):
        url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            "869177a1b0fe97f6fa7f21b4dc6d8978/VIIRS_NOAA20_NRT/66.0,46.5,79.0,51.5/7/2026-08-11"
        )
        hidden = _redact(url)
        assert "869177a1b0fe97f6fa7f21b4dc6d8978" not in hidden
        assert "VIIRS_NOAA20_NRT" in hidden

    def test_non_key_segments_survive(self):
        assert _redact("https://example.com/api/csv/abc/def") == "https://example.com/api/csv/abc/def"

    def test_error_carries_response_body(self, monkeypatch):
        """Раньше показывался только код ошибки, и причина терялась."""

        class Stub:
            status_code = 400
            text = "Invalid day range. Valid range 1-10"

        monkeypatch.setattr("firms_spread.firms.requests.get", lambda *a, **k: Stub())

        with pytest.raises(FirmsError) as excinfo:
            fetch_hotspots("k" * 32, "viirs_noaa20", (66.0, 46.5, 79.0, 51.5), 40, "2026-08-11")

        assert "Valid range 1-10" in str(excinfo.value)
        assert "400" in str(excinfo.value)

    def test_plain_text_rejection_is_surfaced(self, monkeypatch):
        class Stub:
            status_code = 200
            text = "Invalid MAP_KEY"

        monkeypatch.setattr("firms_spread.firms.requests.get", lambda *a, **k: Stub())

        with pytest.raises(FirmsError, match="Invalid MAP_KEY"):
            fetch_hotspots("k" * 32, "viirs_noaa20", (66.0, 46.5, 79.0, 51.5), 7, "2026-08-11")


class TestWindowChunking:
    """FIRMS отдаёт максимум пять суток за запрос — длинное окно режется."""

    def make_stub(self, calls):
        def fake_get(url, **kwargs):
            parts = url.rstrip("/").split("/")
            calls.append({"days": int(parts[-2]), "start": parts[-1]})

            class Response:
                status_code = 200
                text = VIIRS_CSV

            return Response()

        return fake_get

    def test_short_window_is_one_request(self, monkeypatch):
        calls = []
        monkeypatch.setattr("firms_spread.firms.requests.get", self.make_stub(calls))
        fetch_hotspots("k" * 32, "viirs_noaa20", (66.0, 46.5, 79.0, 51.5), 3, "2026-08-11")
        assert len(calls) == 1
        assert calls[0] == {"days": 3, "start": "2026-08-11"}

    def test_seven_days_split_into_five_plus_two(self, monkeypatch):
        calls = []
        monkeypatch.setattr("firms_spread.firms.requests.get", self.make_stub(calls))
        fetch_hotspots(
            "k" * 32, "viirs_noaa20", (66.0, 46.5, 79.0, 51.5), 7, "2026-08-11", verbose=False
        )
        assert [c["days"] for c in calls] == [5, 2]
        assert [c["start"] for c in calls] == ["2026-08-11", "2026-08-16"]

    def test_no_chunk_exceeds_the_limit(self, monkeypatch):
        calls = []
        monkeypatch.setattr("firms_spread.firms.requests.get", self.make_stub(calls))
        fetch_hotspots(
            "k" * 32, "viirs_noaa20", (66.0, 46.5, 79.0, 51.5), 30, "2026-08-01", verbose=False
        )
        assert all(c["days"] <= MAX_DAY_RANGE for c in calls)
        assert sum(c["days"] for c in calls) == 30

    def test_windows_are_contiguous(self, monkeypatch):
        calls = []
        monkeypatch.setattr("firms_spread.firms.requests.get", self.make_stub(calls))
        fetch_hotspots(
            "k" * 32, "viirs_noaa20", (66.0, 46.5, 79.0, 51.5), 12, "2026-08-01", verbose=False
        )
        for earlier, later in zip(calls, calls[1:], strict=False):
            expected = date.fromisoformat(earlier["start"]) + timedelta(days=earlier["days"])
            assert later["start"] == expected.isoformat()

    def test_duplicates_across_windows_removed(self, monkeypatch):
        """Стаб отдаёт одни и те же три строки на каждый запрос."""
        calls = []
        monkeypatch.setattr("firms_spread.firms.requests.get", self.make_stub(calls))
        result = fetch_hotspots(
            "k" * 32, "viirs_noaa20", (66.0, 46.5, 79.0, 51.5), 10, "2026-08-11", verbose=False
        )
        assert len(calls) == 2
        assert len(result) == 3

    def test_rejects_empty_period(self):
        with pytest.raises(FirmsError, match="не короче суток"):
            fetch_hotspots("k" * 32, "viirs_noaa20", (66.0, 46.5, 79.0, 51.5), 0, "2026-08-11")
