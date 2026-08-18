"""Тесты покадрового ведения фронта и сборки итоговой карты."""

from __future__ import annotations

import pytest

from firms_spread import demo
from firms_spread.clustering import build_clusters, cluster_hotspots
from firms_spread.event import build_event_timeline
from firms_spread.render import build_map
from firms_spread.smoke import emission_rates
from firms_spread.spread import (
    area_hectares,
    close_front,
    footprint,
    parts_of,
    pixel_ceiling,
    project_front,
    simulate_event,
)
from firms_spread.timeline import build_detections, build_timeline_data, playback_pace

HORIZON = 12


@pytest.fixture(scope="module")
def event_and_clusters():
    """Полный прогон на синтетике: без сети и без ключа."""
    hotspots = cluster_hotspots(demo.synthetic_hotspots(), eps_km=3.0, min_samples=5)
    built = build_clusters(hotspots)[:4]
    timeline = build_event_timeline(built, HORIZON)

    span = (len(timeline) - 1) * timeline.step_hours
    for cluster in built:
        series = demo.synthetic_wind_series(
            cluster.centroid, start=timeline.stamps[0].isoformat(), hours=span
        )
        cluster.wind = series[0]
        cluster.spread = simulate_event(
            cluster, series, timeline, curing_pct=90.0, horizon_hours=HORIZON
        )

    return timeline, [c for c in built if c.spread]


@pytest.fixture(scope="module")
def clusters(event_and_clusters):
    return event_and_clusters[1]


@pytest.fixture(scope="module")
def timeline(event_and_clusters):
    return event_and_clusters[0]


@pytest.fixture(scope="module")
def hotspots():
    return demo.synthetic_hotspots()


class TestDemoData:
    def test_produces_several_clusters(self, clusters):
        assert len(clusters) >= 3

    def test_includes_scattered_noise(self):
        """В наборе должны быть одиночные точки, иначе фильтр шума не проверяется."""
        labelled = cluster_hotspots(demo.synthetic_hotspots(), eps_km=3.0, min_samples=5)
        assert (labelled["cluster"] < 0).sum() > 0

    def test_wind_series_veers(self):
        series = demo.synthetic_wind_series((49.9, 73.5), hours=HORIZON)
        turn = abs(series[-1]["from_direction"] - series[0]["from_direction"])
        assert turn > 20.0

    def test_wind_series_length(self):
        assert len(demo.synthetic_wind_series((49.9, 73.5), hours=HORIZON)) == HORIZON + 1


class TestSimulateTrack:
    def test_frames_span_whole_timeline(self, clusters, timeline):
        for cluster in clusters:
            assert len(cluster.spread["frames"]) == len(timeline)

    def test_nothing_before_detection(self, clusters):
        """Очаг не может появиться на карте раньше, чем его снял спутник."""
        for cluster in clusters:
            ignition = cluster.spread["ignition_index"]
            assert all(f is None for f in cluster.spread["frames"][:ignition])
            assert cluster.spread["frames"][ignition] is not None

    def test_first_visible_frame_is_the_detection(self, clusters):
        for cluster in clusters:
            first = cluster.spread["frames"][cluster.spread["ignition_index"]]
            assert first["hours"] == 0.0
            assert first["observed"] is True

    def test_area_never_shrinks(self, clusters):
        """Выгоревшее не зарастает: площадь может только расти."""
        for cluster in clusters:
            areas = [f["area_ha"] for f in cluster.spread["frames"] if f]
            assert all(b >= a - 1e-6 for a, b in zip(areas, areas[1:], strict=False))

    def test_distance_accumulates(self, clusters):
        for cluster in clusters:
            totals = [f["total_km"] for f in cluster.spread["frames"] if f]
            assert all(b >= a for a, b in zip(totals, totals[1:], strict=False))

    def test_growth_stops_after_burnout(self, clusters):
        """После горизонта прогноза очаг считается отработавшим."""
        for cluster in clusters:
            burnout = cluster.spread["burnout_index"]
            tail = [f for f in cluster.spread["frames"][burnout:] if f]
            if len(tail) > 1:
                assert tail[-1]["area_ha"] == pytest.approx(tail[0]["area_ha"])
                assert all(f["active"] is False for f in tail[1:])

    def test_area_stays_physical(self, clusters):
        """Без разделения наблюдений и модели площади уходили в миллионы га."""
        for cluster in clusters:
            assert cluster.spread["final_area_ha"] < 1_000_000

    def test_outlines_are_simplified(self, clusters):
        """Упрощение держит размер файла: без него вершин тысячи."""
        for cluster in clusters:
            for frame in cluster.spread["frames"]:
                if frame is None:
                    continue
                vertices = sum(len(r) for part in frame["outline"] for r in part)
                assert vertices < 4000

    def test_outlines_are_closed_rings(self, clusters):
        """Контур многосвязный: список частей, каждая — список колец."""
        for cluster in clusters:
            for frame in cluster.spread["frames"]:
                if frame is None:
                    continue
                assert len(frame["outline"]) >= 1
                for part in frame["outline"]:
                    for ring in part:
                        assert ring[0] == ring[-1]
                        assert len(ring) >= 4

    def test_passes_are_marked_observed(self, clusters):
        for cluster in clusters:
            marked = {
                f["index"] for f in cluster.spread["frames"] if f and f["observed"]
            }
            assert marked == set(cluster.spread["pass_indices"])


class TestTimelineData:
    def test_labels_match_frame_count(self, clusters, timeline):
        labels, tracks = build_timeline_data(clusters, timeline, emission_rates(clusters))
        assert len(labels) == len(timeline)
        assert all(len(track["frames"]) == len(labels) for track in tracks)

    def test_hours_follow_step(self, clusters, timeline):
        labels, _ = build_timeline_data(clusters, timeline, emission_rates(clusters))
        expected = [i * timeline.step_hours for i in range(len(timeline))]
        assert [label["hours"] for label in labels] == expected

    def test_track_per_cluster(self, clusters, timeline):
        _, tracks = build_timeline_data(clusters, timeline, emission_rates(clusters))
        assert len(tracks) == len(clusters)

    def test_hidden_frames_stay_null(self, clusters, timeline):
        """До обнаружения кадр должен быть null, иначе очаг мигнёт заранее."""
        _, tracks = build_timeline_data(clusters, timeline, emission_rates(clusters))
        for track, cluster in zip(tracks, clusters, strict=True):
            ignition = cluster.spread["ignition_index"]
            assert all(f is None for f in track["frames"][:ignition])
            assert track["frames"][ignition] is not None


class TestDetections:
    def test_every_hotspot_gets_a_frame(self, hotspots, timeline):
        points = build_detections(hotspots, timeline)
        assert len(points) > 0
        assert all(0 <= p[2] < len(timeline) for p in points)

    def test_detections_spread_over_time(self, hotspots, timeline):
        """Термоточки должны накапливаться, а не появляться одним кадром."""
        frames = {p[2] for p in build_detections(hotspots, timeline)}
        assert len(frames) > 5

    def test_coordinates_preserved(self, hotspots, timeline):
        points = build_detections(hotspots, timeline)
        lats = [p[0] for p in points]
        assert min(lats) == pytest.approx(hotspots["latitude"].min(), abs=0.001)


class TestEmissionRates:
    def test_one_rate_per_cluster(self, clusters):
        assert len(emission_rates(clusters)) == len(clusters)

    def test_scales_with_power(self, clusters):
        rates = emission_rates(clusters)
        assert rates[0] >= rates[-1]

    def test_bounded(self, clusters):
        assert all(6.0 <= rate <= 32.0 for rate in emission_rates(clusters))


class TestBuildMap:
    def test_writes_html(self, clusters, hotspots, timeline, tmp_path):
        output = tmp_path / "map.html"
        build_map(clusters, hotspots, "Тест", "16.08.2026 09:00 UTC", str(output), event=timeline)
        assert output.exists()
        assert output.stat().st_size > 50_000

    def test_contains_all_controls(self, clusters, hotspots, timeline, tmp_path):
        output = tmp_path / "map.html"
        build_map(clusters, hotspots, "Тест", "16.08.2026 09:00 UTC", str(output), event=timeline)
        html = output.read_text()

        assert 'id="timeline-slider"' in html
        assert 'id="timeline-play"' in html
        assert "smokeControl" in html
        assert "window.smokeControl.setFrame" in html
        assert "syncHotspots" in html
        assert "Индикативная оценка" in html

    def test_flags_disable_layers(self, clusters, hotspots, timeline, tmp_path):
        output = tmp_path / "plain.html"
        build_map(
            clusters,
            hotspots,
            "Тест",
            "16.08.2026 09:00 UTC",
            str(output),
            event=timeline,
            smoke=False,
            timeline=False,
        )
        html = output.read_text()

        assert "smokeControl" not in html
        assert 'id="timeline-slider"' not in html


class TestPlaybackPace:
    def test_long_timelines_speed_up(self):
        """Иначе неделя наблюдений проигрывается почти минуту."""
        assert playback_pace(87) < playback_pace(30)

    def test_full_pass_fits_the_target(self):
        for count in (40, 87, 150):
            duration = count * playback_pace(count) / 1000
            assert 15 <= duration <= 25

    def test_short_timelines_stay_readable(self):
        assert playback_pace(8) == 450 or playback_pace(8) >= 500

    def test_bounded(self):
        for count in (1, 2, 500, 5000):
            assert 120 <= playback_pace(count) <= 600


class TestAreaSanity:
    """Площадь гари не может превысить число снятых пикселей на их площадь."""

    def test_observed_area_under_pixel_ceiling(self, clusters):
        for cluster in clusters:
            assert cluster.spread["observed_area_ha"] <= cluster.spread["pixel_ceiling_ha"] * 1.02

    def test_ceiling_matches_point_count(self, clusters):
        for cluster in clusters:
            expected = pixel_ceiling(cluster.count, "VIIRS")
            assert cluster.spread["pixel_ceiling_ha"] == pytest.approx(expected)

    def test_scattered_points_do_not_fill_the_gaps(self):
        """Выпуклая оболочка раздувала площадь на порядки — так быть не должно."""
        spread_out = [(49.9 + i * 0.05, 73.5 + i * 0.08) for i in range(8)]
        contour = footprint(spread_out, 49.9, 73.5, "VIIRS")
        assert area_hectares(contour) <= pixel_ceiling(8, "VIIRS") * 1.02

    def test_disjoint_points_stay_disjoint(self):
        far_apart = [(49.0, 73.0), (50.0, 74.0)]
        contour = footprint(far_apart, 49.5, 73.5, "VIIRS")
        assert len(parts_of(contour)) == 2

    def test_adjacent_pixels_merge(self):
        touching = [(49.90, 73.50), (49.9015, 73.50)]
        contour = footprint(touching, 49.9, 73.5, "VIIRS")
        assert len(parts_of(contour)) == 1

    def test_closing_merges_nearby_pixels(self):
        """Прогноз от каждого пикселя порознь раздувал зону на порядок."""
        scattered = [(49.90 + i * 0.008, 73.50) for i in range(6)]
        contour = footprint(scattered, 49.9, 73.5, "VIIRS")
        assert len(parts_of(contour)) > 1
        assert len(parts_of(close_front(contour))) == 1

    def test_front_sweep_keeps_parts_separate(self):
        """Общая выпуклая оболочка склеивала разнесённые пятна в один массив."""
        far_apart = [(49.0, 73.0), (50.0, 74.0)]
        contour = footprint(far_apart, 49.5, 73.5, "VIIRS")
        swept, _ = project_front(contour, 45.0, 2.0, 1.0, 4.0)
        assert len(parts_of(swept)) == 2


class TestOutlineDeduplication:
    """После догорания контур замирает — повторы не должны раздувать файл."""

    def test_repeats_are_marked_null(self, clusters, timeline):
        _, tracks = build_timeline_data(clusters, timeline, emission_rates(clusters))
        repeated = sum(
            1 for t in tracks for f in t["frames"] if f and f["outline"] is None
        )
        assert repeated > 0

    def test_first_visible_frame_always_carries_geometry(self, clusters, timeline):
        _, tracks = build_timeline_data(clusters, timeline, emission_rates(clusters))
        for track in tracks:
            first = next(f for f in track["frames"] if f)
            assert first["outline"] is not None

    def test_area_is_kept_on_repeated_frames(self, clusters, timeline):
        """Площадь нужна в сводке на каждом кадре, даже когда контур не менялся."""
        _, tracks = build_timeline_data(clusters, timeline, emission_rates(clusters))
        for track in tracks:
            for frame in track["frames"]:
                if frame and frame["outline"] is None:
                    assert frame["area_ha"] >= 0

    def test_null_only_follows_a_real_outline(self, clusters, timeline):
        _, tracks = build_timeline_data(clusters, timeline, emission_rates(clusters))
        for track in tracks:
            seen = False
            for frame in track["frames"]:
                if not frame:
                    continue
                if frame["outline"] is not None:
                    seen = True
                else:
                    assert seen, "повтор не может идти раньше первого контура"


class TestSmokeContract:
    """Дыму нужен плоский список вершин, а контур стал многосвязным.

    Рассогласование этих двух форматов один раз уже погасило анимацию:
    частицы получали массив вместо широты и координаты обращались в NaN.
    """

    def render(self, clusters, hotspots, timeline, tmp_path):
        output = tmp_path / "smoke.html"
        build_map(
            clusters, hotspots, "Тест", "17.08.2026 20:00 UTC", str(output), event=timeline
        )
        return output.read_text()

    def test_outline_is_flattened_for_smoke(self, clusters, hotspots, timeline, tmp_path):
        html = self.render(clusters, hotspots, timeline, tmp_path)
        assert "function edgeVertices" in html
        assert "outline: edgeVertices(step)" in html

    def test_raw_outline_is_never_passed_through(self, clusters, hotspots, timeline, tmp_path):
        html = self.render(clusters, hotspots, timeline, tmp_path)
        assert "outline: step.outline," not in html

    def test_smoke_reads_vertices_as_pairs(self, clusters, hotspots, timeline, tmp_path):
        html = self.render(clusters, hotspots, timeline, tmp_path)
        assert "lat: node[0]" in html
        assert "lon: node[1]" in html

    def test_flattening_yields_coordinate_pairs(self, clusters):
        """Та же операция, что делает edgeVertices в браузере."""
        for cluster in clusters:
            frame = next(f for f in cluster.spread["frames"] if f)
            vertices = [v for part in frame["outline"] for ring in part for v in ring]
            assert vertices
            assert all(len(v) == 2 for v in vertices)
            assert all(isinstance(c, float) for v in vertices for c in v)


class TestVisualEncoding:
    """Точки окрашиваются по мощности, дым и подложка — в тёмной гамме."""

    def test_detections_carry_power(self, hotspots, timeline):
        points = build_detections(hotspots, timeline)
        assert all(len(p) == 4 for p in points)
        assert any(p[3] > 0 for p in points)

    def test_power_is_numeric_and_non_negative(self, hotspots, timeline):
        for point in build_detections(hotspots, timeline):
            assert isinstance(point[3], float)
            assert point[3] >= 0

    def test_missing_power_becomes_zero(self, timeline):
        """Пропуск FRP не должен ломать окраску."""
        import pandas as pd

        frame = pd.DataFrame(
            {
                "latitude": [49.9],
                "longitude": [73.5],
                "frp": [None],
                "acquired_at": [timeline.stamps[0]],
            }
        )
        assert build_detections(frame, timeline)[0][3] == 0.0

    def test_power_spans_several_colour_steps(self, hotspots, timeline):
        """Одноцветная карта скрывает разницу между тлением и фронтом."""
        powers = [p[3] for p in build_detections(hotspots, timeline)]
        steps = {min(sum(p >= limit for limit in (4, 15, 30, 60)), 4) for p in powers}
        assert len(steps) >= 4

    def test_map_uses_dark_basemap_first(self, clusters, hotspots, timeline, tmp_path):
        output = tmp_path / "dark.html"
        build_map(clusters, hotspots, "Тест", "17.08 20:00", str(output), event=timeline)
        html = output.read_text()

        assert html.index("dark_nolabels") < html.index("World_Imagery")
        assert "function frpColor" in html
        assert "rgba(226,222,248," in html
