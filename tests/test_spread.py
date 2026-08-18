"""Тесты модели распространения травяного пожара."""

from __future__ import annotations

import math

import pytest

from firms_spread.spread import (
    area_hectares,
    curing_coefficient,
    fine_fuel_moisture,
    footprint,
    grass_rate_of_spread,
    length_to_breadth,
    moisture_coefficient,
    project_front,
    to_geographic,
    to_local,
)


class TestFuelMoisture:
    def test_rises_with_humidity(self):
        dry = fine_fuel_moisture(30.0, 15.0)
        damp = fine_fuel_moisture(30.0, 70.0)
        assert damp > dry

    def test_falls_with_temperature(self):
        cool = fine_fuel_moisture(15.0, 40.0)
        hot = fine_fuel_moisture(35.0, 40.0)
        assert hot < cool

    def test_never_below_one_percent(self):
        assert fine_fuel_moisture(60.0, 0.0) >= 1.0

    def test_matches_mcarthur_formula(self):
        # 9.58 - 0.205*30 + 0.138*20 = 6.19
        assert fine_fuel_moisture(30.0, 20.0) == pytest.approx(6.19, abs=0.01)


class TestCoefficients:
    def test_moisture_coefficient_decreases(self):
        values = [moisture_coefficient(m, 20.0) for m in (4, 8, 14, 20)]
        assert values == sorted(values, reverse=True)

    def test_moisture_coefficient_never_negative(self):
        assert moisture_coefficient(40.0, 20.0) >= 0.0

    def test_curing_coefficient_grows_with_drying(self):
        assert curing_coefficient(40.0) < curing_coefficient(70.0) < curing_coefficient(100.0)

    def test_green_grass_barely_carries_fire(self):
        assert curing_coefficient(30.0) < 0.05


class TestRateOfSpread:
    def test_grows_with_wind(self):
        speeds = [grass_rate_of_spread(u, 32, 20, 90)["ros_kmh"] for u in (0, 5, 15, 30)]
        assert speeds == sorted(speeds)

    def test_calm_wind_gives_near_zero(self):
        assert grass_rate_of_spread(0.0, 32, 20, 90)["ros_kmh"] < 0.1

    def test_reference_values(self):
        """Опорные значения зафиксированы, чтобы правки не сдвинули модель."""
        assert grass_rate_of_spread(10, 32, 20, 90)["ros_kmh"] == pytest.approx(1.98, abs=0.05)
        assert grass_rate_of_spread(20, 32, 20, 90)["ros_kmh"] == pytest.approx(4.09, abs=0.05)
        assert grass_rate_of_spread(30, 32, 20, 90)["ros_kmh"] == pytest.approx(5.97, abs=0.05)

    def test_wet_fuel_slows_fire(self):
        dry = grass_rate_of_spread(25, 32, 15, 90)["ros_kmh"]
        wet = grass_rate_of_spread(25, 20, 85, 90)["ros_kmh"]
        assert wet < dry / 2

    def test_green_grass_stops_spread(self):
        assert grass_rate_of_spread(25, 32, 20, 30)["ros_kmh"] < 0.1

    def test_never_negative(self):
        assert grass_rate_of_spread(5, 10, 100, 20)["ros_kmh"] >= 0.0


class TestLengthToBreadth:
    def test_calm_wind_gives_round_fire(self):
        assert length_to_breadth(0.5) == 1.0

    def test_grows_with_wind(self):
        assert length_to_breadth(5) < length_to_breadth(20) < length_to_breadth(45)

    def test_capped(self):
        assert length_to_breadth(500) <= 8.0


class TestProjection:
    @pytest.mark.parametrize(
        "lat,lon", [(49.95, 73.60), (50.20, 72.90), (46.10, 78.40), (51.40, 66.20)]
    )
    def test_roundtrip_is_lossless(self, lat, lon):
        lat0, lon0 = 49.9, 73.5
        x, y = to_local(lat, lon, lat0, lon0)
        back_lat, back_lon = to_geographic(x, y, lat0, lon0)
        assert back_lat == pytest.approx(lat, abs=1e-9)
        assert back_lon == pytest.approx(lon, abs=1e-9)

    def test_north_is_positive_y(self):
        _, y = to_local(50.0, 73.5, 49.9, 73.5)
        assert y > 0

    def test_east_is_positive_x(self):
        x, _ = to_local(49.9, 74.0, 49.9, 73.5)
        assert x > 0


class TestFootprint:
    def test_single_point_becomes_pixel_disc(self):
        hull = footprint([(49.9, 73.5)], 49.9, 73.5, "VIIRS")
        # Диск радиусом 187.5 м — это примерно 11 га.
        assert area_hectares(hull) == pytest.approx(11.0, rel=0.05)

    def test_modis_pixel_is_larger(self):
        viirs = footprint([(49.9, 73.5)], 49.9, 73.5, "VIIRS")
        modis = footprint([(49.9, 73.5)], 49.9, 73.5, "MODIS")
        assert area_hectares(modis) > area_hectares(viirs) * 5

    def test_multiple_points_stay_pixel_sized(self):
        """Раньше здесь бралась выпуклая оболочка и площадь раздувалась."""
        points = [(49.90, 73.50), (49.95, 73.55), (49.88, 73.58)]
        contour = footprint(points, 49.9, 73.5)
        assert contour.is_valid
        # Три разнесённых пикселя — это три диска, а не залитый треугольник.
        assert area_hectares(contour) == pytest.approx(3 * 11.04, rel=0.05)


class TestFrontProjection:
    def test_moves_along_bearing(self):
        hull = footprint([(49.9, 73.5)], 49.9, 73.5)
        for bearing in (0, 45, 90, 180, 270):
            polygon, _ = project_front(hull, bearing, 4.0, 3.0, 4.0)
            dx = polygon.centroid.x - hull.centroid.x
            dy = polygon.centroid.y - hull.centroid.y
            actual = math.degrees(math.atan2(dx, dy)) % 360
            # Угловая разность с учётом перехода через 360 градусов.
            error = abs((actual - bearing + 180) % 360 - 180)
            assert error < 2.0, f"азимут {bearing}: получено {actual}"

    def test_distance_matches_speed_and_time(self):
        hull = footprint([(49.9, 73.5)], 49.9, 73.5)
        _, distance = project_front(hull, 45.0, 5.0, 2.0, 4.0)
        assert distance == pytest.approx(10_000.0)

    def test_zero_speed_leaves_hull_untouched(self):
        hull = footprint([(49.9, 73.5)], 49.9, 73.5)
        polygon, distance = project_front(hull, 45.0, 0.0, 3.0, 4.0)
        assert distance == 0.0
        assert polygon.equals(hull)

    def test_area_only_grows(self):
        hull = footprint([(49.9, 73.5)], 49.9, 73.5)
        polygon, _ = project_front(hull, 30.0, 3.0, 2.0, 4.0)
        assert area_hectares(polygon) > area_hectares(hull)

    def test_result_is_valid_geometry(self):
        hull = footprint([(49.90, 73.50), (49.95, 73.55)], 49.9, 73.5)
        polygon, _ = project_front(hull, 120.0, 6.0, 4.0, 3.0)
        assert polygon.is_valid
        assert polygon.exterior.is_closed
