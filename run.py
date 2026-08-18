"""Запуск анализа: термоточки FIRMS → очаги → снос по ветру → карта со шкалой.

Разбор прошедших пожаров: термоточки FIRMS плюс приземный ветер из
реанализа ERA5. Для прогноза на будущее нужен другой инструмент —
см. раздел «Родственный проект» в README.

Примеры:
    python run.py --key ВАШ_КЛЮЧ --validate      # окно 11–17 августа 2026
    python run.py --demo                      # без ключа, на синтетике
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from firms_spread import demo, era5, firms, validate
from firms_spread.clustering import build_clusters, cluster_hotspots
from firms_spread.event import build_event_timeline
from firms_spread.render import build_map
from firms_spread.spread import simulate_event

DEFAULT_START = "2026-08-11"

REGION_TITLES = {
    "karaganda": "Карагандинская область",
    "akmola": "Акмолинская область",
    "kostanay": "Костанайская область",
    "east-kz": "Восточный Казахстан",
    "kazakhstan": "Казахстан",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуализация сноса степных пожаров по ветру")
    parser.add_argument("--key", default=os.environ.get("FIRMS_MAP_KEY", ""), help="FIRMS MAP_KEY")
    parser.add_argument("--region", default="karaganda", choices=list(firms.REGIONS))
    parser.add_argument("--source", default="viirs_noaa20", choices=list(firms.SOURCES))
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="начало окна наблюдений (по умолчанию — окно событий августа 2026)",
    )
    parser.add_argument("--days", type=int, default=7,
                        help="длина окна; больше 5 суток собирается несколькими запросами")
    parser.add_argument("--horizon", type=int, default=12, help="сколько часов вести фронт")
    parser.add_argument("--step-hours", type=int, default=None,
                        help="шаг кадра, часов (по умолчанию подбирается)")
    parser.add_argument("--validate", action="store_true",
                        help="сверить снос с фактическим смещением очага")
    parser.add_argument("--eps-km", type=float, default=3.0, help="радиус связности DBSCAN")
    parser.add_argument("--min-points", type=int, default=5, help="минимум точек в очаге")
    parser.add_argument("--top", type=int, default=6, help="сколько крупнейших очагов считать")
    parser.add_argument("--curing", type=float, default=90.0, help="усыхание травостоя, %%")
    parser.add_argument("--confidence", default="n,h", help="уровни: l, n, h через запятую")
    parser.add_argument("--output", default="fire_spread.html")
    parser.add_argument("--check", action="store_true",
                        help="показать, какие продукты и даты доступны по ключу")
    parser.add_argument("--demo", action="store_true", help="синтетические данные без сети")
    parser.add_argument("--no-smoke", action="store_true", help="отключить анимацию дыма")
    parser.add_argument("--no-timeline", action="store_true", help="отключить временную шкалу")
    return parser.parse_args(argv)


def load_hotspots(args) -> pd.DataFrame:
    """Термоточки из FIRMS или из демонстрационного набора."""
    if args.demo:
        print("Режим демонстрации: синтетические очаги под Карагандой")
        return demo.synthetic_hotspots()

    if not args.key:
        sys.exit(
            "Нужен FIRMS MAP_KEY: передайте --key или задайте переменную FIRMS_MAP_KEY.\n"
            "Бесплатный ключ: https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
            "Посмотреть работу без ключа: python run.py --demo"
        )

    print(f"Запрашиваю FIRMS: {args.region}, {args.days} дн. с {args.start}")
    return firms.fetch_hotspots(
        map_key=args.key,
        source=args.source,
        bbox=firms.REGIONS[args.region],
        days=args.days,
        start=args.start,
    )


def weather_series(cluster, args, timeline) -> list[dict] | None:
    """Ряд ветра на всё окно события: от первой съёмки до конца горизонта.

    Раньше ряд начинался с последней съёмки, потому что фронт вели только
    вперёд. Теперь очаг реконструируется с момента обнаружения, поэтому
    ветер нужен и за прошедшие часы.
    """
    start = timeline.stamps[0]
    hours = (len(timeline) - 1) * timeline.step_hours

    if args.demo:
        return demo.synthetic_wind_series(
            cluster.centroid, start=start.isoformat(), hours=hours
        )
    try:
        return era5.fetch_wind_series(*cluster.centroid, start.to_pydatetime(), hours=hours)
    except era5.WeatherError as err:
        print(f"  {cluster.title}: ветер недоступен ({err}), очаг пропущен")
        return None


def observation_series(cluster, args) -> list[dict] | None:
    """Ряд ветра за окно наблюдений очага, а не за окно прогноза.

    Проверка сравнивает пролёты, которые уже состоялись, поэтому нужен
    ветер до момента последней съёмки, а не после.
    """
    earliest = cluster.points["acquired_at"].min()
    span = int((cluster.latest - earliest).total_seconds() // 3600) + 2

    if args.demo:
        return demo.synthetic_wind_series(cluster.centroid, start=earliest.isoformat(), hours=span)
    try:
        return era5.fetch_wind_series(*cluster.centroid, earliest.to_pydatetime(), hours=span)
    except era5.WeatherError as err:
        print(f"  {cluster.title}: реанализ за окно наблюдений недоступен ({err})")
        return None


def run_validation(clusters, args) -> None:
    """Сверяет расчётный снос с фактическим смещением очагов."""
    checked = 0

    for cluster in clusters:
        series = observation_series(cluster, args)
        if not series:
            continue

        pairs = validate.compare_with_observations(cluster, series, curing_pct=args.curing)
        if not pairs:
            continue

        print(f"\n{cluster.title}")
        print(validate.format_report(pairs, validate.summarize(pairs)))
        checked += 1

    if not checked:
        print("\n" + validate.format_report([], {}))


def show_availability(map_key: str) -> int:
    """Печатает покрытие продуктов FIRMS по этому ключу."""
    if not map_key:
        sys.exit("Для проверки нужен ключ: --key ВАШ_КЛЮЧ или FIRMS_MAP_KEY")

    try:
        table = firms.check_availability(map_key)
    except firms.FirmsError as err:
        sys.exit(f"Проверка не удалась.\n{err}")

    print("Доступные продукты и диапазоны дат:\n")
    print(table)
    print(
        "\nЗапрашиваемая дата должна попадать в диапазон нужного продукта. "
        "Оперативные (NRT) данные покрывают лишь недавний период; "
        "для более старых дат нужен архивный продукт."
    )
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.check:
        return show_availability(args.key)

    try:
        hotspots = load_hotspots(args)
    except firms.FirmsError as err:
        sys.exit(f"FIRMS отклонил запрос: {err}")

    levels = [x.strip() for x in args.confidence.split(",") if x.strip()]
    hotspots = firms.filter_confidence(hotspots, levels)
    print(f"Термоточек после фильтра достоверности: {len(hotspots)}")

    if hotspots.empty:
        sys.exit("Нет данных для анализа. Расширьте период или ослабьте фильтр.")

    labelled = cluster_hotspots(hotspots, eps_km=args.eps_km, min_samples=args.min_points)
    clusters = build_clusters(labelled, min_points=args.min_points)[: args.top]
    noise = int((labelled["cluster"] < 0).sum())
    print(f"Очагов выделено: {len(clusters)} (одиночных точек отброшено: {noise})")

    if not clusters:
        sys.exit("Плотных очагов не найдено. Увеличьте --eps-km или уменьшите --min-points.")

    notice = era5.availability_notice(max(c.latest for c in clusters).to_pydatetime())
    if notice and not args.demo:
        print(f"  Внимание: {notice}")

    instrument = "MODIS" if args.source == "modis" else "VIIRS"

    event = build_event_timeline(clusters, args.horizon, args.step_hours)
    print(
        f"Шкала события: {event.stamps[0]:%d.%m %H:%M} → {event.stamps[-1]:%d.%m %H:%M} UTC, "
        f"{len(event)} кадров по {event.step_hours} ч"
    )

    for cluster in clusters:
        series = weather_series(cluster, args, event)
        if not series:
            continue

        cluster.wind = series[0]
        cluster.spread = simulate_event(
            cluster,
            series,
            event,
            curing_pct=args.curing,
            instrument=instrument,
            horizon_hours=args.horizon,
        )
        if not cluster.spread:
            print(f"  {cluster.title}: пролёты вне шкалы, очаг пропущен")
            continue

        origin = pd.Timestamp(cluster.spread["ignition_at"])
        observed = cluster.spread["observed_area_ha"]
        ceiling = cluster.spread["pixel_ceiling_ha"]
        forecast = cluster.spread["final_area_ha"]

        print(
            f"  {cluster.title}: {cluster.count} точек, обнаружен {origin:%d.%m %H:%M}, "
            f"пролётов {len(cluster.spread['pass_indices'])}".replace(",", " ")
        )
        print(
            f"      гарь по снимкам: {observed:,.0f} га "
            f"(потолок по пикселям {ceiling:,.0f} га)".replace(",", " ")
        )
        print(
            f"      зона возможного распространения +{args.horizon} ч: "
            f"{forecast:,.0f} га, фронт {cluster.spread['total_km']:.1f} км "
            f"[{era5.source_summary(series)}]".replace(",", " ")
        )

    clusters = [c for c in clusters if c.spread]
    if not clusters:
        sys.exit("Ни для одного очага не удалось получить данные реанализа.")

    if args.validate:
        run_validation(clusters, args)

    moment = max(c.latest for c in clusters).strftime("%d.%m.%Y %H:%M UTC")
    path = build_map(
        clusters,
        hotspots,
        REGION_TITLES.get(args.region, args.region),
        moment,
        args.output,
        event=event,
        smoke=not args.no_smoke,
        timeline=not args.no_timeline,
    )
    print(f"\nКарта сохранена: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
