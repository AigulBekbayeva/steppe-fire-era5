# Steppe Fire ERA5

[![CI](https://github.com/USERNAME/steppe-fire-era5/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/steppe-fire-era5/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Retrospective analysis of steppe wildfires: NASA FIRMS hotspots clustered into
fire complexes, driven forward with ERA5 reanalysis wind, and — the part that
matters — checked against how the fires actually moved between satellite
overpasses.

Built around the August 2026 steppe fires in Karaganda region, Kazakhstan.

**[▶ Live demo map](https://aigulbekbayeva.github.io/steppe-fire-era5/)** — synthetic
data, no API key needed.

---

## Why reanalysis, not forecast

For a fire that has already burned, forecast wind is the wrong input. ERA5 is
ECMWF's reanalysis — a reconstructed weather field that has assimilated the
observations actually recorded at the time, hourly, on a roughly 31 km grid.

It is pulled through Open-Meteo's archive endpoint: the same ERA5 fields,
without CDS registration, without the processing queue, without parsing GRIB.
If you need the native ECMWF files, use `cdsapi` — this project does not.

ERA5 is published with roughly a five-day lag. Hours the archive has not yet
covered are filled from the operational forecast model and tagged as such, so
each run reports what its wind actually came from — `[ERA5]` or
`[ERA5 84 ч + прогноз 12 ч]`. The mixture is visible rather than silent.

## Two phases, two sources of truth

The timeline spans the whole event, not just a forecast window. Play it and the
fires appear one by one as satellites find them, hotspots pile up pass by pass,
and burn outlines grow.

Which contour you are looking at depends on the phase, and the distinction
matters:

**While overpasses continue, observations lead.** Accumulated burned area is
built from observed footprints only. The model still runs between passes, but
its output is drawn as a transient leading edge and is *not* banked into the
area — and each new overpass resets the head to what was actually seen.

The first version banked it, and the result was instructive: modelled excursions
compounded across five days into 17 million hectares and 600 km of travel. The
model has no idea that fires burn out, that crews turn up, or that a front hits
a road. Over a few hours that hardly shows; over days it is nonsense.

**Past the last detection, the model leads** — for `--horizon` hours, after
which the fire is marked burnt out: it stops growing and stops smoking.

A week of observations at hourly resolution would be nearly two hundred frames,
so the frame size is chosen automatically to keep the count near a hundred, and
playback speed adapts so a full pass takes about twenty seconds regardless of
window length.

One thing to watch on long windows: a fire complex that migrates far enough
between overpasses can stop being spatially connected, and DBSCAN will split it
into separate fires. If a single fire appears as several, widen `--eps-km`.

## Validation: the point of the whole thing

A satellite images the same ground several times a day. Between overpasses the
fire's centre of mass moves, and that displacement is an observed fact — one you
can hold the model against.

```
Проверка по наблюдениям: расчётный снос против фактического смещения
------------------------------------------------------------------------------
пролёты                       ч  факт, км   факт  модель  невязка
16.08 06:26 → 11:27         5.0       4.1   231°    224°       7°
16.08 11:27 → 16:27         5.0       5.6    69°     89°      20°
------------------------------------------------------------------------------
```

Run it with `--validate`.

**Read the numbers carefully.** Centre-of-mass displacement is not head-fire
speed. The centre lags the head, because the ground behind the head keeps
burning, and because parts of the fire have burned out and dropped from the data
by the next pass. So distances are systematically understated and only
*direction* is a fair comparison. The report says so on every run — this is a
sanity check on wind-driven direction, not a skill score for the spread model.

## How it works

| Stage | What happens |
|---|---|
| **Fetch** | VIIRS (375 m) or MODIS (1 km) hotspots from the FIRMS API. The service caps a single request at five days, so longer windows are split and stitched |
| **Cluster** | DBSCAN on a haversine metric — fire count is unknown in advance and fire shapes are elongated, so not k-means |
| **Weather** | Hourly ERA5 reanalysis: wind, temperature, humidity |
| **Model** | Head-fire rate of spread per Cheney, Gould & Catchpole (1998), corrected for fuel moisture and grass curing |
| **Reconstruct** | One shared timeline for the whole event. Fires ignite when first detected, hotspots accumulate pass by pass, and each new overpass resets the modelled head to the observed footprint |
| **Forecast** | Past the last detection the model runs forward for `--horizon` hours, then the fire is treated as burnt out |
| **Validate** | Modelled bearing versus observed inter-overpass displacement |
| **Render** | Single-file HTML on a dark basemap: hotspots coloured by fire radiative power, growing burn outline, lavender smoke plume on canvas, time slider |

## Quickstart

```bash
pip install -r requirements.txt

# See it work with no key and no network
python run.py --demo --validate

# Real data — defaults to 11–17 August 2026, the Karaganda event window
export FIRMS_MAP_KEY=your_key
python run.py --validate

# Any other window
python run.py --start 2026-07-20 --days 6 --horizon 12
```

Get a free MAP_KEY instantly at
[firms.modaps.eosdis.nasa.gov/api/map_key](https://firms.modaps.eosdis.nasa.gov/api/map_key/).

Run `python run.py --key YOUR_KEY --check` first. It prints exactly which
products your key can reach and over what dates — the near-real-time products
cover only a recent window, while the `_sp` archive variants go back years, and
the boundary between them moves. Most failed runs are a date outside the chosen
product's range.

### Options

| Flag | Purpose | Default |
|---|---|---|
| `--region` | karaganda, akmola, kostanay, east-kz, kazakhstan | karaganda |
| `--source` | viirs_noaa20, viirs_noaa21, viirs_snpp, modis, plus `_sp` archive variants | viirs_noaa20 |
| `--start` / `--days` | window start and length; over 5 days is fetched in chunks | 2026-08-11 / 7 |
| `--horizon` | hours to model past the last detection | 12 |
| `--step-hours` | timeline frame size; auto-chosen to cap frame count | auto |
| `--validate` | compare modelled bearing with observed displacement | off |
| `--eps-km` | DBSCAN neighbourhood radius | 3.0 |
| `--min-points` | minimum hotspots to count as a fire complex | 5 |
| `--curing` | grass curing, % | 90 |
| `--confidence` | levels to keep: `l`, `n`, `h` | n,h |
| `--no-smoke` / `--no-timeline` | drop the animated layers | off |

## The spread model

```
U10 ≤ 5 km/h:  R = (0.054 + 0.269·U10) · Φm · Φc
U10 > 5 km/h:  R = (1.4 + 0.838·(U10−5)^0.844) · Φm · Φc
```

`Φm` corrects for fine dead fuel moisture, estimated from temperature and
humidity. `Φc` corrects for grass curing (Cruz et al., 2015). Fire ellipse
length-to-breadth is `1.1·U10^0.464`.

ERA5 reports the direction wind blows **from**; fire spreads the opposite way,
so `bearing = wind_direction + 180°`.

### Reference values

At 90% curing, 32 °C, 20% humidity — pinned in the test suite:

| Wind, km/h | Rate of spread, km/h |
|---|---|
| 5 | 0.59 |
| 10 | 1.98 |
| 20 | 4.09 |
| 30 | 5.97 |

Below roughly 50% curing the model correctly stops carrying fire.

## Limitations

**This is an indicative estimate, not an operational product.** It ignores
terrain, barriers such as rivers, roads and ploughed firebreaks, gustiness
within the hour, fuel load and patchiness, and firefighting.

The Cheney model was developed on Australian pasture and is applied here as the
closest published analogue for Eurasian steppe, without local calibration.

ERA5's 31 km grid cell is far coarser than a fire front. Local channelling,
slope winds and gust fronts are averaged away.

A FIRMS hotspot is a pixel with an elevated temperature, not a confirmed
wildfire: gas flares, stubble burning and false positives land in the same
dataset. Clustering discards isolated detections but cannot guarantee every
complex is a natural fire.

Two areas are reported and they mean different things.

**Burned area from imagery** is the union of detected pixels, each a disc the
size of the instrument footprint. It cannot exceed the number of detections
times the pixel area, and the run prints that ceiling next to it so the check is
visible. An earlier version took the convex hull of a cluster's hotspots, which
filled in every unburned gap between scattered detections and inflated the
figure by orders of magnitude.

**Spread envelope** is where fire could reach in the forecast window assuming no
suppression, no barriers and continuous fuel. It is not a prediction of what
will burn, and it will usually dwarf the observed figure — a fire running 50 km
in twelve hours sweeps a very large corridor. Read it as an upper bound on
reach, never as an area estimate.

The smoke animation shows near-surface fire drift, **not** smoke dispersion. Air
quality work needs a transport model such as HYSPLIT.

**This project does not measure burned area.** That needs Sentinel-2 scenes
before and after, and a dNBR calculation.



## Related project

[**fire-spread-sandbox**](https://github.com/aigulbekbayeva/fire-spread-sandbox) —
the forward-looking counterpart. Click any point on a map and it forecasts
spread from live hourly wind, entirely in the browser: one HTML file, no
backend, no key. Same physics, ported to JavaScript, with the same reference
values pinned in its own tests.

This project asks *what happened*; that one asks *what if*.

## Sources

- NASA FIRMS (LANCE, EOSDIS) — VIIRS and MODIS active fire products
- ECMWF ERA5 reanalysis, via the Open-Meteo archive API
- Cheney N.P., Gould J.S., Catchpole W.R. (1998). *Prediction of fire spread in
  grasslands.* International Journal of Wildland Fire, 8(1), 1–13.
- Cruz M.G. et al. (2015). *Empirical-based models for predicting head-fire rate
  of spread in Australian fuels.* Australian Forestry, 78(3), 118–158.

Citations are reproduced from memory — verify them before quoting in a report
or paper.

## License

MIT — see [LICENSE](LICENSE).
