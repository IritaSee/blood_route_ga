# Blood Supply Route Optimization (Malang) - GA

Genetic-algorithm route optimizer for blood deliveries in Malang Regency. This is
a fresh, cut-down repo carried over from an earlier prototype that had grown to
include a deep-learning predictor, multiple exploratory-data-analysis scripts, and
two half-finished parallel GA implementations. After client interviews, the actual
scope is just two things:

1. **GA optimized against historically available routes** - `run_historical.py`
2. **GA + Google Maps API, so the same optimizer can react to live traffic** - `run_live.py`

Everything deep-learning-related from the old repo was dropped entirely, not
ported. Nothing here depends on TensorFlow/Keras.

## Project Layout

```
optimizer/
  data_extractor.py            - parse/clean Data PMI.xlsx & All Droping.xlsx
  geocoder.py                  - cached Nominatim geocoding (address -> lat/lon)
  comparison_reporter.py       - GA vs historical-baseline report generator
  genetic_algorithm.py         - the original GA, ported untouched
  genetic_algorithm_optimized.py - reworked GA (see "Two GA versions" below)
  routing_osrm.py              - static historical routing (OSRM / haversine)
  routing_google_maps.py       - live-traffic routing (Google Maps Distance Matrix)
  route_geometry.py            - road-snapped route geometry for the map (OSRM /route, Google Directions)
  pipeline.py                  - shared orchestration: extract -> geocode -> matrix -> GA -> save
run_historical.py               - entry point 1 (OSRM)
run_live.py                     - entry point 2 (Google Maps, live traffic)
scripts/
  visualize_routes.py           - render baseline/OSRM-GA/Google-GA routes as an OSM map
data/                           - put Data PMI.xlsx / All Droping.xlsx here (gitignored)
tests/                          - GA correctness tests, no network/data required
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put `Data PMI.xlsx` and `All Droping.xlsx` in `data/` (see `data/README.md` -
these are gitignored on purpose, they contain real donor records).

For live traffic, copy `.env.example` to `.env` and set `GOOGLE_MAPS_API_KEY`
(needs the Distance Matrix API enabled in Google Cloud Console).

## Manual coordinate override (recommended for unstable geocoding)

If geocoding coverage is incomplete, you can provide fixed coordinates from a
CSV and force the pipeline to use them first.

Supported override files (first one found is used):

- `data/geocode_overrides.csv`
- `data/geocode_gap_report.csv`
- `data/geocode_gap_report (1).csv`

You can also set a specific file explicitly:

```bash
GEOCODE_OVERRIDE_FILE="data/geocode_overrides.csv" python run_historical.py --population 200 --generations 500 --output-dir results/historical_eval
GEOCODE_OVERRIDE_FILE="data/geocode_overrides.csv" python run_live.py --population 200 --generations 500 --output-dir results/live_eval
```

Override CSV columns used by the loader:

- `name`
- `location_district` (optional but recommended)
- `lat`
- `lon`

Notes:

- Manual CSV overrides are checked before online geocoding.
- Non-location rows such as numeric labels, `JUMLAH`, and `RATA-RATA` are now
  filtered out during location extraction so they do not enter GA.

## Google Maps API tutorial (for `run_live.py`)

If this is your first time with Google Cloud, follow this once:

1. Open Google Cloud Console: https://console.cloud.google.com/
2. Create a project (or pick an existing one).
3. Enable billing for that project (required for Google Maps Platform APIs).
4. Go to **APIs & Services -> Library**.
5. Search and enable **Distance Matrix API** (required for `run_live.py` itself).
   If you also want the Google-GA route drawn with real road geometry in
   `scripts/visualize_routes.py`, additionally enable **Directions API** - without
   it, that layer falls back to OSRM road geometry (the route order/colors are
   unaffected either way).
6. Open **APIs & Services -> Credentials**.
7. Click **Create credentials -> API key**.
8. Copy the generated key.

Recommended security hardening:

1. In the key settings, set **Application restrictions**:
  - For local/dev scripts, use **IP addresses** and whitelist your server IP(s), or
  - Use **None** only temporarily during setup/testing.
2. Set **API restrictions** to **Distance Matrix API** (plus **Directions API** if
   you'll also run `scripts/visualize_routes.py`).
3. Save changes.

Then configure this project:

```bash
cp .env.example .env
```

Open `.env` and set:

```env
GOOGLE_MAPS_API_KEY=your_real_api_key_here
```

Quick validation run:

```bash
python run_live.py --population 20 --generations 30
```

If the API key is invalid/misconfigured, Google typically returns `REQUEST_DENIED`
or an error about billing/API restrictions.

## Usage

### 1. GA from historical routes (OSRM)

```bash
python run_historical.py                          # optimized GA, OSRM routing
python run_historical.py --ga baseline             # original GA instead
python run_historical.py --no-osrm                 # haversine, no network needed
python run_historical.py --population 200 --generations 500
python run_historical.py --seed 42                 # reproducible run (see below)
```

Writes `results/historical/ga_results.json`, `comparison.json` (GA vs baseline),
and a text/JSON comparison report.

### 2. GA + Google Maps live feed

```bash
python run_live.py                                 # optimized GA, live traffic
python run_live.py --ga baseline
python run_live.py --no-live-traffic                # static Google Maps, no traffic
python run_live.py --traffic-model pessimistic
python run_live.py --seed 42                        # reproducible run (see below)
```

Writes to `results/live/` in the same shape as `run_historical.py`. Every route
matrix is built with `departure_time=now`, so re-running later in the day can
produce a different optimal route as traffic changes - that's the point.

### Why `--seed` matters

A genetic algorithm runs on randomness by design: which routes seed the starting
population, which two parent routes get crossed and at what point, which stops get
shuffled by mutation, which individuals win a tournament, and (in the optimized GA)
which individuals get replaced when the search stagnates. That randomness is the
whole point - it's how the GA explores many different stop orders and vehicle
splits instead of settling on the first decent-looking route it finds.

The catch: without a seed, that randomness comes from the OS's entropy source, so
it's different on every run - even with the exact same customers, coordinates, and
distance matrix. Two back-to-back runs of `python run_historical.py` will usually
land on two different "best" routes: same stops, but a different visiting order
and/or a different split between vehicle 1 and vehicle 2, so a slightly different
total distance/cost/makespan too. Neither run is wrong - a GA only guarantees a
*good* solution, not *the* optimal one, and which good solution it lands on depends
on the random path it took to get there.

That's fine when you just want a good route for today. It becomes a problem the
moment you need to reproduce a specific result:

- **The map.** `scripts/visualize_routes.py` draws whatever `stops` sequence
  happens to be in `ga_results.json` right now. Regenerate that file without a seed
  and you'll likely get a *different* route than the one behind your last
  `comparison_report.txt` - the map and the report quietly fall out of sync.
- **Debugging.** If a route looks worse after you change a matrix, a GA parameter,
  or fix a bug, an unseeded run can't tell you whether your change caused it or the
  GA just got unlucky this time. A seeded run removes that doubt: same seed, same
  inputs, same route, every time.
- **Sharing a result.** `--seed 42` (with the same data and parameters) is the only
  way to say "run this and you'll get exactly the route I got" - the same idea as
  sharing a save-game seed in a game with procedurally generated levels.

Passing `--seed <int>` fixes Python's and NumPy's random number generators to a
known starting point before the GA runs, so "random" becomes "the same sequence of
choices, every time." Leave it unset when you're happy letting the GA explore
freely; set it whenever today's run needs to look the same next week.

### 3. Evaluate historical vs live results

To compare both scenarios fairly, run them with the same GA settings:

```bash
python run_historical.py --population 200 --generations 500 --output-dir results/historical_eval
python run_live.py --population 200 --generations 500 --output-dir results/live_eval
```

Each folder will contain:

- `ga_results.json` - core GA output (distance, total time, makespan, total cost,
  and a `stops` list per vehicle - depot-to-depot facility name/lat/lon in visit
  order, as chosen by the GA)
- `comparison.json` - GA vs historical baseline percentage reductions
- `comparison_report.txt` - human-readable summary report

Quick side-by-side metric check:

```bash
python - <<'PY'
import json
from pathlib import Path

hist = json.loads(Path('results/historical_eval/ga_results.json').read_text())
live = json.loads(Path('results/live_eval/ga_results.json').read_text())

metrics = [
  ('total_distance_km', 'Total Distance (km)'),
  ('total_time_hours', 'Total Time (h)'),
  ('makespan_hours', 'Makespan (h)'),
  ('total_cost_idr', 'Total Cost (IDR)'),
]

print('\nGA RESULT COMPARISON (Historical vs Live)')
print('-' * 72)
print(f"{'Metric':<22} {'Historical':>18} {'Live':>18} {'Delta (Live-Hist)':>14}")
print('-' * 72)
for key, label in metrics:
  h = float(hist.get(key, 0))
  l = float(live.get(key, 0))
  d = l - h
  print(f"{label:<22} {h:>18,.2f} {l:>18,.2f} {d:>14,.2f}")
print('-' * 72)
PY
```

Interpretation tips:

- If live traffic is heavier, `total_time_hours` and `makespan_hours` in live mode usually increase.
- If live mode reroutes around congestion, `total_distance_km` can increase while time decreases (detour effect).
- Use `comparison_report.txt` in each folder to see GA improvements against historical baseline, then compare those improvement rates between historical and live runs.

### 4. Visualize routes on a map

Both routers only ever compute a distance/duration *matrix* for the GA - neither
returns the actual road path, so drawing a route requires a separate geometry
fetch. `scripts/visualize_routes.py` does that and renders a single OSM map with
three toggleable layers: the historical baseline (depot &lt;-&gt; each destination,
since those trips are single-destination, not multi-stop), the OSRM-GA route, and
the Google-GA route.

```bash
python run_historical.py --seed 42 --output-dir results/historical_eval_v3
python run_live.py --seed 42 --output-dir results/live_eval_v3
python scripts/visualize_routes.py \
  --historical-dir results/historical_eval_v3 \
  --live-dir results/live_eval_v3 \
  --output results/route_comparison_map.html
```

Then open `results/route_comparison_map.html` in a browser. Notes:

- Both `ga_results.json` files must have been generated after the `stops` field
  was added to `pipeline.py` - re-run the pipeline if you see a `KeyError`/`stops`
  error.
- Road geometry is fetched via OSRM's public `/route` service for the baseline and
  OSRM-GA layers, and Google's Directions API for the Google-GA layer (falls back
  to OSRM geometry if `GOOGLE_MAPS_API_KEY` isn't set or Directions API isn't
  enabled on the key - see the API tutorial above).
- Geometry is cached in `results/route_geometry_cache.db` (configurable via
  `--geometry-cache`), so re-running the script after the first time is fast and
  doesn't re-hit either API.

## Two GA versions

Both live side by side and share the exact same constructor/`run()`/
`get_best_solution_details()` interface, so either can be passed to the pipeline
via `--ga baseline` / `--ga optimized`.

- **`genetic_algorithm.py`** - the original implementation, untouched. Lexicographic
  fitness (makespan -> total time -> cost), tournament selection, elitism.
- **`genetic_algorithm_optimized.py`** - same fitness shape, four concrete fixes
  on top:
  1. **Route order survives crossover.** The original crossover only reassigns
     which vehicle a customer belongs to, then rebuilds each route in ascending
     customer-id order - so visiting order is never actually searched, even
     though the fitness function is entirely order-dependent. The optimized
     version does order crossover (OX) directly on each vehicle's sequence.
  2. **2-opt local search** runs on a sample of offspring each generation to
     clean up crossed legs the GA is bad at fixing via mutation alone.
  3. **Vehicle capacity is enforced.** The original accepted a `vehicle_capacity`
     parameter but had no way to check it against anything. The optimized
     version takes an optional `quantities` array and penalizes overloaded
     routes.
  4. **Stagnation-triggered diversity injection** - if the best fitness hasn't
     improved for `stagnation_limit` generations, half the non-elite population
     is replaced with fresh individuals to escape local optima.

Default is `optimized` in both entry points; pass `--ga baseline` to compare
against the original.

## Known limitations (carried over from the source data, not the code)

- Geocoding many facility names against OpenStreetMap/Nominatim has partial
  coverage - some small clinics and out-of-region UDD/BDRS facilities won't
  resolve to coordinates. Locations that fail to geocode are excluded from the
  distance matrix (`inf` distance), and the GA will report "no feasible solution"
  if too many customers end up unreachable. Pre-fill the geocode cache with known
  coordinates for anything that matters for your run.
- OSRM's public server (`router.project-osrm.org`) is rate-limited and meant for
  light testing, not production load - self-host OSRM if you need reliability.
- Google Maps Distance Matrix pricing is per-element (`origins x destinations`);
  a large fleet/customer count will hit real API costs. The live-traffic cache
  (`cache-ttl`, default 5 min) exists specifically to avoid re-fetching the same
  pair every GA generation within one run.

## Tests

```bash
pytest tests/
```

Uses mock distance/duration matrices, no Excel data or network required. Covers
both GA implementations, crossover/repair correctness, and the capacity penalty.
