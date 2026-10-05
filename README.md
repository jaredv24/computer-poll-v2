# Computer Poll v2.0

Draft v2.0 methodology for The Computer Poll, with a working prototype and a 2015–2025 backtest.

- `index.html`: the methodology page (static, no build step).
- `prototype/v2.py`: the prototype. It reads cfbfastR schedule CSVs from `prototype/data/s<year>.csv`:

  ```sh
  mkdir -p prototype/data
  for y in $(seq 2014 2026); do
    curl -fsSo prototype/data/s$y.csv \
      https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_$y.csv
  done
  pip install numpy scipy pandas
  BOOT=300 python3 prototype/v2.py   # writes prototype/results.json
  ```

## Deploying on Vercel

1. On vercel.com: **Add New… > Project**, import this repository.
2. Framework Preset: **Other**. Leave the build command and output directory empty.
3. **Deploy**. Every push to `main` redeploys automatically.
