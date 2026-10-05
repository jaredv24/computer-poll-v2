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

Vercel rebuilds the poll data on every deploy (`build.sh` runs `pipeline/build_poll.py`,
then assembles the site in `public/`). If the refresh fails, the deploy uses the data
committed in `data/`.

Project settings:

1. **Environment Variables**: `CFBD_API_KEY` (CollegeFootballData key), `DEPLOY_HOOK_URL`
   (from step 2), and `CRON_SECRET` (any long random string).
2. **Git > Deploy Hooks**: create a hook for branch `main` and copy its URL into `DEPLOY_HOOK_URL`.
3. Redeploy once so the build picks up the variables.

`vercel.json` schedules `/api/rebuild` every Sunday 20:15 UTC and Monday 12:15 UTC; it calls
the deploy hook, which rebuilds the data. The GitHub Action in `.github/workflows` is a manual
backup that commits fresh data to the repo.
