#!/usr/bin/env bash
# Vercel build: refresh the poll data, then assemble the static site in public/.
# If the data refresh fails for any reason, the site ships with the data already in the repo.
set -u
cd "$(dirname "$0")"

PY=""
for c in python3.12 python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done

refresh() {
  [ -n "$PY" ] || { echo "No Python found"; return 1; }
  echo "Using $($PY --version 2>&1)"
  $PY -m venv .venv && . .venv/bin/activate || return 1
  python -m pip install -q --disable-pip-version-check numpy scipy pandas pyarrow || return 1
  python pipeline/build_poll.py
}

# keep a copy of the committed data in case the refresh fails or comes back thinner
rm -rf .data_backup && cp -r data .data_backup

good() {
  # with a CFBD key, a good refresh must include efficiency and the AP poll
  [ -z "${CFBD_API_KEY:-}" ] && return 0
  python -c "import json,sys; d=json.load(open('data/poll-' + json.load(open('data/latest.json'))['file'].split('-')[1])); sys.exit(0 if d['efficiency'] and d['apAvailable'] else 1)"
}

if refresh && good; then
  echo "Poll data refreshed"
else
  echo "WARNING: data refresh failed or was incomplete; deploying the data committed in the repo"
  rm -rf data && mv .data_backup data
fi
rm -rf .data_backup

rm -rf public && mkdir -p public/pipeline
cp index.html public/
cp -r vendor data public/
[ -d logos ] && cp -r logos public/
[ -d prototype ] && mkdir -p public/prototype && cp prototype/*.py public/prototype/ 2>/dev/null
cp pipeline/*.py public/pipeline/
echo "Site assembled in public/"
