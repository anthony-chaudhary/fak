#!/usr/bin/env bash
# One tick of the popularization dispatch loop: pick the next un-closed
# popularization issue that is not already being worked, and run one guarded
# --live dispatch tick for it. The account-switcher + preflight cap enforce
# safety; a capped seat holds the tick (no spawn, no error). Idempotent per
# tick — safe to run on a schedule.
#
# Env:
#   POP_MAX_WORKERS  hard worker cap (default 4)
#   POP_BACKEND      claude | opencode (default claude)
#   POP_LIVE         set to "1" to actually spawn (default: dry-run)
set -u
cd /c/work/fak || exit 2

REPO=anthony-chaudhary/fak
MAXW=${POP_MAX_WORKERS:-4}
BACKEND=${POP_BACKEND:-claude}
LIVE_FLAG=""
[ "${POP_LIVE:-0}" = "1" ] && LIVE_FLAG="--live"

# ticket -> lane map lives in the generator; re-emit the lane sidecars once.
LANEMAP="C:/Users/USER/AppData/Local/Temp/pop_lanes.tsv"
if [ ! -s "$LANEMAP" ]; then
  if [ -x ./fak ]; then
    ./fak popularization-tickets --lanes-tsv > "$LANEMAP"
  elif [ -x ./fak.exe ]; then
    ./fak.exe popularization-tickets --lanes-tsv > "$LANEMAP"
  else
    go run ./cmd/fak popularization-tickets --lanes-tsv > "$LANEMAP"
  fi
fi

# open popularization issues, freshest-first (lowest number = filed first = do first)
mapfile -t OPEN < <(gh issue list --repo "$REPO" --label popularization --state open \
                    --limit 100 --json number,title --jq 'sort_by(.number)[] | "\(.number)\t\(.title)"')

for row in "${OPEN[@]}"; do
  num=${row%%$'\t'*}
  title=${row#*$'\t'}
  # resolve lane from title (exact-match against the generator's map)
  lane=$(awk -F'\t' -v t="$title" '$1==t{print $2; exit}' "$LANEMAP")
  [ -z "$lane" ] && continue
  # skip if a worker is already live on this lane (dispatcher's held_lanes handles the hard case; this is a cheap pre-filter)
  echo ">>> tick: #$num  lane=$lane"
  out=$(python tools/issue_resolve_dispatch.py --backend "$BACKEND" --lane "$lane" --issue "$num" \
        --max-workers "$MAXW" $LIVE_FLAG --json 2>&1)
  verdict=$(echo "$out" | python -c "import json,sys
try: print(json.load(sys.stdin).get('verdict','?'))
except: print('PARSE_ERR')")
  echo "    verdict=$verdict"
  case "$verdict" in
    SPAWNED|WOULD_SPAWN) echo "    -> handled #$num; one spawn per tick"; exit 0 ;;
    WEEKLY_CAPPED|SESSION_CAPPED) echo "    -> all seats capped; holding this tick"; exit 0 ;;
    ISSUE_CONTRACT_HOLD) echo "    -> #$num not gate-ready, trying next"; continue ;;
    *) echo "    -> $verdict on #$num, trying next"; continue ;;
  esac
done
echo "no dispatchable popularization issue this tick"
