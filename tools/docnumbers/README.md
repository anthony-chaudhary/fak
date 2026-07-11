# Doc-numbers manifests

The checking layer for **recent-operational cachevalue docs** — pages that quote
this-week's `fak cachevalue report` fleet/dev telemetry (e.g.
`docs/integrations/fable5-more-usage-for-free.md`). Audited by
`tools/cachedoc_numbers_audit.py`, gated in `make cachedoc-numbers-lint`.

## Why this exists (and where the line is)

- `docs/benchmarks/BENCHMARK-AUTHORITY.md` is the source of truth for **committed
  benchmark claims** — captured, reproducible, with a named baseline.
- `tools/readme_freshness_audit.py` keeps the **README's** headline numbers honest.
- This layer covers the gap between them: numbers that are **real but live-moving**
  (fleet share this week, the last few days of dev sessions). They are too
  transient to earn an authority row, yet are exactly where a number silently
  rots, or an arithmetic slip hides — the bug that motivated this tool was
  *"60 discovered — 15 priced, 36 held out"* being read as `60 = 15 + 36` (= 51).

Each guarded doc gets one manifest here, plus a directory of **trimmed, frozen
snapshots** it was captured from. The audit is fully hermetic: it needs no
network, no `fak`, and no live feed. It proves three things line up —

```
doc render  ==  manifest.expected  ==  snapshot.field
```

— and that the arithmetic the doc asserts actually closes.

## Files

```
tools/docnumbers/
  <stem>.json                       # the manifest (one per guarded doc)
  snapshots/<stem>/<source>.json    # trimmed frozen evidence, one per source
  README.md                         # this file
```

## Manifest schema (`fak.docnumbers.v1`)

```jsonc
{
  "schema": "fak.docnumbers.v1",
  "doc": "docs/integrations/…md",           // the guarded doc, repo-relative
  "snapshot_dir": "tools/docnumbers/snapshots/<stem>",
  "snapshot_date": "2026-07-09",            // when the snapshots were captured
  "stale_after_days": 21,                    // WARN (never FAIL) past this age

  "sources": {                               // where each number was captured
    "fleet.json": {
      "cmd": ["cachevalue", "report", "--since", "2026-07-06", "--json"],
      "window": "since_open"                 // see the window taxonomy below
    },
    "session-8a9f820c.json": { "cmd": null, "window": "live_snapshot" }
  },

  "claims": [
    {
      "id": "fleet.fires",
      "appears_as": "2,347 fires over 898 sessions",   // literal substring in the doc
      "source": "fleet.json",
      "provenance": "WITNESSED",             // WITNESSED | OBSERVED | ESTIMATED
      "provenance_inline": true,             // must appear on the same doc line (default true)
      "numbers": [
        { "display": "2,347", "field": "fleet_benefit.compaction_fired", "expected": 2347 },
        { "display": "898",   "field": "fleet_benefit.exit_sessions",    "expected": 898 }
      ]
    }
  ],

  "invariants": [
    { "kind": "sum",     "label": "…", "total": 60, "parts": [15, 36, 9] },
    { "kind": "formula", "label": "…", "expr": "1181.37 - 375.54", "expect": 805.83, "tol": 0.01 }
  ]
}
```

**Per-number fields.** `display` is the rendered string (parsed for value,
suffix `M/K/B`, `%`, `$`, `,`, and a leading `≈`). `expected` is the raw value
of `field` in the snapshot. `scale` (default 1) converts the field's unit to the
display's — e.g. a `cache_hit_frac` of `0.89` rendered as `89.3%` needs
`"scale": 100`. Omit `field` for a **derived** number (e.g. a per-fire ratio); it
is then only rounding-checked against `expected`, and an invariant should pin the
arithmetic.

**Invariants.** `sum` asserts `parts` add to `total`. `formula` evaluates `expr`
(numbers and `+ - * / ( )` only — no identifiers) and checks it equals `expect`
within `tol`. These catch the decomposition/derivation bugs that a per-number
check cannot see.

## Window taxonomy (what `--live` may assert)

`--live` re-runs the cited commands. What it is allowed to conclude depends on
the window, because most cachevalue windows **grow with wall-clock** — a larger
number today is elapsed time, not drift:

| window | meaning | `--live` behaviour |
|---|---|---|
| `bounded` | doubly-bounded & settled (`--since X --until Y`, `Y` in the past) | **equality-checked** (>1% move FAILs) |
| `since_open` | open-ended (`--since X`, no end) — grows daily | **schema-probe only** — cited field must still exist |
| `all_time_last_measured` | last-measured bucket; value moves as buckets land | schema-probe only |
| `live_snapshot` | dev transcripts / per-session status; grow between captures | **never touched** — frozen snapshot is the gate |

So for a typical fleet/dev doc, `--live` proves the cited fields still exist
(catches a renamed/removed field), and the **frozen snapshots** remain the
authority for values. `--live` SKIPs cleanly (exit 0) when `fak` is absent.

## Adding a doc

1. Capture the numbers with `fak cachevalue report … --json` and save trimmed
   snapshots under `snapshots/<stem>/` (only the referenced fields — see the
   existing ones). `--refresh` (below) automates this for cmd-bearing sources.
2. Write `<stem>.json` binding each rendered number to its snapshot field, and
   add the sums/formulas the doc asserts as invariants.
3. `python3 tools/cachedoc_numbers_audit.py` until clean. Add the manifest,
   snapshots, and doc in the same commit.

## Refreshing the numbers

When the doc's window has moved on and the numbers should be updated:

```
python3 tools/cachedoc_numbers_audit.py --refresh   # regenerate snapshots from live fak
# → edit the doc to match, bump snapshot_date, then:
python3 tools/cachedoc_numbers_audit.py             # audit until clean
```

`--refresh` must run from the repo root where `fak` has its data context; it
writes only inside each doc's `snapshot_dir`, and SKIPs sources with no `cmd`
(machine-specific transcript paths — recapture those by hand). The repeatable
pass is also available as the `/refresh-cachedoc-numbers` skill.
