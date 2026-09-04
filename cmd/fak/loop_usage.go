package main

import (
	"fmt"
	"io"
)

func loopUsage(w io.Writer) {
	fmt.Fprint(w, `fak loop - durable long-running loop ledger

  fak loop append --loop ID --kind KIND [--ledger FILE] [--run ID]
                  [--source NAME] [--principal ID] [--status STATUS]
                  [--reason CODE] [--summary TEXT] [--evidence KIND=REF]
                  [--metric NAME=INT64] [--json]
  fak loop run --loop ID [--ledger FILE] [--source cron|launchd|task-scheduler] [--notify-slack] [--no-guard] -- CMD [ARG...]
  fak loop status [--ledger FILE] [--json]
  fak loop health [--ledger FILE] [--registry FILE] [--check] [--json]
  fak loop rollup [--ledger PATH|NODE=PATH ...] [--dir DIR] [--glob '*.jsonl'] [--json]
  fak loop economics [--ledger FILE] [--loop ID] [--provider-cache-tokens N]
                  [--fak-authored-tokens N] [--modeled-tokens-per-avoided N] [--json]
  fak loop admit [--loop ID] [--ledger FILE] [--policy FILE] [--json]
  fak loop policy propose [--ledger FILE] [--policy FILE] [--json]
  fak loop region [--lane LANE] [--tree GLOB ...] [--actor ID] [--self LEASE-ID]
                  [--dir DIR] [--json]
  fak loop recover [--ledger FILE] [--stale-min N] [--now UNIX] [--all] [--json]
  fak loop repair [--ledger FILE] --confirm [--json]
  fak loop drive [--loop ID] [--goal GOAL.md] [--ledger FILE] [--policy FILE]
                  [--max-iters N] [--max-tokens N] [--deadline RFC3339|DUR]
                  [--review-model M] -- CMD [ARG...]
  fak loop drive --template [--loop ID]
  fak loop reap [--reap] [--allow-unfenced] [--supervisor-marker SUB ...]
                  [--worker-marker SUB ...] [--json]

Append records one scheduler/script/control event in the canonical hash-chained
ledger. Run wraps an OS scheduler command under fak guard by default and records
fire/admit/start/end around it; a direct out-of-tree write/delete is refused before
spawn with OUT_OF_TREE_WRITE, and --no-guard is an explicit logged opt-out.
Status folds that ledger into the current loop/run view using the recovered
valid prefix when the hash chain is forked or corrupt, warning on stderr without
rewriting the audit log. Health joins the ledger with the durable registry and
renders live/stale/dark-loop state plus current learning_debt for the
docs-freshness loop. Rollup folds MANY nodes' ledgers into one fleet-wide "how
often did every loop run" view — per-loop run counts, cadence, and last-run —
reusing the fak ps table format; it is a read-only aggregation that ingests
journals and writes nothing. Economics folds that same ledger into one honest
loop-economics readout — baseline vs observed open count, close/retry rate,
duplicate attempts avoided, effective workers, and wall time as WITNESSED figures —
and keeps the provider-cache, fak-authored, and modeled token-saving accounts
strictly separate, each defaulting to not_yet until an explicit witness is folded so
it never invents a saving the ledger cannot prove. Admit applies the tunable
admission policy (default .fak/loop-policy.json, FAK_LOOP_POLICY) to the fold and
prints admit/refuse per loop — exit 3 when any evaluated loop is refused, so a
scheduler line can gate work on it. Recover folds the ledger into the cross-run
RECOVERY worklist: the dispatched runs that started but were never finished
(orphaned) or never witnessed (unwitnessed) — the work to re-dispatch or re-verify.
Repair is the explicit operator mutation: it archives a broken ledger tail and
rewrites only the valid prefix; readers never invoke it automatically. The ledger
records events; admission, scheduler authority, and completion witnesses live in
producers.
Drive reads a GOAL.md goal-spec fresh before every turn, gates each turn through
the loop admission policy, appends fire/admit/start/end/witness events to this ledger,
and re-spawns CMD until the configured DOS witness reports witnessed_done or a
budget is spent. With --review-model it also exports FAK_REVIEW_* so fak commit
asks a scout reviewer to pass/refute the turn diff before committing; review
verdicts are recorded as loop-ledger evidence. A NOT_YET witness refusal is
appended under Scratch and exposed through FAK_GOAL_LAST_REFUSAL so the next
fresh-context turn can see it. A GOAL.md lane:/region: declaration (or --lane/
--tree) additionally holds a region lease on the shared lease fabric while the
drive runs, refusing COLLISION_RISK instead of racing a live peer.
Region is the surface-neutral admission question by itself: "may ACTOR act on
this lane/tree right now?" answered against the live lease fabric and the
dos.toml lane taxonomy (exit 0 admit / 3 refuse) — the check a manual session
or a super-loop enter path runs before touching a region. It decides only;
holding a lease stays with fak leaseref acquire.
Reap scans the live process table for detached loop/drainer supervisors and folds
them through the pure looporphan core: it KEEPs the one parenting live work (or an
attached idle engine), REAPs orphaned/duplicate idle engines, flags same-lane
COLLISIONs for an operator, and fails closed to UNKNOWN on a missing start-time
fence or thin identity. It reports only by default (exit 3 when any supervisor is
reap-eligible); --reap tree-kills exactly the REAP set via the same native reaper
the loop supervisor uses, never the current process or its parent.
`)
}
