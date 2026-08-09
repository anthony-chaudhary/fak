package main

// Beat-while-working for the DOS lane lease (#5864) — the mirror of #4324's
// release-on-exit in dispatch_tick_lease_release.go.
//
// THE GAP. `dos.lane_lease._lease_is_dead` reads `heartbeat_at` as its PRIMARY
// liveness evidence and falls back to `acquired_at`. `dos lease-lane heartbeat`
// is the writer that populates it — and on this fleet NOTHING ever called it.
// Measured over C:\work\fak\.dos\lane-journal.jsonl: 3584 entries — 129 ACQUIRE,
// 98 RELEASE, 94 REFUSE, 3263 ENFORCE, **0 HEARTBEAT** — and 0 of the 31
// structurally-live lease records carry a `heartbeat_at` at all. So the primary
// rung degenerated to "older than 50 min since ACQUIRE" for every lease, and the
// pre-admission ENFORCE read (`live_leases(expire_dead=True)`, which is what
// 3263 of those entries are) drops a still-WORKING holder's lease at 55 minutes
// flat. Of 98 matched ACQUIRE->RELEASE lifetimes in that WAL, 41 ran longer than
// that bound. A heartbeat is what keeps the contention view honest for those.
//
// THE SEAM, AND WHY IT IS THIS ONE. dispatch_tick_lease_release.go names the
// witness sweep as "the one place in the Go dispatch stack that OBSERVES a
// worker finishing" and rides the release there. The beat rides the SAME sweep
// on the other branch — the one where the sweep observes a worker still
// RUNNING (`dispatchPIDAlive(pid)` -> continue). That single call site gives the
// beat the property the whole fix turns on: it is driven by the supervisor
// re-observing the work, so it cannot outlive the work. There is no timer, no
// goroutine, and nothing that keeps ticking once the process is gone; the very
// next sweep that finds the pid dead takes the release branch instead.
//
// THE HAZARD, WHICH IS THE INVERSE OF THE BUG. A heartbeat is the one
// fail-DANGEROUS lease op. A release only FORGETS a lease (-> reclaimable, the
// safe direction); a beat REVIVES one. A beat written for a holder that is
// already gone would make a dead lease look permanently alive — strictly worse
// than today's silence, because the TTL backstop currently doing all the work
// would stop firing too. Every rung of that decision therefore lives in the
// PURE internal/lanebeat, where the dangerous direction is reachable from a
// test: dead pid, no output growth, past the worker's own budget, a foreign
// host, an unattributable holder, or a lease that predates the worker all
// refuse, and a refusal carries no beat identity at all.
//
// WHAT THIS DOES NOT CLAIM. The DOS lane lease is acquired by the WORKER
// (`dos lease-lane acquire`, the lane-lease prompt rule), not by fak's Go stack
// — which takes a separate refs/fak/locks lease through internal/leaseref. So
// fak beats as the SUPERVISOR that spawned the holder and can see it, not as
// the holder. That standing is exactly the evidence the kernel's own PID rung
// wants and cannot get: the pid recorded ON the lease is the ephemeral acquire
// subprocess, which exits by design, so every healthy lease probes dead from the
// kernel's side. The binding that keeps this from becoming a licence to revive
// anyone's orphan is lanebeat's LEASE_PREDATES_HOLDER rung.

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/lanebeat"
)

// dispatchLaneBeatCoalesceS is the write-amplification brake the kernel ships
// for exactly this caller (`--coalesce-within-s`): when the matched lease's
// current beat is younger than this, the kernel elides the append and still
// reports success. Chosen well UNDER the 50-minute TTL the beat is defending, so
// the lease never reads older than this between writes, and comfortably above
// the tick cadence so a busy fleet does not pump the WAL. Eliding can only let
// an existing beat AGE, never fabricate a fresher one — the one-way-safe
// direction.
const dispatchLaneBeatCoalesceS = 120

// dispatchLaneBeatTimeout bounds each `dos` child. A slow kernel must cost the
// sweep a bounded pause, never a wedge.
const dispatchLaneBeatTimeout = 20 * time.Second

// dispatchLaneBeatMaxHold caps how long one worker may be attested at all,
// regardless of how alive it looks. It is the worker's OWN budget — the same
// WorkerTimeoutS + LeaseTTLMarginS the lane acquire is taken under — so a
// process that hangs past its deadline stops being attested and its lane falls
// back to plain TTL expiry instead of being pinned indefinitely by a beat.
const dispatchLaneBeatMaxHold = time.Duration(dispatchtick.DefaultWorkerTimeoutS+dispatchtick.LeaseTTLMarginS) * time.Second

// dispatchLaneBeatReader / dispatchLaneBeatWriter are the two I/O seams, injectable
// so a test pins the decision and the call site without a live `dos`.
var (
	dispatchLaneBeatReader = dispatchLaneBeatLiveLeasesDos
	dispatchLaneBeatWriter = dispatchLaneBeatWriteDos
)

// dispatchLaneBeater carries one sweep's worth of beat state. The live-lease set
// is read at most ONCE per sweep and only when a live worker is actually found,
// so a sweep over a runs dir with nothing running spawns no `dos` child at all.
type dispatchLaneBeater struct {
	root   string
	host   string
	live   []lanebeat.Lease
	loaded bool
	// outcomes counts each closed-vocabulary reason the sweep reached, so "why was
	// this lane not beaten" is countable on the payload instead of being lost.
	outcomes map[string]int
}

func newDispatchLaneBeater(root string) *dispatchLaneBeater {
	host, _ := os.Hostname()
	if v := strings.TrimSpace(os.Getenv("DISPATCH_HOST_ID")); v != "" {
		host = v
	}
	return &dispatchLaneBeater{root: root, host: host, outcomes: map[string]int{}}
}

// beatLiveWorker refreshes the DOS lane lease held on behalf of one worker the
// sweep has just proven is still running. Returns the closed-vocabulary reason.
//
// FAIL-OPEN in every direction: an unreadable live set, an undecidable holder or
// a failed write all leave the lease exactly as it is today (TTL-only), and
// nothing is propagated to the sweep. A beat that does not happen costs the
// pre-#5864 behaviour; a beat that happens wrongly costs a revived orphan, which
// is why every uncertainty resolves to "don't".
func (b *dispatchLaneBeater) beatLiveWorker(log, stem string, pid int, now time.Time) string {
	lane := laneFromSpawnHeader(log)
	if strings.TrimSpace(lane) == "" {
		return b.record(lanebeat.ReasonNoLane)
	}
	h := lanebeat.Holder{
		Lane:   lane,
		HostID: b.host,
		PID:    pid,
		// The caller reached this branch through dispatchPIDAlive: a process-table
		// read taken moments ago, not a claim the worker made about itself.
		Alive:        true,
		StartedAt:    dispatchLaneBeatSpawnedAt(stem),
		LastOutputAt: dispatchLaneBeatLastOutputAt(log),
		MaxHold:      dispatchLaneBeatMaxHold,
	}
	if !b.loadLive() {
		return b.record("LIVE_LEASES_UNREADABLE")
	}
	dec := lanebeat.Decide(h, b.live, now)
	if !dec.Beat {
		return b.record(dec.Reason)
	}
	if !dispatchLaneBeatWriter(b.root, dec) {
		return b.record("BEAT_FAILED")
	}
	return b.record(dec.Reason)
}

func (b *dispatchLaneBeater) record(reason string) string {
	if reason == "" {
		reason = "UNKNOWN"
	}
	b.outcomes[reason]++
	return reason
}

// loadLive reads the structurally-live lease set once per sweep. A read fault
// leaves the set empty AND marks it loaded, so one broken `dos` invocation does
// not make the sweep re-spawn a child per worker.
func (b *dispatchLaneBeater) loadLive() bool {
	if b.loaded {
		return b.live != nil
	}
	b.loaded = true
	live, ok := dispatchLaneBeatReader(b.root)
	if !ok {
		return false
	}
	if live == nil {
		live = []lanebeat.Lease{}
	}
	b.live = live
	return true
}

// summary projects the sweep's beat outcomes for the payload, or nil when the
// sweep never had a live worker to attest.
func (b *dispatchLaneBeater) summary() map[string]any {
	if b == nil || len(b.outcomes) == 0 {
		return nil
	}
	counts := map[string]any{}
	for k, v := range b.outcomes {
		counts[k] = v
	}
	return map[string]any{"beat": b.outcomes[lanebeat.ReasonBeat], "outcomes": counts}
}

// dispatchLaneBeatSpawnedAt recovers when the supervisor started this worker from
// the runs-dir stem it minted at spawn (`resolve-<issue>-<YYYYMMDD>-<HHMMSS>`,
// written UTC). Falls back to the zero time, which lanebeat treats as "no spawn
// evidence" — disabling the deadline rung rather than inventing a deadline.
func dispatchLaneBeatSpawnedAt(stem string) time.Time {
	base := filepath.Base(stem)
	parts := strings.Split(base, "-")
	if len(parts) < 2 {
		return time.Time{}
	}
	stamp := parts[len(parts)-2] + "-" + parts[len(parts)-1]
	t, err := time.ParseInLocation("20060102-150405", stamp, time.UTC)
	if err != nil {
		return time.Time{}
	}
	return t
}

// dispatchLaneBeatLastOutputAt reads when the worker's log last grew. This is the
// PROGRESS oracle: bytes the OS recorded as a side effect of the worker actually
// producing output, never a field the worker writes about itself. A stat fault
// yields the zero time, which lanebeat folds back to the spawn time.
func dispatchLaneBeatLastOutputAt(log string) time.Time {
	st, err := fsStat(log)
	if err != nil {
		return time.Time{}
	}
	return st.ModTime()
}

// dispatchLaneBeatLiveLeasesDos reads the structurally-live lane-lease set from
// the kernel (`dos lease-lane live`). Note this is the STRUCTURAL fold, not the
// dead-elided contention view — which is correct here: the record we intend to
// refresh is precisely one the elided view may already have dropped, and the
// authority to refresh it comes from lanebeat's rungs, not from the kernel
// having independently agreed the holder is alive.
func dispatchLaneBeatLiveLeasesDos(root string) ([]lanebeat.Lease, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), dispatchLaneBeatTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "dos", "lease-lane", "live")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, false
	}
	var raw []map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(out))), &raw); err != nil {
		return nil, false
	}
	leases := make([]lanebeat.Lease, 0, len(raw))
	for _, r := range raw {
		leases = append(leases, lanebeat.Lease{
			Lane:       dispatchMapString(r, "lane"),
			Holder:     dispatchMapString(r, "holder"),
			HostID:     dispatchMapString(r, "host_id"),
			LoopTS:     dispatchMapString(r, "loop_ts"),
			AcquiredAt: dispatchLaneBeatParseStamp(dispatchMapString(r, "acquired_at")),
		})
	}
	return leases, true
}

// dispatchLaneBeatParseStamp accepts the minute-OR-second ISO stamps the kernel
// emits. An unparseable stamp yields the zero time, which lanebeat refuses on
// (unprovable, not old) rather than treating as ancient.
func dispatchLaneBeatParseStamp(s string) time.Time {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05", "2006-01-02T15:04Z", "2006-01-02T15:04"} {
		if t, err := time.ParseInLocation(layout, s, time.UTC); err == nil {
			return t.UTC()
		}
	}
	return time.Time{}
}

// dispatchLaneBeatWriteDos performs the refresh the decision authorized. The
// identity is passed through byte-for-byte from the matched record: the kernel
// credits a beat by (loop_ts, lane) and authenticates it against the recorded
// holder, so re-deriving any of it would mint a different identity and fold as a
// no-op against the real lease.
func dispatchLaneBeatWriteDos(root string, dec lanebeat.Decision) bool {
	if !dec.Beat || dec.Lane == "" || dec.Owner == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), dispatchLaneBeatTimeout)
	defer cancel()
	args := []string{"lease-lane", "heartbeat", "--lane", dec.Lane, "--owner", dec.Owner,
		"--coalesce-within-s", strconv.Itoa(dispatchLaneBeatCoalesceS)}
	if dec.LoopTS != "" {
		args = append(args, "--loop-ts", dec.LoopTS)
	}
	cmd := exec.CommandContext(ctx, "dos", args...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	return cmd.Run() == nil
}
