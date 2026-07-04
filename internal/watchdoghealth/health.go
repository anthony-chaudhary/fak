// Package watchdoghealth is the pure health-digest core for fak's DEFAULT watchdog
// monitors — the OS-scheduled fleet timers (resume, supervisor, dos-dispatch,
// stale-work-garden) that `cmd/fak`'s watchdog-autoheal keeps alive on every `fak serve`
// / `fak guard` boot.
//
// # Why this leaf exists
//
// The autoheal is fire-and-forget: it probes each monitor, restarts a dead-but-installed
// one under a backoff policy, and PERSISTS a per-monitor heal-state (attempts, last
// restart, last reason). But nothing ever reads that state back for an operator: the only
// way to learn whether the default monitors are alive was to scrape the JSON the last boot
// wrote to stderr / autoheal.log. This leaf folds the two facts a shell can gather — the
// live probe and the persisted heal-state — into one inspectable digest so `fak watchdog
// status` can answer "are my default monitors healthy right now, and does the layer need a
// human?" from a closed status vocabulary.
//
// # What stays in the shell
//
// Everything with a clock or a side effect: running the probe command, reading the
// heal-state file, and formatting timestamps. This core reads no clock and does no I/O —
// same monitors in, same digest out — so it is table-testable in isolation and cannot be
// reddened by churn in the (large, contended) cmd/fak package it serves.
package watchdoghealth

import "sort"

// Schema is the digest's self-describing version tag.
const Schema = "fak.watchdog-health.v1"

// Status is the closed per-monitor health vocabulary. A caller must never invent a token
// outside this set — the whole point of the digest is that every state an operator can see
// maps to one of these, with a defined severity for the rollup.
type Status string

const (
	// StatusHealthy: the monitor is installed and the OS reports it alive (Ready/Running,
	// active, or a loaded launchd job). Nothing to do.
	StatusHealthy Status = "HEALTHY"
	// StatusNotInstalled: the monitor's timer is not installed on this host. This is NOT an
	// error — a box that never registered the timer is a deliberate no-op for the autoheal,
	// so it is the LEAST severe status, below healthy, and never trips NeedsAttention.
	StatusNotInstalled Status = "NOT_INSTALLED"
	// StatusHealing: installed, not alive, but a restart has already been attempted (a
	// recorded restart or a non-exhausted attempt streak). The autoheal is mid-recovery;
	// self-correcting, so it is milder than a fresh DOWN.
	StatusHealing Status = "HEALING"
	// StatusDown: installed, not alive, and NO heal has been recorded yet — a fresh death
	// the next autoheal tick will pick up. Actionable, but the machine is expected to fix it.
	StatusDown Status = "DOWN"
	// StatusGaveUp: installed, not alive, and the restart streak is exhausted (attempts have
	// reached the give-up cap). The machine has stopped trying — this needs a human.
	StatusGaveUp Status = "GAVE_UP"
	// StatusUnknown: the probe itself failed, so liveness could not be read. Attention is
	// owed because the state is genuinely unknown, not because it is known-bad.
	StatusUnknown Status = "UNKNOWN"
)

// severity orders the statuses for the worst-of rollup. NOT_INSTALLED sits BELOW healthy so
// a fleet of some-installed, some-not rolls up to HEALTHY (the healthy ones win) while an
// all-not-installed host rolls up to NOT_INSTALLED. Anything at or above attentionFloor
// (DOWN) means the layer is not simply healing itself.
func severity(s Status) int {
	switch s {
	case StatusNotInstalled:
		return 0
	case StatusHealthy:
		return 1
	case StatusHealing:
		return 2
	case StatusDown:
		return 3
	case StatusUnknown:
		return 4
	case StatusGaveUp:
		return 5
	default:
		return 4 // an unrecognized status is treated as unknown, never as healthy
	}
}

// attentionFloor is the lowest severity that means a human (or at least a closer look) is
// owed rather than the autoheal quietly self-correcting. HEALING (severity 2) is below it;
// DOWN / UNKNOWN / GAVE_UP are at or above.
const attentionFloor = 3

// Monitor is the per-monitor input a shell gathers: the identity of one default watchdog,
// the live probe result, and the fields it persisted in its heal-state. The shell maps its
// own probe/state types onto this so the core stays free of cmd/fak's dependencies.
type Monitor struct {
	// ID is the stable monitor id (e.g. "fleet-resume-watchdog").
	ID string `json:"id"`
	// Manager is the OS scheduler backing it ("taskscheduler" | "launchd" | "systemd").
	Manager string `json:"manager,omitempty"`
	// Unit is the scheduler's unit/task/label name.
	Unit string `json:"unit,omitempty"`

	// ProbeErr is true when the probe command itself failed — liveness is unknown.
	ProbeErr bool `json:"probe_err,omitempty"`
	// Installed / Alive are the probe verdict (meaningful only when ProbeErr is false).
	Installed bool `json:"installed"`
	Alive     bool `json:"alive"`
	// Detail is the probe's human one-liner (scheduler state string), passed through.
	Detail string `json:"detail,omitempty"`

	// Attempts is the persisted restart-attempt streak (heal-state).
	Attempts uint64 `json:"attempts,omitempty"`
	// MaxAttempts is the autoheal's give-up cap. <= 0 means the caller could not supply it,
	// which leaves GAVE_UP undecidable — a non-alive monitor with attempts then reads
	// HEALING (the machine is presumed still trying) rather than falsely GAVE_UP.
	MaxAttempts uint64 `json:"max_attempts,omitempty"`
	// LastRestartUnixNano / LastFailureUnixNano / LastProbeAliveUnixNano are the persisted
	// timestamps, passed through for the shell to format. The core only checks whether a
	// restart was ever recorded (> 0), never a duration.
	LastRestartUnixNano    int64 `json:"last_restart_unix_nano,omitempty"`
	LastFailureUnixNano    int64 `json:"last_failure_unix_nano,omitempty"`
	LastProbeAliveUnixNano int64 `json:"last_probe_alive_unix_nano,omitempty"`
	// LastReason is the persisted last heal reason token, passed through.
	LastReason string `json:"last_reason,omitempty"`
}

// Classify folds one monitor's probe + heal-state into its closed Status. The order is the
// autoheal's own precedence: a failed probe is UNKNOWN before anything; a live monitor is
// HEALTHY; a not-installed one is a no-op; and an installed-but-dead one is graded by how
// far its restart streak has run.
func Classify(m Monitor) Status {
	if m.ProbeErr {
		return StatusUnknown
	}
	if m.Alive {
		return StatusHealthy
	}
	if !m.Installed {
		return StatusNotInstalled
	}
	// Installed but not alive: grade by the recovery streak.
	if m.MaxAttempts > 0 && m.Attempts >= m.MaxAttempts {
		return StatusGaveUp
	}
	if m.Attempts > 0 || m.LastRestartUnixNano > 0 {
		return StatusHealing
	}
	return StatusDown
}

// Health is one monitor plus its classified status.
type Health struct {
	Monitor
	Status Status `json:"status"`
}

// NeedsAttention reports whether this monitor's status is at or above the attention floor
// (DOWN / UNKNOWN / GAVE_UP) — i.e. not merely healthy, not-installed, or self-healing.
func (h Health) NeedsAttention() bool { return severity(h.Status) >= attentionFloor }

// Digest is the fleet-wide fold: every monitor's health, the worst-of rollup, per-status
// counts, and the one-bit "does a human need to look?" gate.
type Digest struct {
	Schema string `json:"schema"`
	// Monitors is the per-monitor health, in the input order.
	Monitors []Health `json:"monitors"`
	// Rollup is the worst-of status across all monitors (StatusNotInstalled for an empty
	// input — nothing installed is not a problem).
	Rollup Status `json:"rollup"`
	// Counts is the number of monitors in each status (only non-zero entries).
	Counts map[Status]int `json:"counts,omitempty"`
	// NeedsAttention is true when any monitor is at or above the attention floor. This is
	// the `fak watchdog status --check` exit-code condition.
	NeedsAttention bool `json:"needs_attention"`
}

// Fold classifies every monitor and rolls the results up into a Digest. Pure and total over
// any input: an empty slice yields a NOT_INSTALLED, no-attention digest.
func Fold(monitors []Monitor) Digest {
	d := Digest{
		Schema:   Schema,
		Monitors: make([]Health, 0, len(monitors)),
		Rollup:   StatusNotInstalled,
		Counts:   map[Status]int{},
	}
	for _, m := range monitors {
		h := Health{Monitor: m, Status: Classify(m)}
		d.Monitors = append(d.Monitors, h)
		d.Counts[h.Status]++
		if severity(h.Status) > severity(d.Rollup) {
			d.Rollup = h.Status
		}
		if h.NeedsAttention() {
			d.NeedsAttention = true
		}
	}
	return d
}

// Statuses returns the closed status vocabulary in severity order (least to most severe).
// Exposed so a renderer or a doc generator can list every status without hard-coding it.
func Statuses() []Status {
	all := []Status{
		StatusNotInstalled, StatusHealthy, StatusHealing,
		StatusDown, StatusUnknown, StatusGaveUp,
	}
	sort.SliceStable(all, func(i, j int) bool { return severity(all[i]) < severity(all[j]) })
	return all
}
