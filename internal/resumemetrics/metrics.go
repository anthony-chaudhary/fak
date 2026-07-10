// Package resumemetrics is the PROCESS-GLOBAL expvar surface for the resume/heal watchdog
// (#3803). The watchdog verdict counts used to be reconstructable only by re-reading the
// durable status ledger from disk — so a renamed, rotated, or missing ledger silently read as
// "zero activity", indistinguishable from a healthy-but-idle watchdog. These counters are the
// authoritative IN-PROCESS record instead: every watchdog tick, every launch/skip verdict,
// every autoheal result, and every drain-steward progress witness increments a named
// expvar the moment the decision is made, independent of whether any ledger write lands.
//
// It is deliberately a tiny leaf, not Server state, because its two writers live in different
// places: the standalone `fak resume watchdog` tick and the in-guard autoheal boot path both
// call these funcs, and the gateway's /debug/vars handler folds Snapshot() into its JSON. A
// per-Server metric struct (like gateway.resumeProjMetrics) could not be reached from the
// standalone tick; a package-global expvar can. The vars register on the standard expvar
// registry, so a process that serves expvar also exposes them for free.
//
// Cross-process caveat: expvar is per-process memory. The autoheal path runs INSIDE the guard
// that hosts the gateway, so its counts show up in that gateway's /debug/vars directly. The
// standalone `fak resume watchdog` tick runs in its own short-lived process; its counts are
// authoritative there, and surface wherever that process publishes expvar. This leaf makes the
// counts real at the point of decision; it does not add a new endpoint to serve them.
package resumemetrics

import (
	"expvar"
	"strings"
)

// Published expvar names. Kept as consts so the /debug/vars fold and any external scraper key
// on the same strings the tests assert.
const (
	VarTicks    = "fak_resume_watchdog_ticks"      // total watchdog ticks observed
	VarActions  = "fak_resume_watchdog_action"     // per-verdict launch/skip/defer count
	VarAutoheal = "fak_watchdog_autoheal_result"   // per-result autoheal boot outcome count
	VarMonitor  = "fak_watchdog_monitor_status"    // per-monitor last folded status
	VarRollup   = "fak_watchdoghealth_status"       // the folded cross-monitor rollup status
	VarProgress = "fak_resume_watchdog_progress"    // drain-steward progress witnesses recorded
)

var (
	ticks    = expvar.NewInt(VarTicks)
	actions  = expvar.NewMap(VarActions)
	autoheal = expvar.NewMap(VarAutoheal)
	monitor  = expvar.NewMap(VarMonitor)
	rollup   = expvar.NewString(VarRollup)
	progress = expvar.NewInt(VarProgress)
)

// norm collapses a closed token to its published form: trimmed, lowercased, and never empty
// (an empty verdict token would create a bucket no reader can name, so it folds to "unknown").
func norm(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return "unknown"
	}
	return s
}

// Tick records one watchdog sweep. Bumped once per tick so a live-but-idle watchdog (all SKIPs,
// no launches) is distinguishable from a dead one — zero ticks means the watchdog never ran,
// which is the exact signal the ledger-derived count used to lose.
func Tick() { ticks.Add(1) }

// RecordAction counts one per-session watchdog verdict (the resume.WatchdogAction token:
// launch / skip / defer / …). This is the counter the acceptance test drives per verdict path.
func RecordAction(action string) { actions.Add(norm(action), 1) }

// RecordAutohealResult counts one autoheal boot-path outcome (the closed result token the boot
// heal reports for each monitor it (re)started or found healthy).
func RecordAutohealResult(result string) { autoheal.Add(norm(result), 1) }

// SetMonitorStatus records a monitor's most recent folded status (last-writer-wins, not a
// count — a status is a level, not an event). Stored as an expvar string so /debug/vars shows
// {"monitor_name":"status"} directly.
func SetMonitorStatus(monitorName, status string) {
	sv := new(expvar.String)
	sv.Set(norm(status))
	monitor.Set(norm(monitorName), sv)
}

// SetHealthRollup records the folded cross-monitor rollup status (the single overall verdict).
func SetHealthRollup(status string) { rollup.Set(norm(status)) }

// ProgressWitnessed records one drain-steward progress row — a resume proven to have produced a
// real post-launch turn. It is the live twin of the ledger's "progress" phase rows.
func ProgressWitnessed() { progress.Add(1) }

// Snapshot is a typed, JSON-friendly copy of the current counters for the gateway /debug/vars
// fold and the unit tests. Reading is lock-free: each expvar is independently synchronized, and
// a snapshot need not be atomic across vars for a metrics readout.
type Snapshot struct {
	Ticks             int64             `json:"ticks"`
	ProgressWitnessed int64             `json:"progress_witnessed"`
	Actions           map[string]int64  `json:"actions,omitempty"`
	AutohealResults   map[string]int64  `json:"autoheal_results,omitempty"`
	MonitorStatus     map[string]string `json:"monitor_status,omitempty"`
	HealthRollup      string            `json:"health_rollup,omitempty"`
}

// Read returns the current counters as a typed Snapshot.
func Read() Snapshot {
	return Snapshot{
		Ticks:             ticks.Value(),
		ProgressWitnessed: progress.Value(),
		Actions:           countMap(actions),
		AutohealResults:   countMap(autoheal),
		MonitorStatus:     statusMap(monitor),
		HealthRollup:      rollup.Value(),
	}
}

// Active reports whether any watchdog signal has been recorded this process — the gateway uses
// it to omit the /debug/vars block entirely on a cold process that never ran a watchdog, the
// same nil-when-empty convention the other optional /debug/vars blocks keep.
func Active() bool {
	if ticks.Value() > 0 || progress.Value() > 0 || rollup.Value() != "" {
		return true
	}
	return mapLen(actions) > 0 || mapLen(autoheal) > 0 || mapLen(monitor) > 0
}

// Reset clears every counter. Test-only: production code only ever increments. It lets a test
// assert a verdict path moved its own counter from a known-zero floor without cross-test bleed
// through the shared process-global registry.
func Reset() {
	ticks.Set(0)
	progress.Set(0)
	rollup.Set("")
	actions.Init()
	autoheal.Init()
	monitor.Init()
}

func countMap(m *expvar.Map) map[string]int64 {
	out := map[string]int64{}
	m.Do(func(kv expvar.KeyValue) {
		if iv, ok := kv.Value.(*expvar.Int); ok {
			out[kv.Key] = iv.Value()
		}
	})
	if len(out) == 0 {
		return nil
	}
	return out
}

func statusMap(m *expvar.Map) map[string]string {
	out := map[string]string{}
	m.Do(func(kv expvar.KeyValue) {
		if sv, ok := kv.Value.(*expvar.String); ok {
			out[kv.Key] = sv.Value()
		}
	})
	if len(out) == 0 {
		return nil
	}
	return out
}

func mapLen(m *expvar.Map) int {
	n := 0
	m.Do(func(expvar.KeyValue) { n++ })
	return n
}
