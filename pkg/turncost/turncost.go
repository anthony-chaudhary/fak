// Package turncost defines fak's whole-engine per-turn cost surface: one typed
// record whose phases split an agent turn's wall time across the serving seams —
// admission, prefill, decode, stream emit, upstream proxy hop, and ledger I/O.
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/turncost) on
// purpose: the private serving repo may import fak/pkg/* only and never
// fak/internal/* (Core Import Invariant), yet a per-turn record must span BOTH
// repos — the public gateway plans and decodes, the private proxy makes the
// upstream hop. Placing the record here lets the same value type, phase
// vocabulary, and Prometheus family be produced on either side of the boundary
// and joined on the `/metrics` wire.
//
// Two invariants shape the surface:
//
//   - Additive & fail-open (P2): the record is pure observability. It changes no
//     request parsing, admission, streaming, or routing behavior; every hook is
//     best-effort and a nil Collector or disabled gate is a no-op.
//   - Zero hardcoded fabrication (P4): a phase with no timing available is left
//     ABSENT (zero duration, zero samples), never estimated. The family reports
//     only what a seam actually measured.
//
// The package has no external dependencies and holds no global state: a
// Collector is constructed by its owner and folded into that owner's existing
// hand-rolled Prometheus exposition.
package turncost

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Phase is one attributable segment of a turn's wall time. The six phases below
// are the closed vocabulary; a phase with no measurement contributes nothing.
type Phase string

const (
	// PhaseAdmission is time spent waiting for a serving slot / token budget
	// before generation starts.
	PhaseAdmission Phase = "admission"
	// PhasePrefill is the prompt/prefill pass before the first generated token.
	PhasePrefill Phase = "prefill"
	// PhaseDecode is token generation after prefill.
	PhaseDecode Phase = "decode"
	// PhaseStream is time spent serializing and flushing SSE deltas to the client.
	PhaseStream Phase = "stream"
	// PhaseProxyHop is the round-trip through the upstream proxy hop (private
	// serving path) or the direct upstream call.
	PhaseProxyHop Phase = "proxy_hop"
	// PhaseLedger is time spent on the per-turn session-ledger append.
	PhaseLedger Phase = "ledger"
)

// Phases is the closed, ordered phase vocabulary. Prometheus exposition emits
// one series per phase that has at least one observation, in this order.
var Phases = [...]Phase{
	PhaseAdmission,
	PhasePrefill,
	PhaseDecode,
	PhaseStream,
	PhaseProxyHop,
	PhaseLedger,
}

// Valid reports whether p is a member of the closed phase vocabulary.
func (p Phase) Valid() bool {
	for _, known := range Phases {
		if p == known {
			return true
		}
	}
	return false
}

// TurnCostRecord is the typed per-turn cost surface. A zero value is a valid,
// entirely-unmeasured record: DurationSeconds for any phase that was not timed
// is left at zero rather than guessed, so a consumer can tell "not measured"
// from "measured as ~0" only by the accompanying Samples count.
type TurnCostRecord struct {
	// Trace is the per-turn trace/request id this record attributes. Optional:
	// it is the join key when several records are folded, and is NOT a metric
	// label (bounded-label invariant — see Collector).
	Trace string `json:"trace,omitempty"`
	// Model is the response model, an informational field only.
	Model string `json:"model,omitempty"`
	// Streaming is true when the turn was served as an SSE stream.
	Streaming bool `json:"streaming,omitempty"`
	// TotalSeconds is the whole-turn wall time, when the caller measured it.
	TotalSeconds float64 `json:"total_seconds,omitempty"`
	// Phases maps each measured phase to its duration in seconds. A missing key
	// means that phase was not measured for this turn (absent, not zero).
	Phases map[Phase]float64 `json:"phases,omitempty"`
}

// Add accumulates d seconds into the named phase. A non-positive duration is
// ignored (a phase that took no measurable time is left absent), which keeps the
// record honest under clock granularity without fabricating a sample.
func (r *TurnCostRecord) Add(p Phase, d time.Duration) {
	if d <= 0 || !p.Valid() {
		return
	}
	if r.Phases == nil {
		r.Phases = make(map[Phase]float64, len(Phases))
	}
	r.Phases[p] += d.Seconds()
}

// SetPhase records d seconds for the named phase, replacing any prior value.
func (r *TurnCostRecord) SetPhase(p Phase, d time.Duration) {
	if d < 0 || !p.Valid() {
		return
	}
	if r.Phases == nil {
		r.Phases = make(map[Phase]float64, len(Phases))
	}
	r.Phases[p] = d.Seconds()
}

// Phase returns the recorded seconds for p and whether it was measured.
func (r *TurnCostRecord) Phase(p Phase) (float64, bool) {
	if r == nil || r.Phases == nil {
		return 0, false
	}
	v, ok := r.Phases[p]
	return v, ok
}

// MeasuredPhases returns the phases present in the record, in canonical order.
func (r *TurnCostRecord) MeasuredPhases() []Phase {
	if r == nil || len(r.Phases) == 0 {
		return nil
	}
	out := make([]Phase, 0, len(r.Phases))
	for _, p := range Phases {
		if _, ok := r.Phases[p]; ok {
			out = append(out, p)
		}
	}
	return out
}

// Enabled reports whether per-turn cost exposure is switched on. It is
// opt-in: the surface is OFF unless explicitly enabled with a truthy value,
// matching the additive-observability contract. Both repos share this one gate.
func Enabled() bool {
	return truthy(os.Getenv("FAK_TURN_COST"))
}

// EnabledStreaming is the streaming-surface gate. It defaults to the value of
// FAK_TURN_COST so a single switch controls the family, but may be overridden
// independently for the streaming path.
func EnabledStreaming() bool {
	if v, ok := os.LookupEnv("FAK_TURN_COST_STREAM"); ok {
		return truthy(v)
	}
	return Enabled()
}

func truthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "0", "false", "no", "off", "n":
		return false
	default:
		return true
	}
}

// Collector aggregates turn cost records into a bounded Prometheus surface.
//
// Bounded-label invariant: the exposition family is keyed ONLY by the closed
// phase vocabulary (and a coarse streaming/surface dimension), never by trace,
// model, tenant, or request identity. A high-cardinality label would let an
// external turn id blow up the scrape; the per-turn detail lives in the record
// (native receipts, debug lines), not in the metric labels.
//
// A nil *Collector is safe to use: Observe is a no-op and Prometheus is empty,
// so a caller may wire it unconditionally and let the flag decide exposure.
type Collector struct {
	mu sync.Mutex
	// totals[surface][phase] accumulates observed seconds.
	totals map[string]map[Phase]float64
	// counts[surface][phase] counts observations, so an absent phase is
	// distinguishable from a measured-but-tiny one.
	counts map[string]map[Phase]uint64
	// turns[surface] counts completed turns that contributed a record.
	turns map[string]uint64
}

// surge names the closed surface dimension: "stream" or "buffered".
const (
	surfaceStream   = "stream"
	surfaceBuffered = "buffered"
)

// NewCollector returns an empty, ready-to-use Collector.
func NewCollector() *Collector {
	return &Collector{
		totals: make(map[string]map[Phase]float64, 2),
		counts: make(map[string]map[Phase]uint64, 2),
		turns:  make(map[string]uint64, 2),
	}
}

func surfaceOf(streaming bool) string {
	if streaming {
		return surfaceStream
	}
	return surfaceBuffered
}

// Observe folds one completed turn's record into the aggregate. A nil collector
// or nil record is a no-op, and the record's Trace/Model are deliberately
// ignored (bounded labels). Missing phases contribute nothing.
func (c *Collector) Observe(rec *TurnCostRecord) {
	if c == nil || rec == nil {
		return
	}
	surface := surfaceOf(rec.Streaming)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.turns == nil {
		c.turns = make(map[string]uint64, 2)
	}
	c.turns[surface]++
	if len(rec.Phases) == 0 {
		return
	}
	if c.totals == nil {
		c.totals = make(map[string]map[Phase]float64, 2)
	}
	if c.counts == nil {
		c.counts = make(map[string]map[Phase]uint64, 2)
	}
	tot, ok := c.totals[surface]
	if !ok {
		tot = make(map[Phase]float64, len(Phases))
		c.totals[surface] = tot
	}
	cnt, ok := c.counts[surface]
	if !ok {
		cnt = make(map[Phase]uint64, len(Phases))
		c.counts[surface] = cnt
	}
	for _, p := range Phases {
		secs, present := rec.Phases[p]
		if !present {
			continue
		}
		tot[p] += secs
		cnt[p]++
	}
}

// Prometheus renders the fak_turn_cost_* family in the hand-rolled text
// exposition format used elsewhere in fak. It emits:
//
//	fak_turn_cost_seconds_total{surface,phase}  — cumulative seconds per phase
//	fak_turn_cost_phase_samples_total{surface,phase} — observation count
//	fak_turn_cost_turns_total{surface}          — completed turns observed
//
// Surfaces and phases with no data emit no series (absent, never zero-filled).
func (c *Collector) Prometheus() string {
	if c == nil {
		return ""
	}
	c.mu.Lock()
	totals := make(map[string]map[Phase]float64, len(c.totals))
	for s, m := range c.totals {
		cp := make(map[Phase]float64, len(m))
		for p, v := range m {
			cp[p] = v
		}
		totals[s] = cp
	}
	counts := make(map[string]map[Phase]uint64, len(c.counts))
	for s, m := range c.counts {
		cp := make(map[Phase]uint64, len(m))
		for p, v := range m {
			cp[p] = v
		}
		counts[s] = cp
	}
	turns := make(map[string]uint64, len(c.turns))
	for s, v := range c.turns {
		turns[s] = v
	}
	c.mu.Unlock()

	if len(turns) == 0 {
		return ""
	}

	var b strings.Builder
	writeHelpType(&b, "fak_turn_cost_turns_total",
		"Completed agent turns observed by the per-turn cost surface, by transport.", "counter")
	for _, s := range sortedSurfaces(turns) {
		fmt.Fprintf(&b, "fak_turn_cost_turns_total{surface=%q} %d\n", s, turns[s])
	}

	writeHelpType(&b, "fak_turn_cost_seconds_total",
		"Cumulative measured wall-clock seconds per turn phase, by transport.", "counter")
	writeHelpType(&b, "fak_turn_cost_phase_samples_total",
		"Turn-phase timing observations contributing to fak_turn_cost_seconds_total.", "counter")
	for _, s := range sortedSurfaces(totals) {
		for _, p := range Phases {
			secs, ok := totals[s][p]
			if !ok {
				continue
			}
			fmt.Fprintf(&b, "fak_turn_cost_seconds_total{surface=%q,phase=%q} %s\n",
				s, string(p), promFloat(secs))
			fmt.Fprintf(&b, "fak_turn_cost_phase_samples_total{surface=%q,phase=%q} %d\n",
				s, string(p), counts[s][p])
		}
	}
	return b.String()
}

func sortedSurfaces[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for s := range m {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool {
		// stream before buffered, then lexical — a stable, readable order.
		if out[i] == surfaceStream {
			return true
		}
		if out[j] == surfaceStream {
			return false
		}
		return out[i] < out[j]
	})
	return out
}

// writeHelpType mirrors the host exposition helpers: a HELP then TYPE line.
func writeHelpType(b *strings.Builder, name, help, typ string) {
	fmt.Fprintf(b, "# HELP %s %s\n", name, help)
	fmt.Fprintf(b, "# TYPE %s %s\n", name, typ)
}

func promFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}
