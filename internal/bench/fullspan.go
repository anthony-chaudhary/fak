// fullspan.go is the G5 full-span witnessed trace instrument (issue #2223, epic
// #2218 — risks R5+R6 of docs/notes/DYNAMIC-RANGE-ONE-BINARY-2026-07-01.md).
//
// ONE composed run is traced across four bands of the ladder, each span stamped
// with its band and that band's NOMINAL production clock:
//
//	B6 intent    — a real superloop.Walk fold over a fixture intent (worst-first select)
//	B4 loop tick — a real dispatchtick.EvaluatePreflight fold admitting a worker
//	             for the member the walk selected
//	B2 turn      — one agent turn whose model round-trip is SCRIPTED (no live model):
//	             the proposed call set is fixed, the adjudication and dispatch of
//	             that call set are the real kernel paths
//	B0 decide    — the in-process kernel.Decide fold on each proposed call
//
// The causal chain is real: the walk's worst-first selection names the workspace
// the tick preflights; the tick's SPAWN_OK admits the turn; the turn's proposed
// calls hit the decide fold. Span.Parent records exactly that follows-from /
// contains chain, so cross-band causality (R5) is a walkable edge, not prose.
//
// THE THREE-CLOCKS FENCE (load-bearing): each span's duration is the observed
// cost of fak's OWN fold at that band in THIS run. The bands run on different
// clocks (days / minutes / seconds / µs) dominated by work fak does not own
// (the model, the workers, the operator horizon). Per-band attribution is the
// only readout; nothing here multiplies spans across bands into an end-to-end
// ratio, and the artifact carries no such field by construction
// (docs/notes/EXPLAINER-gate-down-the-stack-2026-06-22.md).
//
// DENY MACRO-COST ATTRIBUTION (R5): every DENY span is stamped with the
// downstream consequence class OBSERVED in the trace — did the turn retry the
// same call (retry_turn), route around the refusal (forked_outcome), or end on
// it (clean_stop)? The classes are counted per refusal reason, so "a correct µs
// DENY that zeros a day of work" becomes a countable rate per policy instead of
// an anecdote. The induced cost recorded is the µs-band cost observed in-trace;
// the production macro-cost of, e.g., a retry turn is a model round-trip at the
// B2 clock — counted as a class, never multiplied into the µs numbers.
package bench

import (
	"context"
	"fmt"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// FullSpanSchema versions the trace artifact.
const FullSpanSchema = "fak.fullspan-trace.v1"

// The closed downstream-consequence vocabulary a DENY span is stamped with —
// what the trace OBSERVED the turn do next after the refusal.
const (
	// ConsequenceRetryTurn: the turn re-proposed the SAME tool after the deny
	// (in production: a model round-trip re-spent at the B2 clock).
	ConsequenceRetryTurn = "retry_turn"
	// ConsequenceForkedOutcome: the turn proposed a DIFFERENT tool next — it
	// routed around the refusal onto another path.
	ConsequenceForkedOutcome = "forked_outcome"
	// ConsequenceCleanStop: nothing followed the deny in the turn — the refusal
	// ended the call set (the SWE-bench empty-patch shape when it recurs
	// turn-over-turn).
	ConsequenceCleanStop = "clean_stop"
)

// Span is one stamped interval of the composed run. Parent is the CAUSAL parent
// span id (-1 = root): B6 selects for B4, B4 admits B2, B2 contains its B0
// decides and dispatches — follows-from/contains, walkable for R5.
type Span struct {
	ID      int    `json:"id"`
	Parent  int    `json:"parent"`
	Band    string `json:"band"`
	Clock   string `json:"clock"` // the band's NOMINAL production clock, not this run's duration
	Name    string `json:"name"`
	StartNS int64  `json:"start_ns"` // offset from the run's monotonic zero
	DurNS   int64  `json:"dur_ns"`
	Tool    string `json:"tool,omitempty"`
	Verdict string `json:"verdict,omitempty"`
	Reason  string `json:"reason,omitempty"`
	// DenyConsequence is stamped on DENY spans by AttributeDenyConsequences.
	DenyConsequence string `json:"deny_consequence,omitempty"`
	Note            string `json:"note,omitempty"`
}

// BandAttribution is the per-band readout: span count and total observed fold
// cost at that band. Per the three-clocks fence these rows are never combined
// across bands.
type BandAttribution struct {
	Band    string `json:"band"`
	Clock   string `json:"clock"`
	Spans   int    `json:"spans"`
	TotalNS int64  `json:"total_ns"`
}

// DenyAttribution is one DENY span's macro-cost row: the consequence class the
// trace observed and the in-trace µs-band cost the refusal induced downstream
// (the follow-up decide+dispatch it provoked; 0 for a clean stop).
type DenyAttribution struct {
	SpanID       int    `json:"span_id"`
	Tool         string `json:"tool"`
	Reason       string `json:"reason"`
	Consequence  string `json:"consequence"`
	DenyNS       int64  `json:"deny_ns"`
	InducedNS    int64  `json:"induced_ns"`
	InducedSpans int    `json:"induced_spans"`
}

// FullSpanTrace is the folded single artifact: one run, four bands, per-band
// attribution, and the deny macro-cost table. It deliberately has NO end-to-end
// or cross-band ratio field.
type FullSpanTrace struct {
	Schema      string `json:"schema"`
	Issue       string `json:"issue"`
	GeneratedAt string `json:"generated_at"`
	Hardware    string `json:"hardware"` // named host (FAK_BENCH_HW), or "unspecified"
	GoOS        string `json:"goos"`
	GoArch      string `json:"goarch"`
	NumCPU      int    `json:"num_cpu"`
	GoVersion   string `json:"go_version"`
	// ScriptedTurn discloses that the B2 model round-trip is scripted: the trace
	// witnesses fak-owned machinery per band; the model's seconds are not run.
	ScriptedTurn bool   `json:"scripted_turn"`
	Fence        string `json:"fence"`

	Spans   []Span            `json:"spans"`
	PerBand []BandAttribution `json:"per_band_attribution"`
	Denies  []DenyAttribution `json:"deny_attribution"`
	// DenyRates counts consequence classes per refusal reason — R5 as a rate.
	DenyRates map[string]map[string]int `json:"deny_rates"`
}

// fenceText is stamped into every artifact so the fence travels with the data.
const fenceText = "per-band attribution only; the four bands run on different clocks and are NEVER multiplied into one end-to-end ratio (three-clocks fence, EXPLAINER-gate-down-the-stack-2026-06-22.md); B2's model round-trip is scripted, so B2 spans witness fak-owned turn machinery, not model seconds"

// scriptedCall is one entry of the fixed proposed call set the scripted turn
// drives. The set is chosen so the default policy produces, deterministically:
// ALLOW, DENY→retry, DENY→fork, ALLOW, DENY→stop — all three consequence
// classes in one turn.
type scriptedCall struct {
	tool string
	args string
	meta map[string]string
}

var readHints = map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}

var scriptedTurn = []scriptedCall{
	{"get_user_details", `{"user_id":"mia_li_3668"}`, readHints},                // ALLOW
	{"write_ledger_entry", `{"amount":1}`, nil},                                 // DENY (DEFAULT_DENY) → same tool next = retry_turn
	{"write_ledger_entry", `{"amount":1,"approved":true}`, nil},                 // DENY (DEFAULT_DENY) → different tool next = forked_outcome
	{"search_direct_flight", `{"origin":"SFO","destination":"JFK"}`, readHints}, // ALLOW (the fork target)
	{"shell_rm_rf", `{"path":"/day-of-work"}`, nil},                             // DENY (POLICY_BLOCK), last = clean_stop
}

// RunFullSpan executes and stamps the composed four-band run against the real
// registered kernel (engine "mock" — the same offline engine `fak bench` replays
// against). It is hermetic: no network, no model, no filesystem writes.
func RunFullSpan(ctx context.Context) (*FullSpanTrace, error) {
	tr := &FullSpanTrace{
		Schema:       FullSpanSchema,
		Issue:        "#2223 (epic #2218, gap G5)",
		GeneratedAt:  time.Now().UTC().Format(time.RFC3339),
		Hardware:     hardwareLabel(),
		GoOS:         runtime.GOOS,
		GoArch:       runtime.GOARCH,
		NumCPU:       runtime.NumCPU(),
		GoVersion:    runtime.Version(),
		ScriptedTurn: true,
		Fence:        fenceText,
	}
	start := time.Now()
	span := func(parent int, band, clock, name string) *Span {
		tr.Spans = append(tr.Spans, Span{
			ID: len(tr.Spans), Parent: parent, Band: band, Clock: clock,
			Name: name, StartNS: time.Since(start).Nanoseconds(),
		})
		return &tr.Spans[len(tr.Spans)-1]
	}
	end := func(s *Span) { s.DurNS = time.Since(start).Nanoseconds() - s.StartNS }

	// --- B6: one real superloop walk fold over a fixture intent ------------
	intent := superloop.Super{
		Name:  "fullspan-fixture-intent",
		Title: "G5 full-span fixture intent",
		About: "a three-member fixture intent whose worst-first selection feeds the B4 tick",
		Members: []superloop.Member{
			{Kind: superloop.KindLoop, Ref: "member-docs", Why: "fixture"},
			{Kind: superloop.KindLoop, Ref: "member-bench", Why: "fixture"},
			{Kind: superloop.KindLoop, Ref: "member-gateway", Why: "fixture"},
		},
	}
	statuses := []superloop.MemberStatus{
		{Member: intent.Members[0], Measured: true, Debt: 1},
		{Member: intent.Members[1], Measured: true, Debt: 7},
		{Member: intent.Members[2], Measured: true, Debt: 3},
	}
	b6 := span(-1, "B6", "days", "superloop.Walk (worst-first select)")
	rep := superloop.Walk(intent, statuses)
	end(b6)
	if len(rep.Worklist) == 0 {
		return tr, fmt.Errorf("fullspan: B6 walk produced an empty worklist")
	}
	selected := rep.Worklist[0].Member.Ref
	b6.Note = "selected worst-first member: " + selected

	// --- B4: one real dispatch-tick preflight fold admitting a worker ------
	b4 := span(b6.ID, "B4", "5-30 min", "dispatchtick.EvaluatePreflight")
	pf := dispatchtick.EvaluatePreflight(dispatchtick.PreflightInput{
		Workspace:  selected,
		MaxWorkers: 2,
		Host:       dispatchtick.HostCheck{Safe: true},
		Account:    dispatchtick.AccountCheck{Available: true, Tag: "fullspan-fixture", Tier: 1, Model: "claude"},
	})
	end(b4)
	b4.Verdict = pf.Verdict
	b4.Note = fmt.Sprintf("workspace=%s cap=%d live=%d", selected, pf.Cap, pf.Live)
	if !pf.OK {
		return tr, fmt.Errorf("fullspan: B4 preflight refused (%s: %s) — turn not admitted", pf.Verdict, pf.Reason)
	}

	// --- B2: one scripted agent turn over the real kernel ------------------
	k := kernel.New("mock")
	res := abi.ActiveResolver()
	b2 := span(b4.ID, "B2", "seconds", "agent turn (scripted call set)")
	b2id := b2.ID
	for _, c := range scriptedTurn {
		ref, err := res.Put(ctx, []byte(c.args))
		if err != nil {
			return tr, fmt.Errorf("fullspan: resolver put: %w", err)
		}
		tc := &abi.ToolCall{Tool: c.tool, Args: ref, Meta: c.meta}

		// B0: the in-process decide fold on the proposed call.
		b0 := span(b2id, "B0", "~0.4-2.4 µs", "kernel.Decide")
		v := k.Decide(ctx, tc)
		end(b0)
		b0.Tool = c.tool
		b0.Verdict = verdictName(v.Kind)
		b0.Reason = abi.ReasonName(v.Reason)

		// The functional dispatch of an admitted call is TURN-band work (the
		// tool executes inside the turn); a refused call never dispatches.
		if v.Kind == abi.VerdictAllow {
			d := span(b2id, "B2", "seconds", "dispatch (Submit+Reap)")
			hdl, _ := k.Submit(ctx, tc)
			if _, err := k.Reap(ctx, hdl); err != nil {
				end(d)
				d.Tool = c.tool
				d.Note = "reap error: " + err.Error()
				continue
			}
			end(d)
			d.Tool = c.tool
		}
	}
	// Close the turn span at index (slice may have been reallocated by append).
	end(&tr.Spans[b2id])

	tr.Denies = AttributeDenyConsequences(tr.Spans)
	// Stamp the class back onto the deny spans and fold the per-reason rates.
	tr.DenyRates = map[string]map[string]int{}
	for _, da := range tr.Denies {
		tr.Spans[da.SpanID].DenyConsequence = da.Consequence
		if tr.DenyRates[da.Reason] == nil {
			tr.DenyRates[da.Reason] = map[string]int{}
		}
		tr.DenyRates[da.Reason][da.Consequence]++
	}
	tr.PerBand = foldPerBand(tr.Spans)
	return tr, nil
}

// AttributeDenyConsequences classifies every DENY span in a stamped span list by
// what the trace shows the turn did next, and attributes the in-trace cost the
// refusal induced. Pure fold; exported so the classifier is testable on
// synthetic traces and reusable on future live-captured ones.
//
//	next proposed call in the same turn has the SAME tool  → retry_turn
//	next proposed call in the same turn is a DIFFERENT tool → forked_outcome
//	no further proposed call in the turn                    → clean_stop
//
// Induced cost: the next proposed call's decide span plus its dispatch span (if
// any) — the observed work the refusal provoked. A clean stop induces nothing.
func AttributeDenyConsequences(spans []Span) []DenyAttribution {
	var out []DenyAttribution
	for i, s := range spans {
		if s.Band != "B0" || s.Verdict != "DENY" {
			continue
		}
		da := DenyAttribution{SpanID: s.ID, Tool: s.Tool, Reason: s.Reason, DenyNS: s.DurNS, Consequence: ConsequenceCleanStop}
		for j := i + 1; j < len(spans); j++ {
			n := spans[j]
			if n.Band != "B0" || n.Parent != s.Parent {
				continue
			}
			if n.Tool == s.Tool {
				da.Consequence = ConsequenceRetryTurn
			} else {
				da.Consequence = ConsequenceForkedOutcome
			}
			da.InducedNS = n.DurNS
			da.InducedSpans = 1
			// A dispatch span for the follow-up call is also induced work.
			for _, d := range spans[j+1:] {
				if d.Parent == n.Parent && d.Band == "B2" && d.Tool == n.Tool && d.StartNS >= n.StartNS {
					da.InducedNS += d.DurNS
					da.InducedSpans++
					break
				}
			}
			break
		}
		out = append(out, da)
	}
	return out
}

func foldPerBand(spans []Span) []BandAttribution {
	order := []string{"B6", "B4", "B2", "B0"}
	byBand := map[string]*BandAttribution{}
	for _, s := range spans {
		b := byBand[s.Band]
		if b == nil {
			b = &BandAttribution{Band: s.Band, Clock: s.Clock}
			byBand[s.Band] = b
		}
		b.Spans++
		b.TotalNS += s.DurNS
	}
	var out []BandAttribution
	for _, name := range order {
		if b := byBand[name]; b != nil {
			out = append(out, *b)
		}
	}
	return out
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "REQUIRE_WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	}
	return fmt.Sprintf("KIND_%d", k)
}
