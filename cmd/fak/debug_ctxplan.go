package main

import (
	"flag"
	"fmt"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// debug_ctxplan.go — issue #1574: `fak debug --cmd context-plan-preview`, the DRY-RUN
// surface for a long managed-context run. It answers "what would the context planner keep
// resident, and what would it leave cold?" BEFORE any run starts — without executing or
// mutating anything. This is a pure render layer: it reuses ctxplan's own plan assembly
// (ctxplan.BuildIndex(...).PlanLayout, the same call MaterializeLayout makes) and then
// ctxplan.PreviewOf/PreviewLayout (internal/ctxplan/preview.go) to project the resulting
// Plan into the five product-facing regions — pinned, recent, deep, elided, and
// query-needed — never re-implementing candidate scoring or budget enforcement here.
//
// The command runs entirely against a small synthetic in-memory store built from --intent/
// --pins/--recent/--deep flags (no session/session-image dependency — a context-plan
// preview is a property of a Forecast+Layout+Budget, not of any particular finished
// session), so an operator can preview a hypothetical long-run shape without first
// producing a transcript. No Store.Materialize call is ever made: previewing can never
// page sealed bytes in.

func cmdDebugContextPlanPreview(argv []string) {
	fs := flag.NewFlagSet("debug --cmd context-plan-preview", flag.ExitOnError)
	intents := fs.String("intent", "", "comma-separated forecast intents (predicted reference strings for the run)")
	pins := fs.String("pins", "", "comma-separated span IDs to force resident (Forecast.Pins)")
	budgetTokens := fs.Int("budget-tokens", 512, "resident token budget for the previewed plan")
	recentN := fs.Int("recent", ctxplan.DefaultRecencyWindow, "max spans in the recent area")
	deepN := fs.Int("deep", ctxplan.DefaultDeepSpans, "max spans in the deep area")
	baseN := fs.Int("base", ctxplan.DefaultBaseSpans, "max spans in the base (pinned) area")
	format := fs.String("format", "text", "text | md | json")
	_ = fs.Parse(argv)

	spans := demoLongRunSpans()
	layout := ctxplan.Layout{
		Base:              ctxplan.AreaPolicy{MaxSpans: *baseN, Precision: ctxplan.PrecisionExact},
		Current:           ctxplan.AreaPolicy{MaxSpans: ctxplan.DefaultCurrentSpans, Precision: ctxplan.PrecisionExact},
		Recent:            ctxplan.AreaPolicy{MaxSpans: *recentN, Precision: ctxplan.PrecisionPlanned},
		Deep:              ctxplan.AreaPolicy{MaxSpans: *deepN, Precision: ctxplan.PrecisionPointer},
		IncludeDurability: []string{ctxplan.DurabilityDurable, ctxplan.DurabilityBounded},
		MaxCandidates:     -1,
	}
	f := ctxplan.Forecast{Intents: splitCSV(*intents), Pins: splitCSV(*pins)}
	pv := ctxplan.PreviewLayout(spans, f, ctxplan.Budget{Tokens: *budgetTokens}, nil, layout)

	switch *format {
	case "json":
		fmt.Println(string(jsonIndent(pv)))
	case "md", "markdown":
		fmt.Print(pv.Markdown())
	default:
		fmt.Print(pv.Explain())
	}
}

// demoLongRunSpans builds the committed, deterministic synthetic history a context-plan
// preview runs against by default: a base system pin, an early durable/bounded fact buried
// deep in history, a run of turn-scoped filler (the "long run" bulk that would otherwise
// blow an unbounded window), and a recent+current tail — the same four-area shape
// layout_test.go / preview_test.go use, sized larger to look like an actual long run.
func demoLongRunSpans() []ctxplan.Span {
	st := ctxplan.NewMemStore()
	st.Add("system", ctxplan.DurabilityDurable, []byte("base system prompt: obey the policy, never leak secrets"), false)
	st.Add("user", ctxplan.DurabilityDurable, []byte("standing goal: migrate the billing service off the legacy queue"), false)
	st.Add("WebSearch", ctxplan.DurabilityBounded, []byte("legacy queue migration runbook: drain, cut over, verify, decommission"), false)
	for i := 0; i < 40; i++ {
		st.Add("Bash", ctxplan.DurabilityTurn, []byte("routine build/test log line "+strconv.Itoa(i)), false)
	}
	st.Add("Read", ctxplan.DurabilityTurn, []byte("recent note: cutover window scheduled for tonight"), false)
	st.Add("Bash", ctxplan.DurabilityTurn, []byte("newest tool result: cutover dry run passed"), false)
	spans, _ := st.Spans(ctx())
	return spans
}
