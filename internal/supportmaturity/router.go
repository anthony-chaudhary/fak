package supportmaturity

// router.go is the C9 deliverable of the support-maturity epic (#1243, #1252). It turns
// the M0–M7 ladder (supportmaturity.go, #1244) into a DISPATCHER: for any cell's rung it
// emits a routed NEXT-ACTION — the concrete dev-loop that should pick the cell up next,
// plus a one-line directive. The instrument stops being a report and becomes a router of
// the right KIND of work for each cell's regime.
//
// The routing follows the epic's Plane-B dev-regime bands (#1250 / C7 owns the typed
// Regime; this file routes a rung straight to its loop so C9 stands on its own and does
// not couple to C7's in-flight type — RegimeID below is the band's id as a plain string,
// re-wireable to C7's Regime.String() when that lands):
//
//	M0 none                    → R0 explore    → idea-scout       ("can this even work?")
//	M1 fenced, M2 loads, M3 runs → R1 prototype  → dispatch         (get-to-green)
//	M4 correct, M5 optimized    → R2 optimize    → rsiloop+shipgate (make-it-fast toward target)
//	M6 parity, M7 beyond-sota   → R3 production   → self-tax         (#1147 regression gate)
//
// The four loops are exactly the four the issue names, one per regime band; the band→loop
// map is a bijection, so a cell's rung uniquely selects the kind of work to dispatch next.
// This file is the EMIT side; handing the action to cmd/dispatchworker, internal/rsiloop,
// or the idea-scout path is the consuming wiring (#1253, C10, is the R2→rsiloop consumer).
// Keeping EMIT pure is what makes the routing a deterministic, golden-testable fact.

// Loop names the dev-loop a cell's regime routes its next work to — the C9 routing
// (#1252). The four loops are exactly the four named in the issue.
type Loop uint8

const (
	// LoopIdeaScout (R0): open an idea-scout research issue — explore whether the cell
	// can work at all (the tools/idea_scout.py feeder path).
	LoopIdeaScout Loop = iota
	// LoopDispatch (R1): queue a dispatch-worker get-to-green run (cmd/dispatchworker).
	LoopDispatch
	// LoopRSIShipgate (R2): start a long-running rsiloop optimization run gated by
	// shipgate keep/revert toward the target rung (internal/rsiloop + internal/shipgate).
	LoopRSIShipgate
	// LoopSelfTax (R3): hold the self-tax #1147 regression gate — keep a production cell
	// at its rung (drop-on-regression, not advance).
	LoopSelfTax
)

// loopMeta names each loop's stable wire token and the one-line directive a consumer
// dispatches on, indexed by Loop. The token matches the issue's vocabulary; the
// directive is the human-readable next-action.
var loopMeta = []struct{ token, directive string }{
	LoopIdeaScout:   {"idea-scout", "open an idea-scout research issue — can this even work?"},
	LoopDispatch:    {"dispatch", "queue a dispatch-worker get-to-green run"},
	LoopRSIShipgate: {"rsiloop+shipgate", "start an rsiloop+shipgate optimization run toward the target rung"},
	LoopSelfTax:     {"self-tax", "hold the self-tax #1147 regression gate"},
}

// String renders the loop as its stable wire token ("idea-scout", "dispatch", …); an
// out-of-range value renders the explicit unknown form.
func (l Loop) String() string {
	if int(l) < len(loopMeta) {
		return loopMeta[l].token
	}
	return "unknown"
}

// Directive renders the loop's one-line next-action directive. An out-of-range loop has
// no directive ("").
func (l Loop) Directive() string {
	if int(l) < len(loopMeta) {
		return loopMeta[l].directive
	}
	return ""
}

// Valid reports whether l is one of the closed loop targets.
func (l Loop) Valid() bool { return int(l) < len(loopMeta) }

// Loops is the closed roster of dev-loops in regime order — R0's idea-scout through R3's
// self-tax.
var Loops = []Loop{LoopIdeaScout, LoopDispatch, LoopRSIShipgate, LoopSelfTax}

// loopForRegime routes each typed dev-regime to its dev-loop.
var loopForRegime = []Loop{
	R0Explore:    LoopIdeaScout,
	R1Prototype:  LoopDispatch,
	R2Optimize:   LoopRSIShipgate,
	R3Production: LoopSelfTax,
}

// LoopForRegime routes a regime to its dev-loop. Out-of-range regimes floor to
// idea-scout, matching RegimeFor's R0 floor for unwitnessed rungs.
func LoopForRegime(g Regime) Loop {
	if int(g) < len(loopForRegime) {
		return loopForRegime[g]
	}
	return LoopIdeaScout
}

// NextAction is the per-cell routed next-action a rung emits — the C9 unit (#1252). It
// bundles the rung, its dev-regime band id, the loop that band routes to, and the
// human-readable directive, so a single value answers "what is this cell, what regime is
// it in, and which loop should pick it up next?".
type NextAction struct {
	Rung   Rung
	Regime Regime
	// RegimeID is the dev-regime band, "R0".."R3", carried for JSON/readout callers that
	// want the stable token without rendering Regime themselves.
	RegimeID string
	Loop     Loop
	// Action is the loop's one-line directive (Loop.Directive) — the next-action prose.
	Action string
}

// NextActionFor emits the routed next-action for a rung in one deterministic call — the
// dispatcher entry point. The lowering is total: every closed rung, and every
// out-of-range value (which floors to R0/idea-scout — the honest "we can witness nothing,
// so explore" default the From* lowerings in supportmaturity.go share), yields a valid
// action, and calling it twice yields an identical result.
func NextActionFor(r Rung) NextAction {
	g := RegimeFor(r)
	l := LoopForRegime(g)
	return NextAction{Rung: r, Regime: g, RegimeID: g.String(), Loop: l, Action: l.Directive()}
}
