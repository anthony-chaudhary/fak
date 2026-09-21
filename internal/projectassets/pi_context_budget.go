package projectassets

// pi_context_budget.go — the SAFE resident-context envelope fak writes into Pi's
// configuration. The doctrine lives in docs/long-context-defaults.md: the provider's
// advertised context window is a hard CAP, never the target resident budget. Before
// this file the Pi writers advertised the raw cap (contextWindow: 131072) and let Pi
// fill it, so a long session sailed into the window with only Pi's small default
// output reserve between it and a hard overflow — the "cap is not target" violation.
//
// Two numbers are written here, and both are DERIVED from the served window:
//
//   - contextWindow: the resident target, held at or below half the served window.
//     Pi checks `contextTokens > contextWindow - reserveTokens` to decide when to
//     auto-compact (docs/compaction.md), so a smaller contextWindow makes Pi compact
//     inside the safe envelope instead of at the ceiling.
//   - compaction.reserveTokens / keepRecentTokens (see pi_settings.go): the explicit
//     output reserve and the recent-token floor the summary keeps, so a compaction that
//     fires does not immediately re-cross the threshold.
//
// The relationship is deliberate and monotone: target = min(window, window/2). It never
// returns the raw window, matching EffectiveContextEnvelope.Target() in internal/ctxplan,
// and it is provenance-labelled MODELED (a doctrine-derived prior) rather than WITNESSED.

// PiSafeWindowNumerator/Denominator express the doctrine fraction: a Pi session should be
// configured to keep its resident target at or below 50% of the served window. Keeping it
// as a ratio (not a literal 0.5 at the call site) lets the doctrine change in one place and
// gives the tests a named constant to assert against.
const (
	PiSafeWindowNumerator   = 1
	PiSafeWindowDenominator = 2
)

// PiBudgetProvenance labels the evidence class behind the written budget. It mirrors
// ctxplan's labels but is kept local so projectassets does not need to import ctxplan for
// a string. MODELED means "doctrine-derived prior, not a same-task fak witness" — the same
// honest label DefaultEnvelopes() carries.
const PiBudgetProvenance = "MODELED"

// DefaultPiServedWindow is the window assumed when the caller does not know the served
// model's real context window (no backend probe, no explicit flag). It matches the
// historical literal the writers used, so an unknown window degrades to the previous
// number as the CAP and still halves it for the resident target.
const DefaultPiServedWindow = 131072

// MinPiResidentTarget is the floor under the derived resident target. A tiny served window
// (say 8192) still needs enough room for the objective, the recent transcript, and a tool
// result; a target below this would make Pi thrash. Half of 8192 is 4096, so this floor is
// only reached by a model smaller than ~8K and is a safety rail, not the common path.
const MinPiResidentTarget = 4096

// MaxPiOutputReserve caps the derived output reserve (written as Pi's compaction
// reserveTokens and as the model's maxTokens). Pi's own default is 16384 and the guard's
// OpenCode fraction is bounded the same way, so the cap keeps a very large resident target
// (the 500k DeepSeek window) from reserving an absurd slab of output tokens: a 500k target
// divides to 125000 without the cap, which would leave the model an output budget no
// backend accepts. The cap is the OUTPUT half of the cap-vs-target split and does not
// shrink the resident target itself.
const MaxPiOutputReserve = 32768

// PiContextBudget is the derived safe envelope written into Pi's configuration. All three
// numbers are computed from ServedWindow; none is a raw literal at a call site.
type PiContextBudget struct {
	// ServedWindow is the hard cap the backend advertises (input + output).
	ServedWindow int
	// ResidentTarget is what is written as Pi's contextWindow: at most half the served
	// window, so Pi auto-compacts well before the hard cap.
	ResidentTarget int
	// OutputReserve is written as compaction.reserveTokens — the tokens held back for the
	// model's answer and tool arguments. Pi subtracts this before comparing to the window.
	OutputReserve int
	// KeepRecentTokens is written as compaction.keepRecentTokens — the recent transcript a
	// compaction must preserve, so a fire leaves usable working context behind it.
	KeepRecentTokens int
	// Provenance labels the evidence class (MODELED: doctrine prior, not a fak witness).
	Provenance string
}

// PiSafeContextBudget derives the safe Pi context envelope from a served window. A
// non-positive or absurdly small window falls back to DefaultPiServedWindow so a caller
// that passed nothing still gets the doctrine-consistent halving rather than a zero.
//
// The three derived quantities:
//   - ResidentTarget = min(window, window/2), floored at MinPiResidentTarget.
//   - OutputReserve  = ResidentTarget/4, clamped to [512, MaxPiOutputReserve] — the same
//     bounded fraction guard uses for OpenCode's output limit (guard_opencode.go), so the
//     two harnesses are configured on one rule instead of two.
//   - KeepRecentTokens = ResidentTarget/2, floored at 2000 — half the resident target is
//     kept as working context, matching Pi's own default ratio (20k kept in a 128k model
//     is ~1/6; half is the conservative end that keeps a fire from rebounding).
func PiSafeContextBudget(servedWindow int) PiContextBudget {
	window := servedWindow
	if window < MinPiResidentTarget*2 {
		// Too small to halve meaningfully (or unset/negative): use the doctrine default.
		// The check is `< MinResidentTarget*2` so a legitimate 8192 window (target 4096)
		// still takes the halving path.
		window = DefaultPiServedWindow
	}

	target := window * PiSafeWindowNumerator / PiSafeWindowDenominator
	if target > window {
		target = window
	}
	if target < MinPiResidentTarget {
		target = MinPiResidentTarget
	}

	reserve := target / 4
	if reserve > MaxPiOutputReserve {
		reserve = MaxPiOutputReserve
	}
	if reserve < 512 {
		reserve = 512
	}

	keep := target / 2
	if keep < 2000 {
		keep = 2000
	}
	if keep > target {
		keep = target
	}

	return PiContextBudget{
		ServedWindow:     window,
		ResidentTarget:   target,
		OutputReserve:    reserve,
		KeepRecentTokens: keep,
		Provenance:       PiBudgetProvenance,
	}
}

// repairPiModelBudget rewrites an existing fak model entry's contextWindow/maxTokens to the
// derived safe budget, returning true when it changed anything.
//
// Two directions are repaired:
//
//   - OVER-large: an entry advertising the raw served window (131072) is lowered to the
//     resident target, so Pi compacts inside the safe envelope rather than at the ceiling.
//   - UNDER-large: an entry advertising a window that is not this model's at all — the
//     pre-per-model bug, where every catalog entry inherited whichever window the backend
//     last advertised (DeepSeek V4.1 Flash was written with the Qwen 65536) — is raised to
//     this model's own resident target. Without this, a catalog written by the buggy path
//     would keep the wrong number forever, because the old repair only lowered.
//
// An entry that already equals the target is left alone (reported unmodified), so the writer
// stays idempotent on a correctly configured file.
func repairPiModelBudget(mObj map[string]interface{}, budget PiContextBudget) bool {
	changed := false
	if cw, ok := numericField(mObj["contextWindow"]); !ok || cw != budget.ResidentTarget {
		mObj["contextWindow"] = budget.ResidentTarget
		changed = true
	}
	if mt, ok := numericField(mObj["maxTokens"]); !ok || mt <= 0 || mt > budget.OutputReserve {
		mObj["maxTokens"] = budget.OutputReserve
		changed = true
	}
	return changed
}

// numericField reads a JSON number that may have decoded as float64 (the default for
// encoding/json into interface{}) or int, returning the value and whether it was a number.
func numericField(v interface{}) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	default:
		return 0, false
	}
}
