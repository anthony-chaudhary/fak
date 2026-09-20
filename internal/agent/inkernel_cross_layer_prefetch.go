package agent

import (
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_cross_layer_prefetch.go — the served seam for cross-LAYER gate prediction (R3.5 of the
// activated-expert offload ladder, #5614 lineage, epic #5606; operator surface #1297/#1401).
//
// What was missing. internal/model can now predict layer L+1's routed set while layer L computes and
// stage the prediction into the routed-expert ring as HINTS (internal/model/expert_readahead.go,
// Session.CrossLayerGatePrefetch), and the per-session precision/recall ledger rides out through the
// expert-residency report. But the knob was only a Session field: nothing a serve builds ever set it,
// so the landed predictor was unreachable from an operator and unobservable through the report. This
// file is the reachable half — one planner field, one per-session install, one setter.
//
// Why the install is PER SESSION and not process-wide. The feature is deliberately a Session field
// (a sibling of ExpertPrefetch), precisely so enabling it for one serve does not turn it on
// fleet-wide. The planner builds a session per request on the device path, so the install must be a
// field write on EVERY session it builds — exactly the shape applyExpertSpill already has.
//
// DEFAULT OFF, BYTE-FOR-BYTE. The planner field's zero value is false, applyCrossLayerGatePrefetch is
// then a no-op, and the session keeps the model's own zero value — the pre-rung forward. SetCrossLayerGatePrefetch(false)
// is the same no-op.

// CrossLayerGatePrefetchEnv is the environment knob that turns per-session next-layer gate prefetch
// on for an already-constructed planner. It is the seam NewInKernelPlanner reads, so a serve flag
// and a shell export resolve to the same planner setting.
const CrossLayerGatePrefetchEnv = "FAK_CROSS_LAYER_GATE_PREFETCH"

// SetCrossLayerGatePrefetch enables (#1297/#5614) next-layer gate prediction on every session this
// planner builds: while layer L computes, layer L+1's router gate is applied to L's hidden state and
// the predicted top-k is staged into the routed-expert ring as HINTS (never a demand, so a mispredict
// cannot change logits). It is a PER-SESSION knob: an operator enables it for this serve without
// turning the feature on fleet-wide. Off (the default, zero value) leaves every session
// byte-for-byte unchanged. Inert without an expert ring.
func (p *InKernelPlanner) SetCrossLayerGatePrefetch(on bool) {
	if p == nil {
		return
	}
	p.crossLayerGatePrefetch = on
}

// CrossLayerGatePrefetchEnabled reports whether this planner installs the per-session knob on the
// sessions it builds. It is exported so a serve can REPORT what it admitted, and so a call site can
// tell a default planner from an opted-in one without reaching into the unexported field.
func (p *InKernelPlanner) CrossLayerGatePrefetchEnabled() bool {
	return p != nil && p.crossLayerGatePrefetch
}

// ParseCrossLayerGatePrefetch parses the boolean spelling an operator may pass on the flag or the
// env. It accepts the go-idiomatic set (true/false/1/0/t/f/yes/no/on/off, case- and
// whitespace-insensitive) via strconv.ParseBool plus on/off, and REFUSES anything else so a
// misspelled value never silently reads as off — the same posture ParseExpertSpillGrade takes.
//
// An empty string means "not set": it returns (false, false, nil), which the callers treat as the
// byte-for-byte default rather than an explicit off.
func ParseCrossLayerGatePrefetch(s string) (on bool, set bool, err error) {
	v := strings.ToLower(strings.TrimSpace(s))
	switch v {
	case "":
		return false, false, nil
	case "on":
		return true, true, nil
	case "off":
		return false, true, nil
	}
	on, err = strconv.ParseBool(v)
	if err != nil {
		return false, false, &CrossLayerGatePrefetchParseError{Value: s}
	}
	return on, true, nil
}

// CrossLayerGatePrefetchParseError is the typed refusal ParseCrossLayerGatePrefetch returns, so a
// caller can name the knob in its own diagnostic rather than string-matching a generic parse error.
type CrossLayerGatePrefetchParseError struct{ Value string }

func (e *CrossLayerGatePrefetchParseError) Error() string {
	return "cross-layer-gate-prefetch: " + strconv.Quote(e.Value) + " is not a boolean; want true/false, 1/0, yes/no, or on/off"
}

// setCrossLayerGatePrefetchFromEnv applies CrossLayerGatePrefetchEnv, if the operator set it. Unset —
// every serve that has not opted in — it is a no-op and every session keeps the model's zero value.
//
// A set-but-unparseable value is LOGGED and ignored, not fatal: this is the opportunistic env door
// on an already-constructed planner, and taking a serve down at construction time for a mistyped
// optional knob trades a lost accelerator for no serve at all. The flag is the STRICT door — a
// caller that parses an operator flag gets the error back and can refuse the launch outright.
func (p *InKernelPlanner) setCrossLayerGatePrefetchFromEnv() {
	on, set, err := ParseCrossLayerGatePrefetch(os.Getenv(CrossLayerGatePrefetchEnv))
	if err != nil {
		log.Printf("fak: %s=%q REFUSED: %v — serving with next-layer gate prefetch OFF", CrossLayerGatePrefetchEnv, os.Getenv(CrossLayerGatePrefetchEnv), err)
		return
	}
	if !set {
		return
	}
	p.SetCrossLayerGatePrefetch(on)
	if on {
		log.Printf("fak: %s=%q -> next-layer gate prefetch ON for every session this planner builds (per-session; requires an expert ring)", CrossLayerGatePrefetchEnv, os.Getenv(CrossLayerGatePrefetchEnv))
	}
}

// applyCrossLayerGatePrefetch installs the per-session knob on a session the planner just built.
// With the field off (the default, and every planner whose operator never opted in) it is a no-op,
// so the session keeps the model's own zero value — byte-for-byte the pre-rung forward.
func (p *InKernelPlanner) applyCrossLayerGatePrefetch(s *model.Session) {
	if p == nil || s == nil || !p.crossLayerGatePrefetch {
		return
	}
	s.CrossLayerGatePrefetch = true
}
