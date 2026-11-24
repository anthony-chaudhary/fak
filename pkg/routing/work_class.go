package routing

import "strings"

// WorkClass is a caller's routing hint about how delay-tolerant a request is.
// It reorders a resolved chain, or walks a declared model_fallback local route
// first (LocalFirstRoute); it never changes a failure class or the attempt
// budget.
type WorkClass string

const (
	WorkClassNone       WorkClass = ""
	WorkClassBackground WorkClass = "background"
)

const (
	// WorkClassHeader carries the hint on an inbound request.
	WorkClassHeader = "X-Fak-Work-Class"
	// WorkClassEnv is the child-process variable a harness config forwards
	// into WorkClassHeader (opencode.json: "{env:FAK_WORK_CLASS}").
	WorkClassEnv = "FAK_WORK_CLASS"
	// LocalFirstAlias is the model-id form of the background hint: the bare id
	// or a "local-first/<model>" qualified id.
	LocalFirstAlias = "local-first"
	// LegacyHaloFirstAlias is the earlier spelling of LocalFirstAlias. Deployed
	// harness configs and routing tables still send it, so it stays accepted.
	LegacyHaloFirstAlias = "halo-first"
	// RuleWorkClassBackground prefixes a decision's matched rule when the
	// background hint reordered its chain, e.g. "workclass:background+alias:x".
	RuleWorkClassBackground = "workclass:background+"
	// RuleLocalFirst marks the local-first hop's rule, e.g.
	// "workclass:background+local_first:<client model>->model:<local route>".
	RuleLocalFirst = "local_first:"
	// LegacyRuleHaloFirst is the marker journals written before RuleLocalFirst
	// carry; readers accept both through IsLocalFirstRule.
	LegacyRuleHaloFirst = "halo_first:"
)

// ParseWorkClass maps a header value to a closed WorkClass; anything
// unrecognised is WorkClassNone.
func ParseWorkClass(v string) WorkClass {
	if strings.EqualFold(strings.TrimSpace(v), string(WorkClassBackground)) {
		return WorkClassBackground
	}
	return WorkClassNone
}

// IsLocalFirstAlias reports whether model is the local-first hint in either
// spelling, bare or qualified with a "/<model>" suffix.
func IsLocalFirstAlias(model string) bool {
	for _, alias := range [...]string{LocalFirstAlias, LegacyHaloFirstAlias} {
		if model == alias || strings.HasPrefix(model, alias+"/") {
			return true
		}
	}
	return false
}

// IsLocalFirstRule reports whether a decision rule records a local-first hop,
// under the current marker or the legacy one.
func IsLocalFirstRule(rule string) bool {
	rule = strings.TrimPrefix(rule, RuleWorkClassBackground)
	return strings.HasPrefix(rule, RuleLocalFirst) || strings.HasPrefix(rule, LegacyRuleHaloFirst)
}

// WorkClassFor resolves the hint from the header value, falling back to the
// local-first model alias.
func WorkClassFor(header, model string) WorkClass {
	if c := ParseWorkClass(header); c != WorkClassNone {
		return c
	}
	if IsLocalFirstAlias(model) {
		return WorkClassBackground
	}
	return WorkClassNone
}

// WorkClassMember is one resolved chain entry as the work-class order sees it.
// Local marks a self-hosted node, as opposed to a hosted provider rung.
type WorkClassMember struct {
	Name    string
	Local   bool
	Healthy bool
}

// OrderForWorkClass returns the chain order a work class asks for and whether
// the hint took effect. Background puts healthy local nodes first, then every
// hosted rung, then unhealthy local nodes, each group in declared order. It
// takes effect only when at least one healthy local node is present; otherwise
// the declared order is returned unchanged.
func OrderForWorkClass(c WorkClass, members []WorkClassMember) ([]string, bool) {
	declared := make([]string, len(members))
	for i, m := range members {
		declared[i] = m.Name
	}
	if c != WorkClassBackground {
		return declared, false
	}
	var local, hosted, sick []string
	for _, m := range members {
		switch {
		case m.Local && m.Healthy:
			local = append(local, m.Name)
		case m.Local:
			sick = append(sick, m.Name)
		default:
			hosted = append(hosted, m.Name)
		}
	}
	if len(local) == 0 {
		return declared, false
	}
	out := make([]string, 0, len(members))
	out = append(out, local...)
	out = append(out, hosted...)
	return append(out, sick...), true
}

// WorkClassRule names the hint on a decision rule. It is idempotent.
func WorkClassRule(c WorkClass, rule string) string {
	if c != WorkClassBackground || strings.HasPrefix(rule, RuleWorkClassBackground) {
		return rule
	}
	return RuleWorkClassBackground + rule
}

// WorkClassLocalMaxInFlight is how many concurrent attempts a local node
// carries before background work overflows past it to hosted. A slow local
// decoder gains nothing from a deeper queue: it only delays a worker a hosted
// rung would serve now.
const WorkClassLocalMaxInFlight = 2

// WorkClassLocalHasCapacity reports whether a local node at inFlight
// concurrent attempts may still take background work first.
func WorkClassLocalHasCapacity(inFlight int64) bool {
	return inFlight < WorkClassLocalMaxInFlight
}

// WorkClassRoute is one model_fallback route as the local-first hop sees it.
type WorkClassRoute struct {
	Model   string
	Members []WorkClassMember
}

// LocalFirstRoute picks the model_fallback route a background request walks
// before its own route: only when its own chain offers no healthy local node,
// and then the first declared fallback route that does. "" means walk as
// declared.
func LocalFirstRoute(c WorkClass, primary []WorkClassMember, fallbacks []WorkClassRoute) string {
	if c != WorkClassBackground || len(LocalPrefix(primary)) > 0 {
		return ""
	}
	for _, r := range fallbacks {
		if len(LocalPrefix(r.Members)) > 0 {
			return r.Model
		}
	}
	return ""
}

// LocalPrefix is the healthy local members in declared order: the only rungs
// a local-first hop dials before it overflows to the caller's own route.
func LocalPrefix(members []WorkClassMember) []string {
	var out []string
	for _, m := range members {
		if m.Local && m.Healthy {
			out = append(out, m.Name)
		}
	}
	return out
}

// WithBackgroundWorkClass marks a worker child as delay-tolerant so its
// harness forwards WorkClassHeader. An explicit value already in env wins,
// including an empty one.
func WithBackgroundWorkClass(env []string) []string {
	prefix := WorkClassEnv + "="
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			return env
		}
	}
	return append(env, prefix+string(WorkClassBackground))
}
