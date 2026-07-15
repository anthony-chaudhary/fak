package gateway

import "github.com/anthony-chaudhary/fak/internal/negframe"

// errorAffordances is the closed, deterministic map from known refusal/error
// reason codes to an immediately executable positive next action. Values are
// fak-authored and therefore safe to route through the positive-voice seam.
var errorAffordances = map[string]string{
	"OFF_TRUNK":                "commit on main with fak commit --path <owned-path> -m <message>",
	"OUT_OF_TREE_WRITE":        "write inside the workspace or place scratch data in the OS temp directory",
	"POLICY_BLOCK":             "choose a tool and arguments admitted by the active policy",
	"OVERHEAD_BUDGET_EXCEEDED": "measure against the declared budget, then reduce the overhead or update the witnessed envelope",
	"INVALID_TOOL_ARGUMENTS":   "correct the tool arguments to match the declared schema and retry",
}

// errorAffordance returns a positive next action for a known reason. Unknown
// reason text is returned byte-identically: the fallback never drops or invents
// information. The mapping output is routed through Reframe at this emit seam;
// Reframe is idempotent, so repeated rendering is stable.
func errorAffordance(reason string) string {
	action, ok := errorAffordances[reason]
	if !ok {
		return reason
	}
	return negframe.Reframe(action)
}
