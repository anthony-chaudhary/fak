// Code generated from a MeasureTree->Repin->FormatBaseline pass. DO NOT EDIT by hand.
// Regenerate only to TIGHTEN after a god-file shrinks; never to raise a cap.

package godfileceiling

// Invariant: baseline caps are monotonically non-increasing and pin only first-party files exceeding HardCeiling.

// Baseline pins today's god-files (> HardCeiling lines) at their current LOC. A
// pinned file may only shrink; an unpinned file may not exceed HardCeiling. See doc.go.
var Baseline = map[string]int{
	"cmd/fak/guard.go":                     1554,
	"internal/adjudicator/decide.go":       1550,
	"internal/agent/inkernel_planner.go":   1509,
	"internal/compute/vulkan.go":           1655,
	"internal/gateway/messages.go":         1504,
	"internal/ifc/ifc.go":                  1581,
	"internal/sessionaudit/sessionaudit.go": 1510,
}
