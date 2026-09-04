// Code generated from a MeasureTree->Repin->FormatBaseline pass. DO NOT EDIT by hand.
// Regenerate only to TIGHTEN after a god-file shrinks; never to raise a cap.

package godfileceiling

// Baseline pins today's god-files (> HardCeiling lines) at their current LOC. A
// pinned file may only shrink; an unpinned file may not exceed HardCeiling. See doc.go.
// _test.go files are NOT pinned here — MeasureTree excludes them (the tests KPI grades
// them, and they churn per new leaf), matching internal/hooks/gate_godfile.go.
var Baseline = map[string]int{
	"cmd/fak/cachevalue_status.go":            3014,
	"cmd/fak/dispatch_tick.go":                1730,
	"cmd/fak/loop.go":                         1544,
	"cmd/fak/release_ship.go":                 1708,
	"internal/agent/chat.go":                  1664,
	"internal/compute/cuda.go":                1562,
	"internal/dispatchtick/router.go":         1768,
	"internal/fleetpane/fleetpane.go":         2091,
	"internal/gateway/gateway.go":             3135,
	"internal/gateway/http.go":                1819,
	"internal/gateway/messages.go":            1739,
	"internal/gateway/metrics.go":             3354,
	"internal/operatorbrief/operatorbrief.go": 1576,
	"internal/sessionaudit/sessionaudit.go":   1737,
}
