// Code generated from a MeasureTree->Repin->FormatBaseline pass. DO NOT EDIT by hand.
// Regenerate only to TIGHTEN after a god-file shrinks; never to raise a cap.

package godfileceiling

// Baseline pins today's god-files (> HardCeiling lines) at their current LOC. A
// pinned file may only shrink; an unpinned file may not exceed HardCeiling. See doc.go.
var Baseline = map[string]int{
	"cmd/fak/cachevalue_status.go":            3014,
	"cmd/fak/dispatch_tick.go":                1730,
	"cmd/fak/dispatch_tick_test.go":           2424,
	"cmd/fak/guard_test.go":                   2234,
	"cmd/fak/loop.go":                         1544,
	"cmd/fak/release_ship.go":                 1708,
	"cmd/fak/release_ship_test.go":            2186,
	"cmd/fak/tui_test.go":                     2055,
	"internal/agent/adapters_test.go":         1928,
	"internal/agent/chat.go":                  1664,
	"internal/architest/architest_test.go":    2549,
	"internal/dispatchtick/router.go":         1768,
	"internal/fleetpane/fleetpane.go":         2091,
	"internal/gateway/gateway.go":             3135,
	"internal/gateway/gateway_test.go":        2943,
	"internal/gateway/http.go":                1819,
	"internal/gateway/messages.go":            1739,
	"internal/gateway/metrics.go":             3354,
	"internal/model/oracle_test.go":           1634,
	"internal/operatorbrief/operatorbrief.go": 1576,
	"internal/sessionaudit/sessionaudit.go":   1737,
}
