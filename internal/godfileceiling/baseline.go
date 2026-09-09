// Code generated from a MeasureTree->Repin->FormatBaseline pass. DO NOT EDIT by hand.
// Regenerate only to TIGHTEN after a god-file shrinks; never to raise a cap.

package godfileceiling

// Invariant: baseline caps are monotonically non-increasing and pin only first-party files exceeding HardCeiling.

// Baseline pins today's god-files (> HardCeiling lines) at their current LOC. A
// pinned file may only shrink; an unpinned file may not exceed HardCeiling. See doc.go.
var Baseline = map[string]int{
	"cmd/fak/guard.go":                                            1554,
	"cmd/modelbench/main.go":                                      1585,
	"internal/adjudicator/decide.go":                              1840,
	"internal/agent/inkernel_planner.go":                          2034,
	"internal/compute/strix/mall_tiling.go":                       1552,
	"internal/compute/vulkan.go":                                  2054,
	"internal/computebuild/vulkan.go":                             1645,
	"internal/ctxmmu/cow.go":                                      1882,
	"internal/debtlane/scan.go":                                   2709,
	"internal/gateway/mcp.go":                                     1510,
	"internal/gateway/messages.go":                                1504,
	"internal/ifc/ifc.go":                                         1581,
	"internal/issuepolicy/contract.go":                            1647,
	"internal/l3server/metrics/collector.go":                      1519,
	"internal/llamacppinterop/strix_comparator_authority_linux.go": 1652,
	"internal/macbench/mtp_comparison.go":                         1600,
	"internal/metalgemm/sdpa_nax_tile.go":                         1504,
	"internal/model/metal_mtp.go":                                 1596,
	"internal/radixkv/radixkv.go":                                 1750,
	"internal/sessionaudit/sessionaudit.go":                       1510,
}
