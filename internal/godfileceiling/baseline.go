// Code generated from a MeasureTree->Repin->FormatBaseline pass. DO NOT EDIT by hand.
// Regenerate only to TIGHTEN after a god-file shrinks; never to raise a cap.

package godfileceiling

// Invariant: baseline caps are monotonically non-increasing and pin only first-party files exceeding HardCeiling.

// Baseline pins today's god-files (> HardCeiling lines) at their current LOC. A
// pinned file may only shrink; an unpinned file may not exceed HardCeiling. See doc.go.
var Baseline = map[string]int{
	"cmd/fak/guard.go":                                             1538,
	"cmd/fak/macbench.go":                                          1702,
	"cmd/fak/main.go":                                              1502,
	"cmd/fak/up.go":                                                1554,
	"cmd/modelbench/main.go":                                       1602,
	"internal/adjudicator/decide.go":                               1845,
	"internal/agent/inkernel_planner.go":                           3017,
	"internal/agentbench/agentbench.go":                            1796,
	"internal/compute/amd_gpudirect_kvpaging.go":                   1511,
	"internal/compute/strix/mall_tiling.go":                        1552,
	"internal/compute/vulkan.go":                                   2276,
	"internal/computebuild/vulkan.go":                              1649,
	"internal/debtlane/scan.go":                                    2811,
	"internal/gateway/admission.go":                                1813,
	"internal/gateway/debug.go":                                    1525,
	"internal/gateway/http.go":                                     1564,
	"internal/gateway/mcp.go":                                      1537,
	"internal/gateway/messages.go":                                 1518,
	"internal/gateway/metrics.go":                                  1544,
	"internal/gateway/metrics_render.go":                           1508,
	"internal/issuepolicy/contract.go":                             1647,
	"internal/l3server/metrics/collector.go":                       1519,
	"internal/llamacppinterop/strix_comparator_authority_linux.go": 1652,
	"internal/macbench/mtp_comparison.go":                          1609,
	"internal/metalgemm/sdpa_nax_tile.go":                          1504,
	"internal/model/metal_mtp.go":                                  1731,
	"internal/model/metal_prefill_hybrid.go":                       1523,
	"internal/radixkv/radixkv.go":                                  1860,
	"internal/sessionaudit/sessionaudit.go":                        1510,
}
