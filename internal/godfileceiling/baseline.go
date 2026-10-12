// Code generated from a MeasureTree->Repin->FormatBaseline pass. DO NOT EDIT by hand.
// Regenerate only to TIGHTEN after a god-file shrinks; never to raise a cap.

package godfileceiling

// Invariant: baseline caps are monotonically non-increasing and pin only first-party files exceeding HardCeiling.

// Baseline pins today's god-files (> HardCeiling lines) at their current LOC. A
// pinned file may only shrink; an unpinned file may not exceed HardCeiling. See doc.go.
var Baseline = map[string]int{
	"internal/adjudicator/decide.go":                               1733,
	"internal/agent/inkernel_planner.go":                           2821,
	"internal/agentbench/agentbench.go":                            1643,
	"internal/compute/amd_gpudirect_kvpaging.go":                   1511,
	"internal/compute/strix/mall_tiling.go":                        1552,
	"internal/compute/vulkan.go":                                   2028,
	"internal/debtlane/scan.go":                                    2197,
	"internal/gateway/admission.go":                                1712,
	"internal/gateway/http.go":                                     1511,
	"internal/issuepolicy/contract.go":                             1584,
	"internal/llamacppinterop/strix_comparator_authority_linux.go": 1652,
	"internal/macbench/mtp_comparison.go":                          1609,
	"internal/metalgemm/sdpa_nax_tile.go":                          1504,
	"internal/radixkv/radixkv.go":                                  1706,
	"internal/sessionaudit/sessionaudit.go":                        1510,
}
