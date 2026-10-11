package compute

import "fmt"

// V41BF16PublishBackend is the ordinary finite BF16 publication for every V4.1
// full-profile dtype boundary outside the attention-input norm. It shares the
// ordinary checked integer module with V41AttentionBF16Backend and adds no
// SPIR-V. A result is fresh, disjoint F32 [N] storage whose bits are finite
// widened BF16; a nonfinite input or a rounded overflow is an arithmetic error
// attributed to the lowest index, and its output is withheld.
//
// Supports describes the loaded operation and the selected device's count
// domain, never transient batch or submission health. A selected failure has
// no decline outcome. The synchronous operation does not take over an active
// generic batch or restore transaction.
type V41BF16PublishBackend interface {
	SupportsV41BF16Publish(count int) bool
	V41BF16Publish(input Tensor, site uint32) (Tensor, error)
}

// The ordinary publication sites. Final-norm 1/2 and query-norm 5/6 keep the
// identities the checked witness profiles already use; attention 3/4 stays on
// its own entry. New boundaries take IDs from 7 upward.
const (
	V41BF16PublishFinalNormInput                 = V41BF16CheckedFinalNormInput
	V41BF16PublishFinalNormOutput                = V41BF16CheckedFinalNormOutput
	V41BF16PublishQueryNormInput                 = V41BF16CheckedQueryNormInput
	V41BF16PublishQueryNormOutput                = V41BF16CheckedQueryNormOutput
	V41BF16PublishKVNormInput             uint32 = 7
	V41BF16PublishKVNormOutput            uint32 = 8
	V41BF16PublishFFNNormInput            uint32 = 9
	V41BF16PublishFFNNormOutput           uint32 = 10
	V41BF16PublishTailRoPEInput           uint32 = 11
	V41BF16PublishTailRoPEOutput          uint32 = 12
	V41BF16PublishInverseOutputInput      uint32 = 13
	V41BF16PublishInverseOutputOutput     uint32 = 14
	V41BF16PublishGroupedOutputInput      uint32 = 15
	V41BF16PublishGroupedOutputProjection uint32 = 16
	V41BF16PublishSharedExpertInput       uint32 = 17
	V41BF16PublishSharedExpertProjection  uint32 = 18
	V41BF16PublishSharedExpertDownInput   uint32 = 19
	V41BF16PublishRoutedGateUp            uint32 = 20
	V41BF16PublishRoutedDown              uint32 = 21
	V41BF16PublishIndexerProjection       uint32 = 22
	V41BF16PublishIndexerKeyNorm          uint32 = 23
	V41BF16PublishIndexerRotary           uint32 = 24
	V41BF16PublishIndexerScaledWeight     uint32 = 25
	V41BF16PublishCompressorNormInput     uint32 = 26
	V41BF16PublishCompressorNormOutput    uint32 = 27
)

var v41BF16PublishSiteNames = map[uint32]string{
	V41BF16PublishKVNormInput:             "kv-norm input",
	V41BF16PublishKVNormOutput:            "kv-norm output",
	V41BF16PublishFFNNormInput:            "ffn-norm input",
	V41BF16PublishFFNNormOutput:           "ffn-norm output",
	V41BF16PublishTailRoPEInput:           "tail-RoPE input",
	V41BF16PublishTailRoPEOutput:          "tail-RoPE output",
	V41BF16PublishInverseOutputInput:      "inverse-output input",
	V41BF16PublishInverseOutputOutput:     "inverse-output output",
	V41BF16PublishGroupedOutputInput:      "grouped-output input",
	V41BF16PublishGroupedOutputProjection: "grouped-output projection",
	V41BF16PublishSharedExpertInput:       "shared-expert input",
	V41BF16PublishSharedExpertProjection:  "shared-expert projection",
	V41BF16PublishSharedExpertDownInput:   "shared-expert down input",
	V41BF16PublishRoutedGateUp:            "routed-expert gate/up",
	V41BF16PublishRoutedDown:              "routed-expert down",
	V41BF16PublishIndexerProjection:       "indexer projection",
	V41BF16PublishIndexerKeyNorm:          "indexer key-norm output",
	V41BF16PublishIndexerRotary:           "indexer rotary",
	V41BF16PublishIndexerScaledWeight:     "indexer scaled weight",
	V41BF16PublishCompressorNormInput:     "compressor-norm input",
	V41BF16PublishCompressorNormOutput:    "compressor-norm output",
}

// V41BF16PublishSite is the closed ordinary site policy. The native entry
// enforces the same set independently.
func V41BF16PublishSite(site uint32) bool {
	switch site {
	case V41BF16PublishFinalNormInput, V41BF16PublishFinalNormOutput,
		V41BF16PublishQueryNormInput, V41BF16PublishQueryNormOutput:
		return true
	}
	_, ok := v41BF16PublishSiteNames[site]
	return ok
}

func validateV41BF16Publish(input Tensor, site uint32) (int, error) {
	if !V41BF16PublishSite(site) {
		return 0, fmt.Errorf("compute: ordinary BF16 publication requires a known publication site")
	}
	return validateV41BF16Boundary(input)
}

// A matching envelope cannot authorize a site outside the ordinary policy.
func decodeV41BF16PublishStatus(raw []byte, count int, site uint32) error {
	if !V41BF16PublishSite(site) {
		return &V41BF16CheckedProtocolError{Message: "invalid ordinary publication site"}
	}
	return decodeV41BF16CheckedEnvelope(raw, count, site)
}
