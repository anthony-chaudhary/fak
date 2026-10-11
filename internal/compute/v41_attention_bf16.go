package compute

import "fmt"

// V41AttentionBF16Backend is the ordinary finite publication at the two real
// attention-input norm edges. It shares the checked integer module, not the
// witness-only final-norm or query-prefix method set. A result is fresh,
// disjoint F32 [N] storage whose bits are finite widened BF16.
//
// Supports describes the loaded operation and the selected device's count
// domain, never transient batch or submission health. A selected failure has
// no decline outcome. The synchronous operation does not take over an active
// generic batch or restore transaction.
type V41AttentionBF16Backend interface {
	SupportsV41AttentionBF16(count int) bool
	V41AttentionBF16(input Tensor, site uint32) (Tensor, error)
}

func validateV41AttentionBF16(input Tensor, site uint32) (int, error) {
	if site != V41BF16CheckedAttentionNormInput && site != V41BF16CheckedAttentionNormOutput {
		return 0, fmt.Errorf("compute: attention BF16 requires an attention-norm input or output site")
	}
	return validateV41BF16Boundary(input)
}

// The closed 3/4 policy is separate from the unchanged final-norm 1/2 and
// query-prefix 3..6 policies. A matching envelope cannot authorize a new site.
func decodeV41AttentionBF16Status(raw []byte, count int, site uint32) error {
	if site != V41BF16CheckedAttentionNormInput && site != V41BF16CheckedAttentionNormOutput {
		return &V41BF16CheckedProtocolError{Message: "invalid attention-norm publication site"}
	}
	return decodeV41BF16CheckedEnvelope(raw, count, site)
}
