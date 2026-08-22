package toolcallcontrol

import "fmt"

// ResponseDisposition selects whether a completed response can enter the parent
// context or needs the separately implemented bounded branch path.
type ResponseDisposition string

const (
	ResponsePass   ResponseDisposition = "pass"
	ResponseBranch ResponseDisposition = "branch"
)

// ResponseDimension identifies one independently enforced response envelope.
type ResponseDimension string

const (
	ResponseItems           ResponseDimension = "items"
	ResponseBytes           ResponseDimension = "bytes"
	ResponseTokensEstimated ResponseDimension = "tokens_estimated"
)

// ResponseTokenEstimateBasis labels how an estimated token count was derived.
type ResponseTokenEstimateBasis string

const ResponseTokenEstimateBytesDiv4Ceil ResponseTokenEstimateBasis = "bytes_div_4_ceil_v1"

// ResponseLimit sets a ceiling for exactly one response dimension.
type ResponseLimit struct {
	Dimension ResponseDimension `json:"dimension"`
	Maximum   int64             `json:"maximum"`
}

// ResponseActual records content-free post-execution measurements.
type ResponseActual struct {
	Items              int64                      `json:"items"`
	Bytes              int64                      `json:"bytes"`
	TokensEstimated    int64                      `json:"tokens_estimated"`
	TokenEstimateBasis ResponseTokenEstimateBasis `json:"token_estimate_basis"`
}

// ResponseAccounting is the typed pass-or-branch receipt for one completed call.
type ResponseAccounting struct {
	Disposition ResponseDisposition `json:"disposition"`
	Actual      ResponseActual      `json:"actual"`
	Exceeded    []ResponseDimension `json:"exceeded,omitempty"`
}

// AccountResponse measures a raw result and evaluates each declared envelope
// independently. The executor supplies item count because tool result shapes are
// contract-specific; this seam never guesses cardinality from arbitrary JSON.
func AccountResponse(payload []byte, actualItems int64, limits []ResponseLimit) (ResponseAccounting, error) {
	if actualItems < 0 {
		return ResponseAccounting{}, fmt.Errorf("actual response items must be non-negative")
	}
	actualBytes := int64(len(payload))
	receipt := ResponseAccounting{
		Disposition: ResponsePass,
		Actual: ResponseActual{
			Items:              actualItems,
			Bytes:              actualBytes,
			TokensEstimated:    estimateResponseTokens(actualBytes),
			TokenEstimateBasis: ResponseTokenEstimateBytesDiv4Ceil,
		},
	}

	byDimension := make(map[ResponseDimension]int64, len(limits))
	for _, limit := range limits {
		if !validResponseDimension(limit.Dimension) {
			return ResponseAccounting{}, fmt.Errorf("unknown response dimension %q", limit.Dimension)
		}
		if limit.Maximum < 0 {
			return ResponseAccounting{}, fmt.Errorf("response %s maximum must be non-negative", limit.Dimension)
		}
		if _, exists := byDimension[limit.Dimension]; exists {
			return ResponseAccounting{}, fmt.Errorf("duplicate response dimension %q", limit.Dimension)
		}
		byDimension[limit.Dimension] = limit.Maximum
	}

	for _, dimension := range []ResponseDimension{ResponseItems, ResponseBytes, ResponseTokensEstimated} {
		maximum, limited := byDimension[dimension]
		if limited && receipt.Actual.value(dimension) > maximum {
			receipt.Exceeded = append(receipt.Exceeded, dimension)
		}
	}
	if len(receipt.Exceeded) > 0 {
		receipt.Disposition = ResponseBranch
	}
	return receipt, nil
}

func validResponseDimension(dimension ResponseDimension) bool {
	switch dimension {
	case ResponseItems, ResponseBytes, ResponseTokensEstimated:
		return true
	default:
		return false
	}
}

func (actual ResponseActual) value(dimension ResponseDimension) int64 {
	switch dimension {
	case ResponseItems:
		return actual.Items
	case ResponseBytes:
		return actual.Bytes
	case ResponseTokensEstimated:
		return actual.TokensEstimated
	default:
		return 0
	}
}

func estimateResponseTokens(byteCount int64) int64 {
	estimate := byteCount / 4
	if byteCount%4 != 0 {
		estimate++
	}
	return estimate
}
