package compute

import (
	"fmt"
)

// QuantGranularity defines the scale resolution for quantized attention operands.
type QuantGranularity string

const (
	QuantGranularityTensor QuantGranularity = "tensor"
	QuantGranularityHead   QuantGranularity = "head"
	QuantGranularityBlock  QuantGranularity = "block"
)

// PagedQuantOperandContract specifies the binding contract for quantized paged attention.
type PagedQuantOperandContract struct {
	Precision   string           `json:"precision"`   // "fp8", "int8", "fp16"
	Granularity QuantGranularity `json:"granularity"` // "tensor", "head", "block"
	NumHeads    int              `json:"num_heads"`
	HeadDim     int              `json:"head_dim"`
	PageSize    int              `json:"page_size"`
	GroupSize   int              `json:"group_size"`
}

// PagedQuantOperandReceipt records validation, scale count, and memory metrics.
type PagedQuantOperandReceipt struct {
	Contract           PagedQuantOperandContract `json:"contract"`
	TotalTokens        int                       `json:"total_tokens"`
	TotalPages         int                       `json:"total_pages"`
	ExpectedScales     int                       `json:"expected_scales"`
	MemoryBytesTotal   int                       `json:"memory_bytes_total"`
	DequantCosineFloor float64                   `json:"dequant_cosine_floor"`
}

// ExpectedScaleCount derives the scale element count required for the contract.
func (c PagedQuantOperandContract) ExpectedScaleCount(totalPages int) (int, error) {
	if c.NumHeads <= 0 || c.HeadDim <= 0 || c.PageSize <= 0 {
		return 0, fmt.Errorf("invalid geometry: heads=%d, headDim=%d, pageSize=%d", c.NumHeads, c.HeadDim, c.PageSize)
	}

	switch c.Granularity {
	case QuantGranularityTensor:
		return 1, nil
	case QuantGranularityHead:
		return c.NumHeads, nil
	case QuantGranularityBlock:
		if c.GroupSize <= 0 || c.PageSize%c.GroupSize != 0 {
			return 0, fmt.Errorf("pageSize %d must be divisible by groupSize %d", c.PageSize, c.GroupSize)
		}
		groupsPerPage := c.PageSize / c.GroupSize
		return totalPages * groupsPerPage * c.NumHeads, nil
	default:
		return 0, fmt.Errorf("unsupported quantization granularity: %q", c.Granularity)
	}
}

// ValidatePagedQuantOperandContract validates compatibility and computes storage bounds.
func ValidatePagedQuantOperandContract(
	contract PagedQuantOperandContract,
	totalPages int,
	totalTokens int,
	keyScalesLen int,
) (PagedQuantOperandReceipt, error) {
	var receipt PagedQuantOperandReceipt
	if contract.Precision != "int8" && contract.Precision != "fp8" && contract.Precision != "fp16" {
		return receipt, fmt.Errorf("unsupported precision: %q (want int8, fp8, or fp16)", contract.Precision)
	}
	if totalPages <= 0 || totalTokens <= 0 {
		return receipt, fmt.Errorf("totalPages and totalTokens must be positive: pages=%d, tokens=%d", totalPages, totalTokens)
	}
	if totalTokens > totalPages*contract.PageSize {
		return receipt, fmt.Errorf("totalTokens %d exceeds capacity of %d pages (%d)", totalTokens, totalPages, totalPages*contract.PageSize)
	}

	expectedScales, err := contract.ExpectedScaleCount(totalPages)
	if err != nil {
		return receipt, err
	}
	if keyScalesLen != expectedScales {
		return receipt, fmt.Errorf("keyScales length %d != expected %d for granularity %s", keyScalesLen, expectedScales, contract.Granularity)
	}

	bytesPerTokenElem := 1
	if contract.Precision == "fp16" {
		bytesPerTokenElem = 2
	}
	dataBytes := totalPages * contract.PageSize * contract.NumHeads * contract.HeadDim * bytesPerTokenElem
	scaleBytes := expectedScales * 4
	totalBytes := dataBytes + scaleBytes

	var cosineFloor float64
	switch contract.Granularity {
	case QuantGranularityBlock:
		cosineFloor = 0.999
	case QuantGranularityHead:
		cosineFloor = 0.995
	case QuantGranularityTensor:
		cosineFloor = 0.990
	}

	receipt = PagedQuantOperandReceipt{
		Contract:           contract,
		TotalTokens:        totalTokens,
		TotalPages:         totalPages,
		ExpectedScales:     expectedScales,
		MemoryBytesTotal:   totalBytes,
		DequantCosineFloor: cosineFloor,
	}

	return receipt, nil
}

// ResolveTokenScale computes the scale factor for a given token position and head.
func (c PagedQuantOperandContract) ResolveTokenScale(
	scales []float32,
	physicalPage int32,
	offsetInPage int,
	head int,
) float32 {
	switch c.Granularity {
	case QuantGranularityTensor:
		return scales[0]
	case QuantGranularityHead:
		return scales[head]
	case QuantGranularityBlock:
		groupsPerPage := c.PageSize / c.GroupSize
		groupInPage := offsetInPage / c.GroupSize
		scaleIdx := int(physicalPage)*groupsPerPage*c.NumHeads + groupInPage*c.NumHeads + head
		return scales[scaleIdx]
	default:
		return 1.0
	}
}

// DequantizePagedKVToken reconstructs a single head vector from paged quantized storage.
func DequantizePagedKVToken(
	keyData []int8,
	scales []float32,
	contract PagedQuantOperandContract,
	pageTable []int32,
	logicalPos int,
	head int,
) ([]float32, error) {
	if logicalPos < 0 {
		return nil, fmt.Errorf("negative logical position %d", logicalPos)
	}
	if head < 0 || head >= contract.NumHeads {
		return nil, fmt.Errorf("head %d out of bounds [0, %d)", head, contract.NumHeads)
	}

	lPage := logicalPos / contract.PageSize
	offsetInPage := logicalPos % contract.PageSize

	if lPage >= len(pageTable) {
		return nil, fmt.Errorf("logical page %d exceeds page table length %d", lPage, len(pageTable))
	}
	pPage := pageTable[lPage]
	if pPage < 0 {
		return nil, fmt.Errorf("unmapped physical page %d", pPage)
	}

	scale := contract.ResolveTokenScale(scales, pPage, offsetInPage, head)

	// Offset into keyData: physicalToken * (numHeads * headDim) + head * headDim
	hd := contract.HeadDim
	stride := contract.NumHeads * hd
	tokenOffset := (int(pPage)*contract.PageSize + offsetInPage) * stride
	headOffset := tokenOffset + head*hd

	if headOffset+hd > len(keyData) {
		return nil, fmt.Errorf("keyData read out of bounds: offset=%d len=%d", headOffset+hd, len(keyData))
	}

	out := make([]float32, hd)
	for d := 0; d < hd; d++ {
		out[d] = float32(keyData[headOffset+d]) * scale
	}

	return out, nil
}
