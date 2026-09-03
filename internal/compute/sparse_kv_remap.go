package compute

import (
	"fmt"
)

// SparseKVRemapReceipt records execution metrics of the sparse logical-to-physical index translation.
type SparseKVRemapReceipt struct {
	TotalEntries            int `json:"total_entries"`
	MappedEntries           int `json:"mapped_entries"`
	PreservedInvalidEntries int `json:"preserved_invalid_entries"`
	UniquePhysicalPages     int `json:"unique_physical_pages"`
	PageSize                int `json:"page_size"`
}

// RemapSparseLogicalToPhysicalKV translates sparse logical token indices into physical paged rows
// without materializing full dense index tensors, while strictly preserving -1 invalid/padding entries.
func RemapSparseLogicalToPhysicalKV(
	logicalIndices []int32,
	pageTable []int32,
	reqTableOffsets []int,
	reqTokensCount []int,
	pageSize int,
	outPhysical []int32,
) (SparseKVRemapReceipt, error) {
	var receipt SparseKVRemapReceipt
	if pageSize <= 0 {
		return receipt, fmt.Errorf("pageSize must be positive, got %d", pageSize)
	}
	if len(reqTableOffsets) != len(reqTokensCount) {
		return receipt, fmt.Errorf("mismatched request counts: offsets=%d, counts=%d", len(reqTableOffsets), len(reqTokensCount))
	}

	totalTokens := 0
	for _, cnt := range reqTokensCount {
		if cnt < 0 {
			return receipt, fmt.Errorf("negative token count: %d", cnt)
		}
		totalTokens += cnt
	}

	if len(logicalIndices) != totalTokens {
		return receipt, fmt.Errorf("logicalIndices length %d != sum of token counts %d", len(logicalIndices), totalTokens)
	}
	if len(outPhysical) != totalTokens {
		return receipt, fmt.Errorf("outPhysical length %d != sum of token counts %d", len(outPhysical), totalTokens)
	}

	uniquePages := make(map[int32]bool)
	var mapped, preserved int

	tokenCursor := 0
	for reqIdx, numTokens := range reqTokensCount {
		tableOffset := reqTableOffsets[reqIdx]
		if tableOffset < 0 || tableOffset > len(pageTable) {
			return receipt, fmt.Errorf("request %d table offset %d out of bounds [0, %d]", reqIdx, tableOffset, len(pageTable))
		}

		maxPages := len(pageTable) - tableOffset
		if reqIdx+1 < len(reqTableOffsets) && reqTableOffsets[reqIdx+1] >= tableOffset {
			maxPages = reqTableOffsets[reqIdx+1] - tableOffset
		}

		for i := 0; i < numTokens; i++ {
			logicalIdx := logicalIndices[tokenCursor+i]

			if logicalIdx == -1 {
				outPhysical[tokenCursor+i] = -1
				preserved++
				continue
			}

			if logicalIdx < -1 {
				return receipt, fmt.Errorf("invalid negative logical index: %d", logicalIdx)
			}

			logicalPage := int(logicalIdx) / pageSize
			offsetInPage := int(logicalIdx) % pageSize

			if logicalPage >= maxPages {
				return receipt, fmt.Errorf("logical page %d exceeds request page table capacity %d", logicalPage, maxPages)
			}

			physPage := pageTable[tableOffset+logicalPage]
			if physPage < 0 {
				return receipt, fmt.Errorf("unmapped physical page %d for logical page %d", physPage, logicalPage)
			}

			physIndex := physPage*int32(pageSize) + int32(offsetInPage)
			outPhysical[tokenCursor+i] = physIndex
			uniquePages[physPage] = true
			mapped++
		}
		tokenCursor += numTokens
	}

	receipt = SparseKVRemapReceipt{
		TotalEntries:            totalTokens,
		MappedEntries:           mapped,
		PreservedInvalidEntries: preserved,
		UniquePhysicalPages:     len(uniquePages),
		PageSize:                pageSize,
	}

	return receipt, nil
}
