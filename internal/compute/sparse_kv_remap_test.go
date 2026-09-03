package compute

import (
	"reflect"
	"testing"
)

func referenceSparseLogicalToPhysical(
	logicalIndices []int32,
	pageTable []int32,
	reqTableOffsets []int,
	reqTokensCount []int,
	pageSize int,
) []int32 {
	out := make([]int32, len(logicalIndices))
	cursor := 0
	for reqIdx, count := range reqTokensCount {
		offset := reqTableOffsets[reqIdx]
		for i := 0; i < count; i++ {
			logIdx := logicalIndices[cursor+i]
			if logIdx == -1 {
				out[cursor+i] = -1
				continue
			}
			lPage := int(logIdx) / pageSize
			inPage := int(logIdx) % pageSize
			pPage := pageTable[offset+lPage]
			out[cursor+i] = pPage*int32(pageSize) + int32(inPage)
		}
		cursor += count
	}
	return out
}

func TestSparseKVIndexRemapWitness(t *testing.T) {
	// First witness requirements (#9929):
	// 1. Ragged offsets across requests
	// 2. Shuffled LUTs (arbitrary physical page order)
	// 3. Invalid slots (-1) strictly preserved
	// 4. Duplicate indices mapped correctly
	// 5. OOB canaries left untampered
	// 6. Match reference oracle

	pageSize := 16

	pageTable := []int32{
		50, 12, 99, 3, // req 0
		8, 14, // req 1
		77, 21, 5, // req 2
	}
	reqTableOffsets := []int{0, 4, 6}

	logicalIndices := []int32{
		0, 17, -1, 35, 17,
		3, -1, 20,
		-1, 33, 40, -1,
	}
	reqTokensCount := []int{5, 3, 4}

	totalTokens := len(logicalIndices)

	canaryFront := int32(0xCAFE)
	canaryBack := int32(0xBEEF)
	buffer := make([]int32, totalTokens+2)
	buffer[0] = canaryFront
	buffer[len(buffer)-1] = canaryBack
	outPhysical := buffer[1 : len(buffer)-1]

	receipt, err := RemapSparseLogicalToPhysicalKV(
		logicalIndices,
		pageTable,
		reqTableOffsets,
		reqTokensCount,
		pageSize,
		outPhysical,
	)
	if err != nil {
		t.Fatalf("RemapSparseLogicalToPhysicalKV failed: %v", err)
	}

	if buffer[0] != canaryFront || buffer[len(buffer)-1] != canaryBack {
		t.Fatalf("OOB canary corrupted: front=%x (want %x), back=%x (want %x)", buffer[0], canaryFront, buffer[len(buffer)-1], canaryBack)
	}

	want := referenceSparseLogicalToPhysical(logicalIndices, pageTable, reqTableOffsets, reqTokensCount, pageSize)
	if !reflect.DeepEqual(outPhysical, want) {
		t.Fatalf("physical remap mismatch:\ngot =%v\nwant=%v", outPhysical, want)
	}

	if receipt.PreservedInvalidEntries != 4 {
		t.Fatalf("expected 4 preserved invalid entries, got %d", receipt.PreservedInvalidEntries)
	}
	if outPhysical[2] != -1 || outPhysical[6] != -1 || outPhysical[8] != -1 || outPhysical[11] != -1 {
		t.Fatalf("invalid slots not -1: %v", outPhysical)
	}

	if outPhysical[1] != 193 || outPhysical[4] != 193 {
		t.Fatalf("duplicate index 17 did not map to 193: got %d and %d", outPhysical[1], outPhysical[4])
	}
	if outPhysical[0] != 800 {
		t.Fatalf("logical 0 mapped to %d, want 800", outPhysical[0])
	}
	if outPhysical[3] != 1587 {
		t.Fatalf("logical 35 mapped to %d, want 1587", outPhysical[3])
	}

	if receipt.TotalEntries != 12 {
		t.Fatalf("expected 12 total entries, got %d", receipt.TotalEntries)
	}
	if receipt.MappedEntries != 8 {
		t.Fatalf("expected 8 mapped entries, got %d", receipt.MappedEntries)
	}
}

func TestSparseKVIndexRemapFailClosed(t *testing.T) {
	pageTable := []int32{1, 2, 3}
	reqOffsets := []int{0}
	reqTokens := []int{2}
	out := make([]int32, 2)

	if _, err := RemapSparseLogicalToPhysicalKV([]int32{0, 1}, pageTable, reqOffsets, reqTokens, 0, out); err == nil {
		t.Fatal("expected error on pageSize <= 0")
	}

	if _, err := RemapSparseLogicalToPhysicalKV([]int32{0}, pageTable, reqOffsets, reqTokens, 16, out); err == nil {
		t.Fatal("expected error on mismatched tokens length")
	}

	if _, err := RemapSparseLogicalToPhysicalKV([]int32{0, -5}, pageTable, reqOffsets, reqTokens, 16, out); err == nil {
		t.Fatal("expected error on index < -1")
	}

	if _, err := RemapSparseLogicalToPhysicalKV([]int32{0, 100}, pageTable, reqOffsets, reqTokens, 16, out); err == nil {
		t.Fatal("expected error on page capacity exceeded")
	}
}
