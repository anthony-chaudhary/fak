package compute

import "strconv"

func singleResourceCapExceeded(nbytes int, capBytes int64) bool {
	return capBytes > 0 && int64(nbytes) > capBytes
}

type q8RowChunk struct {
	start int
	rows  int
}

func q8RowChunksForCap(out, in, block int, capBytes int64) ([]q8RowChunk, bool, bool) {
	if out <= 0 || in <= 0 || block <= 0 || in%block != 0 {
		return []q8RowChunk{{start: 0, rows: out}}, false, true
	}
	if capBytes <= 0 {
		return []q8RowChunk{{start: 0, rows: out}}, false, true
	}
	codeRowBytes := int64(in)
	scaleRowBytes := int64(in/block) * int64(F32.Bytes())
	rowsByCode := capBytes / codeRowBytes
	rowsByScale := capBytes / scaleRowBytes
	rowsPerChunk := rowsByCode
	if rowsByScale < rowsPerChunk {
		rowsPerChunk = rowsByScale
	}
	if rowsPerChunk <= 0 {
		return nil, true, false
	}
	if rowsPerChunk >= int64(out) {
		return []q8RowChunk{{start: 0, rows: out}}, false, true
	}
	chunks := make([]q8RowChunk, 0, (int64(out)+rowsPerChunk-1)/rowsPerChunk)
	for start := 0; start < out; start += int(rowsPerChunk) {
		rows := int(rowsPerChunk)
		if start+rows > out {
			rows = out - start
		}
		chunks = append(chunks, q8RowChunk{start: start, rows: rows})
	}
	return chunks, true, true
}

func formatVulkanResourceCapError(what string, nbytes int, capBytes, storageRange, allocationSize int64) string {
	return "compute: vulkan " + what + " allocation of " + strconv.Itoa(nbytes) +
		" bytes exceeds device single-resource cap " + strconv.FormatInt(capBytes, 10) +
		" bytes (maxStorageBufferRange=" + strconv.FormatInt(storageRange, 10) +
		", maxMemoryAllocationSize=" + strconv.FormatInt(allocationSize, 10) +
		"); split/chunk this tensor before uploading"
}

func shapeText(shape []int) string {
	if len(shape) == 0 {
		return "[]"
	}
	s := "["
	for i, d := range shape {
		if i > 0 {
			s += ","
		}
		s += strconv.Itoa(d)
	}
	return s + "]"
}
