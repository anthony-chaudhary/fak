package ggufload

import (
	"errors"
	"fmt"
	"io"
	"math"
)

// maxTensorRowF32Bytes bounds both the decoded row and its raw payload. Row
// reads serve streamed tables, so malformed metadata must fail before make.
const maxTensorRowF32Bytes uint64 = 64 << 20

// ErrTensorRow identifies a refused or failed row-scoped tensor read.
var ErrTensorRow = errors.New("gguf: tensor row read failed")

// TensorRowF32 reads and decodes one contiguous row of a two-dimensional GGUF
// tensor. GGUF dimension zero is the row width; quantized rows must end on a
// storage-block boundary so a row never borrows bytes from its neighbour.
func (s *WeightSource) TensorRowF32(name string, row int) ([]float32, TensorInfo, error) {
	if s == nil || s.File == nil {
		return nil, TensorInfo{}, fmt.Errorf("%w: nil weight source", ErrTensorRow)
	}
	info, ok := s.Tensor(name)
	if !ok {
		return nil, TensorInfo{}, fmt.Errorf("%w: missing tensor %s", ErrTensorRow, name)
	}
	if len(info.Dims) != 2 {
		return nil, info, fmt.Errorf("%w: tensor %s has %d dimensions, want 2", ErrTensorRow, name, len(info.Dims))
	}
	if row < 0 || uint64(row) >= info.Dims[1] {
		return nil, info, fmt.Errorf("%w: tensor %s row %d is outside [0,%d)", ErrTensorRow, name, row, info.Dims[1])
	}

	width := info.Dims[0]
	if width > maxTensorRowF32Bytes/4 {
		return nil, info, fmt.Errorf("%w: tensor %s row would decode to more than %d bytes", ErrTensorRow, name, maxTensorRowF32Bytes)
	}
	rowInfo := info
	rowInfo.Dims = []uint64{width}
	rowBytes, err := tensorPayloadBytes(rowInfo)
	if err != nil {
		return nil, info, fmt.Errorf("%w: %v", ErrTensorRow, err)
	}
	if rowBytes > maxTensorRowF32Bytes || rowBytes > uint64(math.MaxInt) {
		return nil, info, fmt.Errorf("%w: tensor %s row payload exceeds %d bytes", ErrTensorRow, name, maxTensorRowF32Bytes)
	}
	totalBytes, err := tensorPayloadBytes(info)
	if err != nil {
		return nil, info, fmt.Errorf("%w: %v", ErrTensorRow, err)
	}

	r, size, err := s.tensorReader(info)
	if err != nil {
		return nil, info, fmt.Errorf("%w: %v", ErrTensorRow, err)
	}
	if info.FileOffset < 0 || totalBytes > math.MaxInt64 || info.FileOffset > math.MaxInt64-int64(totalBytes) || info.FileOffset+int64(totalBytes) > size {
		return nil, info, fmt.Errorf("%w: tensor %s overruns file", ErrTensorRow, name)
	}
	rowIndex := uint64(row)
	if rowBytes != 0 && rowIndex > math.MaxUint64/rowBytes {
		return nil, info, fmt.Errorf("%w: tensor %s row offset overflows uint64", ErrTensorRow, name)
	}
	delta := rowIndex * rowBytes
	if delta > math.MaxInt64 || info.FileOffset > math.MaxInt64-int64(delta) {
		return nil, info, fmt.Errorf("%w: tensor %s row offset overflows int64", ErrTensorRow, name)
	}
	offset := info.FileOffset + int64(delta)

	raw := make([]byte, int(rowBytes))
	n, readErr := r.ReadAt(raw, offset)
	if readErr != nil || n != len(raw) {
		if readErr == nil {
			readErr = io.ErrUnexpectedEOF
		}
		return nil, info, fmt.Errorf("%w: read tensor %s row %d: %w", ErrTensorRow, name, row, readErr)
	}
	out, err := dequantF32(rowInfo, raw)
	if err != nil {
		return nil, info, fmt.Errorf("%w: %v", ErrTensorRow, err)
	}
	return out, info, nil
}
