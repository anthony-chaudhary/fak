package ggufload

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestReadTensorFileOffsetBounds(t *testing.T) {
	t.Run("unsigned addition cannot wrap", func(t *testing.T) {
		fixture, data := tensorOffsetFixture(t, 32, func(data uint64) uint64 {
			return math.MaxUint64 - (data - 1)
		})
		if data == 0 {
			t.Fatal("fixture must have a non-zero tensor data offset")
		}
		if _, err := Read(bytes.NewReader(fixture)); err == nil {
			t.Fatal("Read accepted a tensor offset whose absolute file offset wraps uint64")
		}
	})

	t.Run("largest aligned offset below MaxInt64 is representable", func(t *testing.T) {
		const alignment = uint32(8)
		const maxAlignedInt64 = uint64(math.MaxInt64) &^ uint64(alignment-1)
		fixture, data := tensorOffsetFixture(t, alignment, func(data uint64) uint64 {
			return maxAlignedInt64 - data
		})
		f, err := Read(bytes.NewReader(fixture))
		if err != nil {
			t.Fatalf("Read rejected largest aligned tensor file offset below MaxInt64: %v", err)
		}
		if got := f.Tensors[0].FileOffset; got != int64(maxAlignedInt64) {
			t.Fatalf("tensor file offset = %d, want %d (data offset %d)", got, maxAlignedInt64, data)
		}
	})

	t.Run("first aligned offset above MaxInt64 is rejected", func(t *testing.T) {
		const firstAlignedAboveMaxInt64 = uint64(math.MaxInt64) + 1
		fixture, _ := tensorOffsetFixture(t, 8, func(data uint64) uint64 {
			return firstAlignedAboveMaxInt64 - data
		})
		if _, err := Read(bytes.NewReader(fixture)); err == nil {
			t.Fatal("Read accepted the first aligned tensor file offset above MaxInt64")
		}
	})

	t.Run("normal aligned offset remains exact", func(t *testing.T) {
		const relativeOffset = uint64(64)
		fixture, data := tensorOffsetFixture(t, 32, func(uint64) uint64 { return relativeOffset })
		f, err := Read(bytes.NewReader(fixture))
		if err != nil {
			t.Fatalf("Read rejected normal aligned tensor offset: %v", err)
		}
		if got, want := f.Tensors[0].FileOffset, int64(data+relativeOffset); got != want {
			t.Fatalf("tensor file offset = %d, want %d", got, want)
		}
	})
}

func tensorOffsetFixture(t *testing.T, alignment uint32, offset func(data uint64) uint64) ([]byte, uint64) {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 1)
	writeKVUint32(&b, "general.alignment", alignment)
	writeTensorInfoForTest(&b, "weight", []uint64{1}, TensorF32, 0)

	const encodedOffsetBytes = 8
	offsetPos := b.Len() - encodedOffsetBytes
	data := alignOffset(uint64(b.Len()), uint64(alignment))
	binary.LittleEndian.PutUint64(b.Bytes()[offsetPos:], offset(data))
	return b.Bytes(), data
}
