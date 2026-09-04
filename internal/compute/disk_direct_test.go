package compute

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"testing"
	"unsafe"
)

func TestAlignedBlockBufferAllocationAndAlignment(t *testing.T) {
	sizes := []int{
		0, -10, 1, 16, 512, 4095, 4096, 4097, 8192,
		65536, 1024 * 1024, 2 * 1024 * 1024, 8 * 1024 * 1024,
	}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			buf := NewAlignedBlockBuffer(size)
			if buf == nil {
				t.Fatal("NewAlignedBlockBuffer returned nil")
			}

			if size <= 0 {
				if buf.Len() != 0 || buf.Cap() != 0 {
					t.Fatalf("expected 0 len/cap for size %d, got len=%d cap=%d", size, buf.Len(), buf.Cap())
				}
				if buf.Bytes() != nil {
					t.Fatalf("expected nil bytes for size %d", size)
				}
				if !buf.IsAligned() {
					t.Fatal("expected empty buffer to report aligned")
				}
				return
			}

			if buf.Len() != size {
				t.Fatalf("expected len %d, got %d", size, buf.Len())
			}
			if buf.Cap() != size {
				t.Fatalf("expected cap %d, got %d", size, buf.Cap())
			}
			if len(buf.Block) != size {
				t.Fatalf("expected Block len %d, got %d", size, len(buf.Block))
			}

			addr := uintptr(unsafe.Pointer(&buf.Block[0]))
			if addr%uintptr(DirectIOAlignment) != 0 {
				t.Fatalf("buffer pointer %x is not aligned to %d bytes (remainder: %d)",
					addr, DirectIOAlignment, addr%uintptr(DirectIOAlignment))
			}
			if !buf.IsAligned() {
				t.Fatal("IsAligned() reported false for aligned buffer")
			}

			// Verify read/write integrity at boundaries.
			if size == 1 {
				buf.Block[0] = 0xAA
				if buf.Block[0] != 0xAA {
					t.Fatal("readback mismatch for size 1")
				}
			} else if size == 2 {
				buf.Block[0] = 0xAA
				buf.Block[1] = 0xBB
				if buf.Block[0] != 0xAA || buf.Block[1] != 0xBB {
					t.Fatal("readback mismatch for size 2")
				}
			} else {
				buf.Block[0] = 0xAA
				buf.Block[size/2] = 0xBB
				buf.Block[size-1] = 0xCC

				if buf.Block[0] != 0xAA || buf.Block[size/2] != 0xBB || buf.Block[size-1] != 0xCC {
					t.Fatal("readback mismatch after writing boundary bytes")
				}
			}

			if size >= 16 {
				sub := buf.SubBlock(4, 12)
				if len(sub) != 8 {
					t.Fatalf("expected SubBlock len 8, got %d", len(sub))
				}
			}

			buf.Reset()
			for i := range buf.Block {
				if buf.Block[i] != 0 {
					t.Fatalf("Reset() did not clear buffer at index %d", i)
				}
			}
		})
	}
}

func TestDropFilePagesPlatformSafe(t *testing.T) {
	// Zero fd or invalid length should safely return nil.
	if err := DropFilePages(0, 0, 4096); err != nil {
		t.Fatalf("DropFilePages(0, 0, 4096) failed: %v", err)
	}
	if err := DropFilePages(0, 0, 0); err != nil {
		t.Fatalf("DropFilePages(0, 0, 0) failed: %v", err)
	}
	if err := DropFilePages(0, 0, -1); err != nil {
		t.Fatalf("DropFilePages(0, 0, -1) failed: %v", err)
	}

	// Test with a real file on disk.
	tmpFile, err := os.CreateTemp(t.TempDir(), "fak_direct_drop_test_*")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer tmpFile.Close()

	testData := make([]byte, 64*1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	if _, err := tmpFile.Write(testData); err != nil {
		t.Fatalf("failed to write test data: %v", err)
	}
	if err := tmpFile.Sync(); err != nil {
		t.Fatalf("failed to sync test file: %v", err)
	}

	fd := tmpFile.Fd()

	// Evict full file.
	if err := DropFilePages(fd, 0, int64(len(testData))); err != nil {
		t.Fatalf("DropFilePages full file eviction failed: %v", err)
	}

	// Evict block sub-ranges.
	if err := DropFilePages(fd, 4096, 8192); err != nil {
		t.Fatalf("DropFilePages sub-range eviction failed: %v", err)
	}
	if err := DropFilePages(fd, 32768, 4096); err != nil {
		t.Fatalf("DropFilePages block eviction failed: %v", err)
	}

	// Evict with negative offset (must clamp to 0 without failing).
	if err := DropFilePages(fd, -100, 4096); err != nil {
		t.Fatalf("DropFilePages negative offset failed: %v", err)
	}

	// Evict with 0 length (safe no-op).
	if err := DropFilePages(fd, 0, 0); err != nil {
		t.Fatalf("DropFilePages zero length failed: %v", err)
	}
}

func TestDirectIOReader(t *testing.T) {
	data := make([]byte, 32*1024)
	for i := range data {
		data[i] = byte((i * 31) ^ 0x3C)
	}

	// 1. In-memory ReaderAt test
	memReader := bytes.NewReader(data)
	r := NewDirectIOReader(memReader, int64(len(data)), 4096, true)

	if r.Size() != int64(len(data)) {
		t.Fatalf("expected size %d, got %d", len(data), r.Size())
	}
	if r.BlockSize() != 4096 {
		t.Fatalf("expected block size 4096, got %d", r.BlockSize())
	}
	if !r.IsEvictEnabled() {
		t.Fatal("expected evict enabled")
	}

	block, err := r.ReadAlignedBlock(4096, 4096)
	if err != nil {
		t.Fatalf("ReadAlignedBlock failed: %v", err)
	}
	if !bytes.Equal(block, data[4096:8192]) {
		t.Fatal("ReadAlignedBlock mismatch")
	}

	seqBuf := make([]byte, 1024)
	n, err := r.Read(seqBuf)
	if err != nil || n != 1024 {
		t.Fatalf("sequential Read failed: n=%d err=%v", n, err)
	}
	if !bytes.Equal(seqBuf, data[:1024]) {
		t.Fatal("sequential Read mismatch")
	}
	if r.Offset() != 1024 {
		t.Fatalf("expected offset 1024, got %d", r.Offset())
	}

	// Test Seek
	pos, err := r.Seek(8192, io.SeekStart)
	if err != nil || pos != 8192 {
		t.Fatalf("Seek failed: pos=%d err=%v", pos, err)
	}
	n, err = r.Read(seqBuf)
	if err != nil || n != 1024 {
		t.Fatalf("Read after Seek failed: n=%d err=%v", n, err)
	}
	if !bytes.Equal(seqBuf, data[8192:9216]) {
		t.Fatal("Read after Seek mismatch")
	}

	// 2. Real file test
	tmpFile, err := os.CreateTemp(t.TempDir(), "fak_direct_reader_*")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	defer tmpFile.Close()

	if _, err := tmpFile.Write(data); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	if err := tmpFile.Sync(); err != nil {
		t.Fatalf("sync temp file: %v", err)
	}

	fReader, err := NewDirectIOReaderFromFile(tmpFile, true)
	if err != nil {
		t.Fatalf("NewDirectIOReaderFromFile failed: %v", err)
	}
	if fReader.Fd() != tmpFile.Fd() {
		t.Fatalf("expected fd %v, got %v", tmpFile.Fd(), fReader.Fd())
	}
	if fReader.Size() != int64(len(data)) {
		t.Fatalf("expected file size %d, got %d", len(data), fReader.Size())
	}

	if err := fReader.Evict(0, int64(len(data))); err != nil {
		t.Fatalf("fReader.Evict failed: %v", err)
	}

	readAll := make([]byte, len(data))
	nTotal, err := io.ReadFull(fReader, readAll)
	if err != nil || nTotal != len(data) {
		t.Fatalf("io.ReadFull failed: n=%d err=%v", nTotal, err)
	}
	if !bytes.Equal(readAll, data) {
		t.Fatal("read file data mismatch")
	}
}

// syntheticWeightsReader provides deterministic synthetic weights without holding 100+ MB in RAM.
type syntheticWeightsReader struct {
	size int64
}

func (s *syntheticWeightsReader) ReadAt(p []byte, off int64) (int, error) {
	if off >= s.size {
		return 0, io.EOF
	}
	toRead := len(p)
	remaining := s.size - off
	var err error
	if int64(toRead) >= remaining {
		toRead = int(remaining)
		err = io.EOF
	}
	for i := 0; i < toRead; i++ {
		p[i] = byte((off + int64(i)) ^ 0x5A)
	}
	return toRead, err
}

func TestStreamWeightsWithEviction100MB(t *testing.T) {
	const totalSize = int64(104 * 1024 * 1024) // 104 MiB (> 100MB)
	const chunkSize = 2 * 1024 * 1024          // 2 MiB chunks

	// 1. Synthetic streaming with integrity and chunk verification
	synth := &syntheticWeightsReader{size: totalSize}

	var processedBytes int64
	var chunkCount int
	hasher := sha256.New()

	streamed, err := StreamWeightsWithEviction(synth, totalSize, chunkSize, func(chunk []byte) error {
		chunkCount++
		// Verify chunk size
		expectedLen := chunkSize
		if processedBytes+int64(expectedLen) > totalSize {
			expectedLen = int(totalSize - processedBytes)
		}
		if len(chunk) != expectedLen {
			return fmt.Errorf("chunk %d len %d != expected %d", chunkCount, len(chunk), expectedLen)
		}

		// Verify chunk bytes at start, middle, and end
		offset := processedBytes
		if chunk[0] != byte(offset^0x5A) {
			return fmt.Errorf("chunk start mismatch at %d", offset)
		}
		mid := len(chunk) / 2
		if chunk[mid] != byte((offset+int64(mid))^0x5A) {
			return fmt.Errorf("chunk mid mismatch at %d", offset+int64(mid))
		}
		last := len(chunk) - 1
		if chunk[last] != byte((offset+int64(last))^0x5A) {
			return fmt.Errorf("chunk last mismatch at %d", offset+int64(last))
		}

		hasher.Write(chunk)
		processedBytes += int64(len(chunk))
		return nil
	}, true)

	if err != nil {
		t.Fatalf("StreamWeightsWithEviction failed: %v", err)
	}
	if streamed != totalSize {
		t.Fatalf("streamed %d bytes, want %d", streamed, totalSize)
	}
	if processedBytes != totalSize {
		t.Fatalf("processed %d bytes, want %d", processedBytes, totalSize)
	}
	expectedChunks := int(totalSize / chunkSize)
	if totalSize%chunkSize != 0 {
		expectedChunks++
	}
	if chunkCount != expectedChunks {
		t.Fatalf("chunk count %d, want %d", chunkCount, expectedChunks)
	}

	// 2. Real file test with active page eviction
	tmpFile, err := os.CreateTemp(t.TempDir(), "fak_stream_weights_*")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	defer tmpFile.Close()

	// Write 8 MiB to temp file for real OS page cache eviction validation
	const fileTestSize = int64(8 * 1024 * 1024)
	fileChunk := make([]byte, 1024*1024)
	for i := range fileChunk {
		fileChunk[i] = byte(i ^ 0x33)
	}
	for written := int64(0); written < fileTestSize; written += int64(len(fileChunk)) {
		if _, err := tmpFile.Write(fileChunk); err != nil {
			t.Fatalf("write temp file: %v", err)
		}
	}
	if err := tmpFile.Sync(); err != nil {
		t.Fatalf("sync temp file: %v", err)
	}

	var fileStreamedBytes int64
	streamedFile, err := StreamWeightsWithEviction(tmpFile, fileTestSize, 1024*1024, func(chunk []byte) error {
		fileStreamedBytes += int64(len(chunk))
		return nil
	}, true)
	if err != nil {
		t.Fatalf("StreamWeightsWithEviction on file failed: %v", err)
	}
	if streamedFile != fileTestSize || fileStreamedBytes != fileTestSize {
		t.Fatalf("streamed file %d / %d bytes, want %d", streamedFile, fileStreamedBytes, fileTestSize)
	}

	// 3. Error propagation test: onChunk error must halt streaming
	testErr := errors.New("simulated callback failure")
	_, err = StreamWeightsWithEviction(synth, totalSize, chunkSize, func(chunk []byte) error {
		return testErr
	}, true)
	if !errors.Is(err, testErr) {
		t.Fatalf("expected error %v, got %v", testErr, err)
	}
}

func BenchmarkStreamWeightsAligned(b *testing.B) {
	const dataSize = int64(64 * 1024 * 1024) // 64 MiB
	const chunkSize = 2 * 1024 * 1024        // 2 MiB chunks
	synth := &syntheticWeightsReader{size: dataSize}

	b.SetBytes(dataSize)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var bytesRead int64
		_, err := StreamWeightsWithEviction(synth, dataSize, chunkSize, func(chunk []byte) error {
			bytesRead += int64(len(chunk))
			return nil
		}, false)
		if err != nil {
			b.Fatalf("aligned stream failed: %v", err)
		}
	}
}

func BenchmarkStreamWeightsStandardBuffered(b *testing.B) {
	const dataSize = int64(64 * 1024 * 1024) // 64 MiB
	const chunkSize = 2 * 1024 * 1024        // 2 MiB chunks
	synth := &syntheticWeightsReader{size: dataSize}

	b.SetBytes(dataSize)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Standard unaligned buffer copy
		buf := make([]byte, chunkSize)
		var total int64
		for off := int64(0); off < dataSize; {
			toRead := chunkSize
			if remaining := dataSize - off; int64(toRead) > remaining {
				toRead = int(remaining)
			}
			n, err := synth.ReadAt(buf[:toRead], off)
			if n > 0 {
				total += int64(n)
				off += int64(n)
			}
			if err != nil && err != io.EOF {
				b.Fatalf("standard read failed: %v", err)
			}
			if n == 0 {
				break
			}
		}
	}
}
