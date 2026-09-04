package blobcommon

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sync"
	"testing"
)

func TestNewFixedChunker(t *testing.T) {
	// Invalid sizes
	for _, size := range []int{0, -1, -100} {
		c, err := NewFixedChunker(size)
		if c != nil || !errors.Is(err, ErrInvalidChunkSize) {
			t.Fatalf("NewFixedChunker(%d) expected ErrInvalidChunkSize, got c=%v, err=%v", size, c, err)
		}
	}

	// Valid size
	c, err := NewFixedChunker(1024)
	if err != nil || c == nil {
		t.Fatalf("NewFixedChunker(1024) failed: c=%v, err=%v", c, err)
	}
}

func TestNewDelimitedChunker(t *testing.T) {
	// Empty delimiter
	if _, err := NewDelimitedChunker(nil, 100); !errors.Is(err, ErrInvalidDelimiter) {
		t.Fatalf("expected ErrInvalidDelimiter on nil delimiter, got %v", err)
	}
	if _, err := NewDelimitedChunker([]byte{}, 100); !errors.Is(err, ErrInvalidDelimiter) {
		t.Fatalf("expected ErrInvalidDelimiter on empty delimiter, got %v", err)
	}

	// Non-positive max size
	if _, err := NewDelimitedChunker([]byte("\n"), 0); !errors.Is(err, ErrInvalidChunkSize) {
		t.Fatalf("expected ErrInvalidChunkSize on maxChunkSize=0, got %v", err)
	}
	if _, err := NewDelimitedChunker([]byte("\n"), -5); !errors.Is(err, ErrInvalidChunkSize) {
		t.Fatalf("expected ErrInvalidChunkSize on maxChunkSize=-5, got %v", err)
	}

	// maxChunkSize smaller than delimiter length
	if _, err := NewDelimitedChunker([]byte("delimiter-too-long"), 5); !errors.Is(err, ErrInvalidChunkSize) {
		t.Fatalf("expected ErrInvalidChunkSize when maxChunkSize < len(delim), got %v", err)
	}

	// Valid delimited chunker
	c, err := NewDelimitedChunker([]byte("\n"), 100)
	if err != nil || c == nil {
		t.Fatalf("NewDelimitedChunker failed: %v", err)
	}
}

func TestFixedChunkerEdgeCases(t *testing.T) {
	chunker, err := NewFixedChunker(10)
	if err != nil {
		t.Fatalf("failed to create chunker: %v", err)
	}

	// 1. Edge Case: Empty input
	t.Run("empty input", func(t *testing.T) {
		chunks, err := chunker.ChunkBytes([]byte{})
		if err != nil {
			t.Fatalf("ChunkBytes(empty) error: %v", err)
		}
		if len(chunks) != 0 {
			t.Fatalf("expected 0 chunks for empty input, got %d", len(chunks))
		}
	})

	// 2. Edge Case: Smaller than chunk size (7 bytes < 10)
	t.Run("smaller than chunk size", func(t *testing.T) {
		data := []byte("1234567")
		chunks, err := chunker.ChunkBytes(data)
		if err != nil {
			t.Fatalf("ChunkBytes failed: %v", err)
		}
		if len(chunks) != 1 {
			t.Fatalf("expected 1 chunk, got %d", len(chunks))
		}
		c0 := chunks[0]
		if c0.Index != 0 || c0.Offset != 0 || c0.Size != 7 {
			t.Fatalf("unexpected chunk metadata: index=%d offset=%d size=%d", c0.Index, c0.Offset, c0.Size)
		}
		if !bytes.Equal(c0.Data, data) {
			t.Fatalf("chunk data mismatch: got %q, want %q", c0.Data, data)
		}
		if c0.Digest != DigestBytes(data) {
			t.Fatalf("chunk digest mismatch: got %s, want %s", c0.Digest, DigestBytes(data))
		}
	})

	// 3. Edge Case: Exact multiple of chunk size (30 bytes == 3 * 10)
	t.Run("exact multiple", func(t *testing.T) {
		data := []byte("012345678901234567890123456789")
		chunks, err := chunker.ChunkBytes(data)
		if err != nil {
			t.Fatalf("ChunkBytes failed: %v", err)
		}
		if len(chunks) != 3 {
			t.Fatalf("expected exactly 3 chunks, got %d", len(chunks))
		}

		assertChunkInvariants(t, chunks, data)

		expectedOffsets := []int64{0, 10, 20}
		for i, c := range chunks {
			if c.Index != i {
				t.Errorf("chunk %d has Index %d, want %d", i, c.Index, i)
			}
			if c.Offset != expectedOffsets[i] {
				t.Errorf("chunk %d has Offset %d, want %d", i, c.Offset, expectedOffsets[i])
			}
			if c.Size != 10 {
				t.Errorf("chunk %d has Size %d, want 10", i, c.Size)
			}
		}
	})

	// 4. Non-multiple remainder (25 bytes == 2 * 10 + 5)
	t.Run("non multiple remainder", func(t *testing.T) {
		data := []byte("abcdefghijklmnopqrstuvwxy") // 25 bytes
		chunks, err := chunker.ChunkBytes(data)
		if err != nil {
			t.Fatalf("ChunkBytes failed: %v", err)
		}
		if len(chunks) != 3 {
			t.Fatalf("expected 3 chunks, got %d", len(chunks))
		}
		assertChunkInvariants(t, chunks, data)

		if chunks[0].Size != 10 || chunks[1].Size != 10 || chunks[2].Size != 5 {
			t.Fatalf("unexpected chunk sizes: %d, %d, %d", chunks[0].Size, chunks[1].Size, chunks[2].Size)
		}
		if chunks[2].Offset != 20 {
			t.Fatalf("chunk 2 offset %d, want 20", chunks[2].Offset)
		}
	})
}

func TestDelimitedChunker(t *testing.T) {
	chunker, err := NewDelimitedChunker([]byte("\n"), 20)
	if err != nil {
		t.Fatalf("NewDelimitedChunker failed: %v", err)
	}

	// 1. Empty input
	t.Run("empty input", func(t *testing.T) {
		chunks, err := chunker.ChunkBytes([]byte{})
		if err != nil {
			t.Fatalf("ChunkBytes empty failed: %v", err)
		}
		if len(chunks) != 0 {
			t.Fatalf("expected 0 chunks, got %d", len(chunks))
		}
	})

	// 2. Delimited lines with trailing newline
	t.Run("clean lines with trailing newline", func(t *testing.T) {
		data := []byte("alpha\nbeta\ngamma\n")
		chunks, err := chunker.ChunkBytes(data)
		if err != nil {
			t.Fatalf("ChunkBytes failed: %v", err)
		}
		if len(chunks) != 3 {
			t.Fatalf("expected 3 chunks, got %d", len(chunks))
		}
		assertChunkInvariants(t, chunks, data)

		expectedStrings := []string{"alpha\n", "beta\n", "gamma\n"}
		for i, c := range chunks {
			if string(c.Data) != expectedStrings[i] {
				t.Fatalf("chunk %d data = %q, want %q", i, c.Data, expectedStrings[i])
			}
		}
	})

	// 3. Delimited lines without trailing newline
	t.Run("without trailing newline", func(t *testing.T) {
		data := []byte("part1\npart2")
		chunks, err := chunker.ChunkBytes(data)
		if err != nil {
			t.Fatalf("ChunkBytes failed: %v", err)
		}
		if len(chunks) != 2 {
			t.Fatalf("expected 2 chunks, got %d", len(chunks))
		}
		assertChunkInvariants(t, chunks, data)
		if string(chunks[0].Data) != "part1\n" || string(chunks[1].Data) != "part2" {
			t.Fatalf("unexpected chunk parts: %q, %q", chunks[0].Data, chunks[1].Data)
		}
	})

	// 4. Line exceeding maxChunkSize fallback
	t.Run("line exceeding maxChunkSize", func(t *testing.T) {
		// maxChunkSize is 20; 30 chars then newline
		longLine := "012345678901234567890123456789\n"
		chunks, err := chunker.ChunkBytes([]byte(longLine))
		if err != nil {
			t.Fatalf("ChunkBytes failed: %v", err)
		}
		if len(chunks) != 2 {
			t.Fatalf("expected 2 chunks for long line, got %d", len(chunks))
		}
		assertChunkInvariants(t, chunks, []byte(longLine))
		if chunks[0].Size != 20 {
			t.Errorf("chunk 0 size = %d, want 20", chunks[0].Size)
		}
		if chunks[1].Size != 11 { // remaining 10 digits + '\n'
			t.Errorf("chunk 1 size = %d, want 11", chunks[1].Size)
		}
	})

	// 5. Multi-byte delimiter
	t.Run("multibyte delimiter", func(t *testing.T) {
		mbChunker, err := NewDelimitedChunker([]byte("\r\n"), 50)
		if err != nil {
			t.Fatalf("failed to create multibyte chunker: %v", err)
		}
		data := []byte("row1\r\nrow2\r\nrow3")
		chunks, err := mbChunker.ChunkBytes(data)
		if err != nil {
			t.Fatalf("ChunkBytes failed: %v", err)
		}
		if len(chunks) != 3 {
			t.Fatalf("expected 3 chunks, got %d", len(chunks))
		}
		assertChunkInvariants(t, chunks, data)
	})
}

func TestStreamingParity(t *testing.T) {
	data := []byte("record001\nrecord002\nrecord003\nrecord004\n")

	chunker, err := NewDelimitedChunker([]byte("\n"), 15)
	if err != nil {
		t.Fatalf("NewDelimitedChunker failed: %v", err)
	}

	byteChunks, err := chunker.ChunkBytes(data)
	if err != nil {
		t.Fatalf("ChunkBytes failed: %v", err)
	}

	streamChunks, err := chunker.AllChunks(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("AllChunks failed: %v", err)
	}

	if !reflect.DeepEqual(byteChunks, streamChunks) {
		t.Fatalf("streaming chunks do not match byte chunks:\nbyte: %+v\nstream: %+v", byteChunks, streamChunks)
	}
}

func TestChunkReaderGuards(t *testing.T) {
	chunker, err := NewFixedChunker(10)
	if err != nil {
		t.Fatalf("NewFixedChunker failed: %v", err)
	}

	// Nil reader guard
	if err := chunker.ChunkReader(nil, func(c Chunk) error { return nil }); !errors.Is(err, ErrCorruptedBlob) {
		t.Fatalf("expected ErrCorruptedBlob on nil reader, got %v", err)
	}

	// Nil callback guard
	if err := chunker.ChunkReader(bytes.NewReader([]byte("test")), nil); err == nil {
		t.Fatalf("expected error on nil callback")
	}

	// Early abort via callback
	abortErr := errors.New("abort stream")
	data := []byte("01234567890123456789")
	callCount := 0
	err = chunker.ChunkReader(bytes.NewReader(data), func(c Chunk) error {
		callCount++
		return abortErr
	})
	if !errors.Is(err, abortErr) {
		t.Fatalf("expected abortErr, got %v", err)
	}
	if callCount != 1 {
		t.Fatalf("expected exactly 1 call before abort, got %d", callCount)
	}

	// Broken reader
	broken := &errReader{err: io.ErrUnexpectedEOF}
	err = chunker.ChunkReader(broken, func(c Chunk) error { return nil })
	// Should terminate cleanly or report
	if err != nil && !errors.Is(err, ErrCorruptedBlob) {
		t.Fatalf("unexpected error from broken reader: %v", err)
	}
}

func TestChunkerConcurrency(t *testing.T) {
	chunker, err := NewFixedChunker(16)
	if err != nil {
		t.Fatalf("NewFixedChunker failed: %v", err)
	}

	var wg sync.WaitGroup
	workers := 10
	iterations := 25

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				payload := []byte(fmt.Sprintf("worker-%d-iteration-%d-payload-padding-bytes", workerID, i))
				chunks, err := chunker.ChunkBytes(payload)
				if err != nil {
					t.Errorf("worker %d iteration %d failed: %v", workerID, i, err)
					return
				}
				assertChunkInvariants(t, chunks, payload)
			}
		}(w)
	}
	wg.Wait()
}

func assertChunkInvariants(t *testing.T, chunks []Chunk, originalData []byte) {
	t.Helper()

	var assembled []byte
	var expectedOffset int64

	for i, c := range chunks {
		// Monotonic index
		if c.Index != i {
			t.Fatalf("chunk %d has incorrect Index %d", i, c.Index)
		}
		// Strictly monotonic offset
		if c.Offset != expectedOffset {
			t.Fatalf("chunk %d has Offset %d, want strictly monotonic %d", i, c.Offset, expectedOffset)
		}
		// Size matches data length
		if c.Size != len(c.Data) {
			t.Fatalf("chunk %d Size %d != len(Data) %d", i, c.Size, len(c.Data))
		}
		// Digest integrity
		expectedDigest := DigestBytes(c.Data)
		if c.Digest != expectedDigest {
			t.Fatalf("chunk %d Digest mismatch: got %s, want %s", i, c.Digest, expectedDigest)
		}

		assembled = append(assembled, c.Data...)
		expectedOffset += int64(c.Size)
	}

	// Sum of sizes equals total length
	if int64(len(assembled)) != int64(len(originalData)) {
		t.Fatalf("sum of chunk sizes %d != original data length %d", len(assembled), len(originalData))
	}

	// Byte-for-byte reconstruction
	if !bytes.Equal(assembled, originalData) {
		t.Fatalf("assembled chunks do not match original data")
	}
}
