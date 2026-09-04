// Package blobcommon provides core blob utilities for content-addressed storage,
// digest generation, integrity verification, and chunking over byte slices and streams.
//
// All operations adhere to fail-closed semantics on corrupted, truncated, or
// out-of-bounds blobs.
//
// Invariant: Content digests are cryptographically deterministic SHA-256 hex representations.
// Guard: fail-closed on corrupted blobs
package blobcommon

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"strings"
)

var (
	// ErrEmptyBlob indicates a zero-length blob was provided when prohibited.
	ErrEmptyBlob = errors.New("blobcommon: empty blob not allowed")

	// ErrBlobTooLarge indicates blob size exceeds the configured upper limit.
	ErrBlobTooLarge = errors.New("blobcommon: blob size exceeds maximum limit")

	// ErrBlobTooSmall indicates blob size is below the configured lower limit.
	ErrBlobTooSmall = errors.New("blobcommon: blob size below minimum limit")

	// ErrInvalidDigest indicates a malformed or invalid digest string format.
	ErrInvalidDigest = errors.New("blobcommon: invalid digest format")

	// ErrDigestMismatch indicates the computed digest does not match the expected digest.
	ErrDigestMismatch = errors.New("blobcommon: digest verification mismatch")

	// ErrInvalidChunkSize indicates a non-positive or unsupported chunk size was specified.
	ErrInvalidChunkSize = errors.New("blobcommon: invalid chunk size")

	// ErrInvalidDelimiter indicates an empty or unsupported chunking delimiter was specified.
	ErrInvalidDelimiter = errors.New("blobcommon: invalid delimiter")

	// ErrCorruptedBlob indicates corrupted data encountered during stream verification.
	ErrCorruptedBlob = errors.New("blobcommon: corrupted blob detected")
)

// BlobLimits defines size and occupancy constraints for blob validation.
type BlobLimits struct {
	// MaxSize is the maximum allowed byte length. A value of 0 indicates unbounded.
	MaxSize int64
	// MinSize is the minimum allowed byte length.
	MinSize int64
	// AllowEmpty specifies whether zero-byte blobs are permissible.
	AllowEmpty bool
}

// IsEmptyBlob reports whether data represents a zero-length blob.
func IsEmptyBlob(data []byte) bool {
	return len(data) == 0
}

// ValidateBlob checks the provided byte slice against the supplied BlobLimits.
//
// Guard: fail-closed on corrupted blobs
// Invariant: Any blob violating configured limits or integrity checks is rejected.
func ValidateBlob(data []byte, limits BlobLimits) error {
	size := int64(len(data))
	if size == 0 {
		if !limits.AllowEmpty {
			return ErrEmptyBlob
		}
		if limits.MinSize > 0 {
			return ErrBlobTooSmall
		}
		return nil
	}

	if limits.MinSize > 0 && size < limits.MinSize {
		return ErrBlobTooSmall
	}
	if limits.MaxSize > 0 && size > limits.MaxSize {
		return ErrBlobTooLarge
	}
	return nil
}

// ValidateBlobReader reads r to completion and validates its total size against limits.
// Returns the total number of bytes read from r.
//
// Guard: fail-closed on corrupted blobs
// Invariant: Any blob violating configured limits or integrity checks is rejected.
func ValidateBlobReader(r io.Reader, limits BlobLimits) (int64, error) {
	if r == nil {
		return 0, fmt.Errorf("%w: nil reader", ErrCorruptedBlob)
	}

	var total int64
	buf := make([]byte, 32*1024)

	for {
		n, err := r.Read(buf)
		if n > 0 {
			total += int64(n)
			if limits.MaxSize > 0 && total > limits.MaxSize {
				return total, ErrBlobTooLarge
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return total, fmt.Errorf("%w: %v", ErrCorruptedBlob, err)
		}
	}

	if total == 0 {
		if !limits.AllowEmpty {
			return 0, ErrEmptyBlob
		}
		if limits.MinSize > 0 {
			return 0, ErrBlobTooSmall
		}
		return 0, nil
	}

	if limits.MinSize > 0 && total < limits.MinSize {
		return total, ErrBlobTooSmall
	}

	return total, nil
}

// ValidateDigest checks whether digest is a syntactically valid 64-character lowercase SHA-256 hex string.
//
// Guard: fail-closed on corrupted blobs
func ValidateDigest(digest string) error {
	if len(digest) != 64 {
		return fmt.Errorf("%w: expected 64 hex characters, got %d", ErrInvalidDigest, len(digest))
	}
	for i := 0; i < len(digest); i++ {
		c := digest[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return fmt.Errorf("%w: non-hex or uppercase character at index %d", ErrInvalidDigest, i)
		}
	}
	return nil
}

// DigestBytes computes the canonical SHA-256 hex digest for data.
//
// Invariant: Digest generation is deterministic; identical bytes yield identical SHA-256 hex.
func DigestBytes(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

// DigestReader streams r through SHA-256, returning the hex digest and total byte count.
//
// Invariant: Digest generation is deterministic; identical bytes yield identical SHA-256 hex.
func DigestReader(r io.Reader) (string, int64, error) {
	if r == nil {
		return "", 0, fmt.Errorf("%w: nil reader", ErrCorruptedBlob)
	}
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		return "", n, fmt.Errorf("%w: %v", ErrCorruptedBlob, err)
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// VerifyDigestPrefix checks whether digest is valid and starts with prefix.
// The prefix must be a non-empty, valid hex sequence with length <= 64.
//
// Guard: fail-closed on corrupted blobs
func VerifyDigestPrefix(digest, prefix string) bool {
	if len(prefix) == 0 || len(prefix) > 64 {
		return false
	}
	if err := ValidateDigest(digest); err != nil {
		return false
	}
	normPrefix := strings.ToLower(prefix)
	for i := 0; i < len(normPrefix); i++ {
		c := normPrefix[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return strings.HasPrefix(digest, normPrefix)
}

// VerifyChecksum compares data against expectedDigest in constant time.
//
// Guard: fail-closed on corrupted blobs
func VerifyChecksum(data []byte, expectedDigest string) error {
	normExpected := strings.ToLower(expectedDigest)
	if err := ValidateDigest(normExpected); err != nil {
		return err
	}
	actual := DigestBytes(data)
	if subtle.ConstantTimeCompare([]byte(actual), []byte(normExpected)) != 1 {
		return fmt.Errorf("%w: expected %s, got %s", ErrDigestMismatch, normExpected, actual)
	}
	return nil
}

// VerifyReaderChecksum reads r to completion, computes its SHA-256 digest, and compares against expectedDigest.
// Returns total bytes read and an error if verification fails.
//
// Guard: fail-closed on corrupted blobs
func VerifyReaderChecksum(r io.Reader, expectedDigest string) (int64, error) {
	normExpected := strings.ToLower(expectedDigest)
	if err := ValidateDigest(normExpected); err != nil {
		return 0, err
	}
	actual, n, err := DigestReader(r)
	if err != nil {
		return n, err
	}
	if subtle.ConstantTimeCompare([]byte(actual), []byte(normExpected)) != 1 {
		return n, fmt.Errorf("%w: expected %s, got %s", ErrDigestMismatch, normExpected, actual)
	}
	return n, nil
}

// ValidateAndVerifyBlob enforces size constraints and verifies the SHA-256 digest of data.
//
// Guard: fail-closed on corrupted blobs
// Invariant: Any blob violating configured limits or integrity checks is rejected.
func ValidateAndVerifyBlob(data []byte, limits BlobLimits, expectedDigest string) error {
	if err := ValidateBlob(data, limits); err != nil {
		return err
	}
	return VerifyChecksum(data, expectedDigest)
}

// ValidateAndVerifyReader streams r, validates size bounds, and verifies expectedDigest in a single pass.
//
// Guard: fail-closed on corrupted blobs
// Invariant: Any blob violating configured limits or integrity checks is rejected.
func ValidateAndVerifyReader(r io.Reader, limits BlobLimits, expectedDigest string) (int64, error) {
	normExpected := strings.ToLower(expectedDigest)
	if err := ValidateDigest(normExpected); err != nil {
		return 0, err
	}
	if r == nil {
		return 0, fmt.Errorf("%w: nil reader", ErrCorruptedBlob)
	}

	h := sha256.New()
	var total int64
	buf := make([]byte, 32*1024)

	for {
		n, err := r.Read(buf)
		if n > 0 {
			total += int64(n)
			if limits.MaxSize > 0 && total > limits.MaxSize {
				return total, ErrBlobTooLarge
			}
			h.Write(buf[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return total, fmt.Errorf("%w: %v", ErrCorruptedBlob, err)
		}
	}

	if total == 0 {
		if !limits.AllowEmpty {
			return 0, ErrEmptyBlob
		}
		if limits.MinSize > 0 {
			return 0, ErrBlobTooSmall
		}
	} else if limits.MinSize > 0 && total < limits.MinSize {
		return total, ErrBlobTooSmall
	}

	actual := hex.EncodeToString(h.Sum(nil))
	if subtle.ConstantTimeCompare([]byte(actual), []byte(normExpected)) != 1 {
		return total, fmt.Errorf("%w: expected %s, got %s", ErrDigestMismatch, normExpected, actual)
	}

	return total, nil
}

// Chunk represents a contiguous segment of blob data with positional coordinates.
type Chunk struct {
	// Index is the zero-based sequential chunk index.
	Index int
	// Offset is the byte offset of chunk start relative to the entire blob.
	Offset int64
	// Size is the length of Data in bytes.
	Size int
	// Data contains the raw bytes belonging to this chunk.
	Data []byte
	// Digest is the SHA-256 hex digest of Data.
	Digest string
}

type chunkerMode int

const (
	modeFixed chunkerMode = iota
	modeDelimited
)

// Chunker partitions byte sequences and stream readers into fixed-size or delimited chunks.
// Chunker instances are immutable and safe for concurrent use by multiple goroutines.
type Chunker struct {
	mode         chunkerMode
	chunkSize    int
	delimiter    []byte
	maxChunkSize int
}

// NewFixedChunker creates a Chunker that splits data into fixed-size segments of chunkSize bytes.
//
// Guard: fail-closed on corrupted blobs
func NewFixedChunker(chunkSize int) (*Chunker, error) {
	if chunkSize <= 0 {
		return nil, fmt.Errorf("%w: chunkSize must be positive, got %d", ErrInvalidChunkSize, chunkSize)
	}
	return &Chunker{
		mode:      modeFixed,
		chunkSize: chunkSize,
	}, nil
}

// NewDelimitedChunker creates a Chunker that splits data on delimiter boundaries with a fallback maxChunkSize.
// The delimiter bytes are preserved at the end of each delimited chunk, preserving offset continuity.
//
// Guard: fail-closed on corrupted blobs
func NewDelimitedChunker(delimiter []byte, maxChunkSize int) (*Chunker, error) {
	if len(delimiter) == 0 {
		return nil, fmt.Errorf("%w: delimiter cannot be empty", ErrInvalidDelimiter)
	}
	if maxChunkSize <= 0 {
		return nil, fmt.Errorf("%w: maxChunkSize must be positive, got %d", ErrInvalidChunkSize, maxChunkSize)
	}
	if maxChunkSize < len(delimiter) {
		return nil, fmt.Errorf("%w: maxChunkSize (%d) must be at least delimiter length (%d)", ErrInvalidChunkSize, maxChunkSize, len(delimiter))
	}
	delimCopy := make([]byte, len(delimiter))
	copy(delimCopy, delimiter)
	return &Chunker{
		mode:         modeDelimited,
		delimiter:    delimCopy,
		maxChunkSize: maxChunkSize,
	}, nil
}

// ChunkBytes partitions data into a slice of Chunks.
// An empty slice returns an empty list of chunks and a nil error.
//
// Invariant: Chunk offsets are strictly monotonic; sum of chunk sizes equals total input length.
// Guard: fail-closed on corrupted blobs
func (c *Chunker) ChunkBytes(data []byte) ([]Chunk, error) {
	if len(data) == 0 {
		return []Chunk{}, nil
	}
	var chunks []Chunk
	err := c.ChunkReader(bytes.NewReader(data), func(chunk Chunk) error {
		chunks = append(chunks, chunk)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return chunks, nil
}

// ChunkReader streams r and invokes fn for each sequential chunk.
// If fn returns an error, streaming terminates and the error is returned.
//
// Invariant: Chunk offsets are strictly monotonic; sum of chunk sizes equals total input length.
// Guard: fail-closed on corrupted blobs
func (c *Chunker) ChunkReader(r io.Reader, fn func(chunk Chunk) error) error {
	if r == nil {
		return fmt.Errorf("%w: nil reader", ErrCorruptedBlob)
	}
	if fn == nil {
		return errors.New("blobcommon: nil chunk handler function")
	}

	switch c.mode {
	case modeFixed:
		return c.chunkReaderFixed(r, fn)
	case modeDelimited:
		return c.chunkReaderDelimited(r, fn)
	default:
		return fmt.Errorf("%w: unknown chunker mode", ErrCorruptedBlob)
	}
}

// AllChunks reads r to completion using ChunkReader and returns all collected chunks.
//
// Invariant: Chunk offsets are strictly monotonic; sum of chunk sizes equals total input length.
func (c *Chunker) AllChunks(r io.Reader) ([]Chunk, error) {
	var chunks []Chunk
	err := c.ChunkReader(r, func(chunk Chunk) error {
		chunks = append(chunks, chunk)
		return nil
	})
	if err != nil {
		return nil, err
	}
	if chunks == nil {
		return []Chunk{}, nil
	}
	return chunks, nil
}

func (c *Chunker) chunkReaderFixed(r io.Reader, fn func(chunk Chunk) error) error {
	var offset int64
	idx := 0
	buf := make([]byte, c.chunkSize)

	for {
		n, err := io.ReadFull(r, buf)
		if n > 0 {
			chunkData := make([]byte, n)
			copy(chunkData, buf[:n])
			chunk := Chunk{
				Index:  idx,
				Offset: offset,
				Size:   n,
				Data:   chunkData,
				Digest: DigestBytes(chunkData),
			}
			if err := fn(chunk); err != nil {
				return err
			}
			offset += int64(n)
			idx++
		}
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return fmt.Errorf("%w: %v", ErrCorruptedBlob, err)
		}
	}
	return nil
}

func (c *Chunker) chunkReaderDelimited(r io.Reader, fn func(chunk Chunk) error) error {
	var offset int64
	idx := 0
	var acc []byte
	readBuf := make([]byte, 32*1024)
	if len(readBuf) > c.maxChunkSize {
		readBuf = make([]byte, c.maxChunkSize)
	}

	for {
		// Process any complete delimited or max-sized segments in accumulator
		for len(acc) > 0 {
			searchLimit := len(acc)
			if searchLimit > c.maxChunkSize {
				searchLimit = c.maxChunkSize
			}

			pos := bytes.Index(acc[:searchLimit], c.delimiter)
			if pos >= 0 {
				end := pos + len(c.delimiter)
				chunkData := make([]byte, end)
				copy(chunkData, acc[:end])
				chunk := Chunk{
					Index:  idx,
					Offset: offset,
					Size:   end,
					Data:   chunkData,
					Digest: DigestBytes(chunkData),
				}
				if err := fn(chunk); err != nil {
					return err
				}
				offset += int64(end)
				idx++
				acc = acc[end:]
			} else if len(acc) >= c.maxChunkSize {
				chunkData := make([]byte, c.maxChunkSize)
				copy(chunkData, acc[:c.maxChunkSize])
				chunk := Chunk{
					Index:  idx,
					Offset: offset,
					Size:   c.maxChunkSize,
					Data:   chunkData,
					Digest: DigestBytes(chunkData),
				}
				if err := fn(chunk); err != nil {
					return err
				}
				offset += int64(c.maxChunkSize)
				idx++
				acc = acc[c.maxChunkSize:]
			} else {
				// Need more bytes from reader
				break
			}
		}

		n, rErr := r.Read(readBuf)
		if n > 0 {
			acc = append(acc, readBuf[:n]...)
		}
		if rErr == io.EOF {
			// Flush any remaining bytes in accumulator
			for len(acc) > 0 {
				chunkLen := len(acc)
				if chunkLen > c.maxChunkSize {
					chunkLen = c.maxChunkSize
				}
				pos := bytes.Index(acc[:chunkLen], c.delimiter)
				if pos >= 0 {
					chunkLen = pos + len(c.delimiter)
				}
				chunkData := make([]byte, chunkLen)
				copy(chunkData, acc[:chunkLen])
				chunk := Chunk{
					Index:  idx,
					Offset: offset,
					Size:   chunkLen,
					Data:   chunkData,
					Digest: DigestBytes(chunkData),
				}
				if err := fn(chunk); err != nil {
					return err
				}
				offset += int64(chunkLen)
				idx++
				acc = acc[chunkLen:]
			}
			break
		}
		if rErr != nil {
			return fmt.Errorf("%w: %v", ErrCorruptedBlob, rErr)
		}
	}

	return nil
}
