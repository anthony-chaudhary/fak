package blobcommon

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

// Known test vectors for SHA-256
const (
	emptySHA256      = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	helloWorldStr    = "hello world"
	helloWorldSHA256 = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
)

func TestDigestBytes(t *testing.T) {
	// Empty input verification
	if got := DigestBytes([]byte{}); got != emptySHA256 {
		t.Fatalf("DigestBytes(empty) = %s, want %s", got, emptySHA256)
	}
	if got := DigestBytes(nil); got != emptySHA256 {
		t.Fatalf("DigestBytes(nil) = %s, want %s", got, emptySHA256)
	}

	// Known vector
	got := DigestBytes([]byte(helloWorldStr))
	if got != helloWorldSHA256 {
		t.Fatalf("DigestBytes(%q) = %s, want %s", helloWorldStr, got, helloWorldSHA256)
	}

	// Determinism invariant
	for i := 0; i < 5; i++ {
		if run := DigestBytes([]byte(helloWorldStr)); run != got {
			t.Fatalf("DigestBytes non-deterministic on run %d: %s != %s", i, run, got)
		}
	}
}

func TestDigestReader(t *testing.T) {
	r := strings.NewReader(helloWorldStr)
	digest, n, err := DigestReader(r)
	if err != nil {
		t.Fatalf("DigestReader failed: %v", err)
	}
	if n != int64(len(helloWorldStr)) {
		t.Fatalf("DigestReader read %d bytes, want %d", n, len(helloWorldStr))
	}
	if digest != helloWorldSHA256 {
		t.Fatalf("DigestReader digest = %s, want %s", digest, helloWorldSHA256)
	}

	// Nil reader guard
	if _, _, err := DigestReader(nil); !errors.Is(err, ErrCorruptedBlob) {
		t.Fatalf("expected ErrCorruptedBlob on nil reader, got %v", err)
	}

	// Faulty reader
	faulty := &errReader{err: errors.New("i/o device failure")}
	if _, _, err := DigestReader(faulty); !errors.Is(err, ErrCorruptedBlob) {
		t.Fatalf("expected wrapped ErrCorruptedBlob on read error, got %v", err)
	}
}

func TestValidateDigest(t *testing.T) {
	tests := []struct {
		name    string
		digest  string
		wantErr bool
	}{
		{"valid canonical", helloWorldSHA256, false},
		{"valid empty digest", emptySHA256, false},
		{"too short", helloWorldSHA256[:63], true},
		{"too long", helloWorldSHA256 + "a", true},
		{"empty string", "", true},
		{"uppercase hex", strings.ToUpper(helloWorldSHA256), true},
		{"non-hex character", helloWorldSHA256[:63] + "g", true},
		{"punctuation character", helloWorldSHA256[:63] + "!", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateDigest(tc.digest)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ValidateDigest(%q) error = %v, wantErr = %v", tc.digest, err, tc.wantErr)
			}
			if tc.wantErr && !errors.Is(err, ErrInvalidDigest) {
				t.Fatalf("expected ErrInvalidDigest, got %v", err)
			}
		})
	}
}

func TestVerifyDigestPrefix(t *testing.T) {
	digest := helloWorldSHA256

	// Matching prefixes
	if !VerifyDigestPrefix(digest, digest[:8]) {
		t.Fatalf("expected prefix %s to match %s", digest[:8], digest)
	}
	if !VerifyDigestPrefix(digest, digest[:16]) {
		t.Fatalf("expected prefix %s to match %s", digest[:16], digest)
	}
	if !VerifyDigestPrefix(digest, digest) {
		t.Fatalf("expected full digest prefix to match")
	}
	// Case-insensitivity in prefix
	if !VerifyDigestPrefix(digest, strings.ToUpper(digest[:8])) {
		t.Fatalf("expected case-insensitive prefix match")
	}

	// Non-matching prefix
	if VerifyDigestPrefix(digest, "00000000") {
		t.Fatalf("expected non-matching prefix to return false")
	}

	// Guards: invalid prefix
	if VerifyDigestPrefix(digest, "") {
		t.Fatalf("empty prefix must return false")
	}
	if VerifyDigestPrefix(digest, digest+"a") {
		t.Fatalf("prefix longer than 64 must return false")
	}
	if VerifyDigestPrefix(digest, "zzzz") {
		t.Fatalf("non-hex prefix must return false")
	}

	// Guards: invalid digest
	if VerifyDigestPrefix("invalid_digest", "b94d") {
		t.Fatalf("invalid digest must return false")
	}
}

func TestVerifyChecksum(t *testing.T) {
	data := []byte(helloWorldStr)

	// Happy path
	if err := VerifyChecksum(data, helloWorldSHA256); err != nil {
		t.Fatalf("VerifyChecksum failed on valid pair: %v", err)
	}
	// Case-insensitive expected
	if err := VerifyChecksum(data, strings.ToUpper(helloWorldSHA256)); err != nil {
		t.Fatalf("VerifyChecksum failed on uppercase expected digest: %v", err)
	}

	// Corrupted data
	corrupted := []byte("hello world!")
	if err := VerifyChecksum(corrupted, helloWorldSHA256); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("expected ErrDigestMismatch on corrupted data, got %v", err)
	}

	// Invalid expected digest
	if err := VerifyChecksum(data, "not-a-digest"); !errors.Is(err, ErrInvalidDigest) {
		t.Fatalf("expected ErrInvalidDigest on bad digest format, got %v", err)
	}
}

func TestVerifyReaderChecksum(t *testing.T) {
	r := strings.NewReader(helloWorldStr)
	n, err := VerifyReaderChecksum(r, helloWorldSHA256)
	if err != nil {
		t.Fatalf("VerifyReaderChecksum failed: %v", err)
	}
	if n != int64(len(helloWorldStr)) {
		t.Fatalf("read %d bytes, want %d", n, len(helloWorldStr))
	}

	// Mismatched digest
	r2 := strings.NewReader(helloWorldStr)
	if _, err := VerifyReaderChecksum(r2, emptySHA256); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("expected ErrDigestMismatch, got %v", err)
	}

	// Malformed digest
	r3 := strings.NewReader(helloWorldStr)
	if _, err := VerifyReaderChecksum(r3, "bad"); !errors.Is(err, ErrInvalidDigest) {
		t.Fatalf("expected ErrInvalidDigest, got %v", err)
	}
}

func TestBlobValidation(t *testing.T) {
	empty := []byte{}
	payload := []byte("sample payload")

	if !IsEmptyBlob(empty) {
		t.Fatalf("expected IsEmptyBlob(empty) == true")
	}
	if !IsEmptyBlob(nil) {
		t.Fatalf("expected IsEmptyBlob(nil) == true")
	}
	if IsEmptyBlob(payload) {
		t.Fatalf("expected IsEmptyBlob(payload) == false")
	}

	// ValidateBlob limits
	limitsDisallowEmpty := BlobLimits{AllowEmpty: false, MinSize: 0, MaxSize: 100}
	if err := ValidateBlob(empty, limitsDisallowEmpty); !errors.Is(err, ErrEmptyBlob) {
		t.Fatalf("expected ErrEmptyBlob, got %v", err)
	}

	limitsAllowEmpty := BlobLimits{AllowEmpty: true, MinSize: 0, MaxSize: 100}
	if err := ValidateBlob(empty, limitsAllowEmpty); err != nil {
		t.Fatalf("expected nil error on allowed empty blob, got %v", err)
	}

	limitsMin := BlobLimits{AllowEmpty: true, MinSize: 20, MaxSize: 100}
	if err := ValidateBlob(payload, limitsMin); !errors.Is(err, ErrBlobTooSmall) {
		t.Fatalf("expected ErrBlobTooSmall, got %v", err)
	}

	limitsMax := BlobLimits{AllowEmpty: false, MinSize: 0, MaxSize: 5}
	if err := ValidateBlob(payload, limitsMax); !errors.Is(err, ErrBlobTooLarge) {
		t.Fatalf("expected ErrBlobTooLarge, got %v", err)
	}

	limitsExact := BlobLimits{AllowEmpty: false, MinSize: int64(len(payload)), MaxSize: int64(len(payload))}
	if err := ValidateBlob(payload, limitsExact); err != nil {
		t.Fatalf("expected valid on exact boundaries, got %v", err)
	}
}

func TestValidateBlobReader(t *testing.T) {
	payload := []byte("reader test payload")

	// AllowEmpty = false on empty stream
	if _, err := ValidateBlobReader(bytes.NewReader(nil), BlobLimits{AllowEmpty: false}); !errors.Is(err, ErrEmptyBlob) {
		t.Fatalf("expected ErrEmptyBlob on empty reader, got %v", err)
	}

	// AllowEmpty = true on empty stream
	n, err := ValidateBlobReader(bytes.NewReader(nil), BlobLimits{AllowEmpty: true})
	if err != nil || n != 0 {
		t.Fatalf("expected (0, nil), got (%d, %v)", n, err)
	}

	// MaxSize exceeded during streaming
	limitsMax := BlobLimits{MaxSize: 10}
	if _, err := ValidateBlobReader(bytes.NewReader(payload), limitsMax); !errors.Is(err, ErrBlobTooLarge) {
		t.Fatalf("expected ErrBlobTooLarge, got %v", err)
	}

	// MinSize not reached
	limitsMin := BlobLimits{MinSize: 100}
	if _, err := ValidateBlobReader(bytes.NewReader(payload), limitsMin); !errors.Is(err, ErrBlobTooSmall) {
		t.Fatalf("expected ErrBlobTooSmall, got %v", err)
	}

	// Nil reader guard
	if _, err := ValidateBlobReader(nil, BlobLimits{}); !errors.Is(err, ErrCorruptedBlob) {
		t.Fatalf("expected ErrCorruptedBlob on nil reader, got %v", err)
	}
}

func TestValidateAndVerifyBlob(t *testing.T) {
	data := []byte(helloWorldStr)
	limits := BlobLimits{MinSize: 1, MaxSize: 50, AllowEmpty: false}

	// Success
	if err := ValidateAndVerifyBlob(data, limits, helloWorldSHA256); err != nil {
		t.Fatalf("ValidateAndVerifyBlob failed: %v", err)
	}

	// Fail on limits
	if err := ValidateAndVerifyBlob(data, BlobLimits{MaxSize: 5}, helloWorldSHA256); !errors.Is(err, ErrBlobTooLarge) {
		t.Fatalf("expected ErrBlobTooLarge, got %v", err)
	}

	// Fail on checksum
	if err := ValidateAndVerifyBlob(data, limits, emptySHA256); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("expected ErrDigestMismatch, got %v", err)
	}
}

func TestValidateAndVerifyReader(t *testing.T) {
	data := []byte(helloWorldStr)
	limits := BlobLimits{MinSize: 1, MaxSize: 50, AllowEmpty: false}

	// Success
	n, err := ValidateAndVerifyReader(bytes.NewReader(data), limits, helloWorldSHA256)
	if err != nil {
		t.Fatalf("ValidateAndVerifyReader failed: %v", err)
	}
	if n != int64(len(data)) {
		t.Fatalf("expected %d bytes, got %d", len(data), n)
	}

	// Fail on limits (too large)
	if _, err := ValidateAndVerifyReader(bytes.NewReader(data), BlobLimits{MaxSize: 5}, helloWorldSHA256); !errors.Is(err, ErrBlobTooLarge) {
		t.Fatalf("expected ErrBlobTooLarge, got %v", err)
	}

	// Fail on limits (empty disallowed)
	if _, err := ValidateAndVerifyReader(bytes.NewReader(nil), BlobLimits{AllowEmpty: false}, emptySHA256); !errors.Is(err, ErrEmptyBlob) {
		t.Fatalf("expected ErrEmptyBlob, got %v", err)
	}

	// Fail on mismatch
	if _, err := ValidateAndVerifyReader(bytes.NewReader(data), limits, emptySHA256); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("expected ErrDigestMismatch, got %v", err)
	}

	// Fail on nil reader
	if _, err := ValidateAndVerifyReader(nil, limits, helloWorldSHA256); !errors.Is(err, ErrCorruptedBlob) {
		t.Fatalf("expected ErrCorruptedBlob on nil reader, got %v", err)
	}
}

type errReader struct {
	err error
}

func (e *errReader) Read(p []byte) (n int, err error) {
	return 0, e.err
}
