package protocol

import (
	"encoding/binary"
	"strings"
	"testing"
)

func TestDecodeMGetBody_OverflowCount(t *testing.T) {
	// Craft a body with count=2^30 but only 8 bytes total â†’ should error, not OOM
	body := make([]byte, 8)
	binary.LittleEndian.PutUint32(body[0:4], 1<<30)
	binary.LittleEndian.PutUint32(body[4:8], 0)

	_, err := DecodeMGetBody(body)
	if err == nil {
		t.Fatal("expected error for overflow count, got nil")
	}
	if !strings.Contains(err.Error(), "exceeds max") && !strings.Contains(err.Error(), "impossible") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestDecodeMSetBody_OverflowCount(t *testing.T) {
	body := make([]byte, 8)
	binary.LittleEndian.PutUint32(body[0:4], 1<<30)

	_, _, err := DecodeMSetBody(body)
	if err == nil {
		t.Fatal("expected error for overflow count, got nil")
	}
}

func TestDecodeMultiValueResponse_OverflowCount(t *testing.T) {
	body := make([]byte, 8)
	binary.LittleEndian.PutUint32(body[0:4], 1<<30)

	_, _, err := DecodeMultiValueResponse(body)
	if err == nil {
		t.Fatal("expected error for overflow count, got nil")
	}
}

func TestDecodeMGetBody_MaxBatchCount(t *testing.T) {
	// count = MaxBatchCount + 1 should be rejected
	body := make([]byte, 4)
	binary.LittleEndian.PutUint32(body[0:4], MaxBatchCount+1)

	_, err := DecodeMGetBody(body)
	if err == nil {
		t.Fatal("expected error for count > MaxBatchCount")
	}
}

func TestDecodeMGetBody_ValidSmall(t *testing.T) {
	// Valid batch with 2 keys should work
	keys := [][]byte{[]byte("key1"), []byte("key2")}
	body := EncodeMGetBody(keys)

	decoded, err := DecodeMGetBody(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(decoded) != 2 {
		t.Fatalf("expected 2 keys, got %d", len(decoded))
	}
	if string(decoded[0]) != "key1" || string(decoded[1]) != "key2" {
		t.Errorf("decoded keys mismatch: %q, %q", decoded[0], decoded[1])
	}
}
