package blob

// env_test.go — proves New() honors the FAK_BLOB_MAX_BYTES env override (a raw byte budget;
// sensible default + escape hatch for memory-tight / high-memory hosts).

import (
	"context"
	"testing"
)

func TestEnvOverridesByteBudget(t *testing.T) {
	t.Setenv("FAK_BLOB_MAX_BYTES", "4096") // 4 KiB resident budget
	ctx := context.Background()
	s := New() // reads FAK_BLOB_MAX_BYTES at construction
	for i := 0; i < 200; i++ {
		s.Put(ctx, distinctPayload(i, 1024)) // each >InlineMax, distinct → CAS-resident
	}
	if _, bytes, _ := s.Resident(); bytes > 4096 {
		t.Fatalf("env budget FAK_BLOB_MAX_BYTES=4096 not honored: resident=%d bytes, want ≤ 4096", bytes)
	}
}
