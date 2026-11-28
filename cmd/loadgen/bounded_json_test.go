package main

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestBoundedJSONChunks(t *testing.T) {
	recs := make([]any, 50)
	for i := range recs {
		recs[i] = map[string]int{"id": i}
	}
	chunks, err := BoundedJSONChunks(recs, 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected the bound to split 50 records, got %d chunk(s)", len(chunks))
	}
	next := 0
	for i, c := range chunks {
		if len(c) > 64 {
			t.Fatalf("chunk %d is %d bytes, over the 64 bound", i, len(c))
		}
		var got []map[string]int
		if err := json.Unmarshal(c, &got); err != nil {
			t.Fatalf("chunk %d is not a JSON array: %v", i, err)
		}
		for _, r := range got {
			if r["id"] != next {
				t.Fatalf("order broken: got id %d want %d", r["id"], next)
			}
			next++
		}
	}
	if next != len(recs) {
		t.Fatalf("round trip lost records: %d of %d", next, len(recs))
	}

	big, err := BoundedJSONChunks([]any{map[string]string{"k": "0123456789012345678901234567890123456789"}, 1}, 16)
	if err != nil || len(big) != 2 {
		t.Fatalf("oversized record must take its own chunk: chunks=%d err=%v", len(big), err)
	}
	if empty, err := BoundedJSONChunks(nil, 0); err != nil || len(empty) != 0 {
		t.Fatalf("no records must yield no chunks: %v %v", empty, err)
	}
	if _, err := BoundedJSONChunks(recs, -1); !errors.Is(err, errJSONChunkBound) {
		t.Fatalf("negative bound err = %v", err)
	}
}
