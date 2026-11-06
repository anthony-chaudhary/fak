package sink

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// diverseRecords returns seven records exercising the full v1 field
// surface: every optional field present, every optional field absent,
// and each closed classification.
func diverseRecords() []billing.UsageRecord {
	t0 := time.Date(2026, 11, 4, 12, 0, 0, 0, time.UTC)
	t1 := t0.Add(3 * time.Second)
	return []billing.UsageRecord{
		{ // fully populated
			SchemaVersion:    billing.SchemaVersionV1,
			EventID:          "full-1",
			RequestID:        "req-1",
			TenantID:         "tenant-1",
			KeyID:            "key-1",
			Model:            "model-a",
			ModelRevision:    "rev-1",
			Backend:          "backend-x",
			Route:            "route-y",
			RequestedAt:      t0,
			CompletedAt:      &t1,
			PromptTokens:     ptr(100),
			CompletionTokens: ptr(50),
			CachedTokens:     ptr(25),
			Classification:   billing.UsageRecorded,
			PriceBookVersion: "pbv-1",
			WitnessSHA:       "witness-1",
		},
		{ // minimal required fields only
			SchemaVersion:  billing.SchemaVersionV1,
			EventID:        "minimal-2",
			Model:          "model-a",
			RequestedAt:    t0,
			Classification: billing.UsageMissing,
		},
		{ // known zero with explicit zeros
			SchemaVersion:    billing.SchemaVersionV1,
			EventID:          "zero-3",
			Model:            "model-b",
			RequestedAt:      t0,
			PromptTokens:     ptr(0),
			CompletionTokens: ptr(0),
			Classification:   billing.UsageZeroKnown,
		},
		{ // not applicable
			SchemaVersion:  billing.SchemaVersionV1,
			EventID:        "na-4",
			Model:          "model-b",
			RequestedAt:    t0,
			Classification: billing.UsageNotApplicable,
		},
		{ // recorded, completion-only
			SchemaVersion:    billing.SchemaVersionV1,
			EventID:          "comp-5",
			RequestID:        "req-5",
			Model:            "model-c",
			RequestedAt:      t0,
			CompletedAt:      &t1,
			CompletionTokens: ptr(7),
			Classification:   billing.UsageRecorded,
		},
		{ // missing usage with identity fields
			SchemaVersion:  billing.SchemaVersionV1,
			EventID:        "miss-6",
			TenantID:       "tenant-2",
			KeyID:          "key-2",
			Model:          "model-c",
			RequestedAt:    t0,
			Backend:        "backend-z",
			Classification: billing.UsageMissing,
		},
		{ // cached subset of prompt
			SchemaVersion:    billing.SchemaVersionV1,
			EventID:          "cache-7",
			Model:            "model-d",
			RequestedAt:      t0,
			PromptTokens:     ptr(64),
			CachedTokens:     ptr(32),
			CompletionTokens: ptr(16),
			Classification:   billing.UsageRecorded,
		},
	}
}

// sameRecords compares two record slices field by field, using time
// equality for timestamps.
func sameRecords(t *testing.T, got, want []billing.UsageRecord) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d records, want %d", len(got), len(want))
	}
	for i := range want {
		g, w := got[i], want[i]
		if g.SchemaVersion != w.SchemaVersion ||
			g.EventID != w.EventID ||
			g.RequestID != w.RequestID ||
			g.TenantID != w.TenantID ||
			g.KeyID != w.KeyID ||
			g.Model != w.Model ||
			g.ModelRevision != w.ModelRevision ||
			g.Backend != w.Backend ||
			g.Route != w.Route ||
			!g.RequestedAt.Equal(w.RequestedAt) ||
			g.Classification != w.Classification ||
			g.PriceBookVersion != w.PriceBookVersion ||
			g.WitnessSHA != w.WitnessSHA {
			t.Fatalf("record %d mismatch:\n got %+v\nwant %+v", i, g, w)
		}
		samePtr(t, g.CompletedAt, w.CompletedAt, i, "CompletedAt")
		samePtr(t, g.PromptTokens, w.PromptTokens, i, "PromptTokens")
		samePtr(t, g.CompletionTokens, w.CompletionTokens, i, "CompletionTokens")
		samePtr(t, g.CachedTokens, w.CachedTokens, i, "CachedTokens")
	}
}

func samePtr[T comparable](t *testing.T, got, want *T, idx int, name string) {
	t.Helper()
	if (got == nil) != (want == nil) {
		t.Fatalf("record %d %s: got %v, want %v (nil-ness differs)", idx, name, got, want)
	}
	if got != nil && *got != *want {
		t.Fatalf("record %d %s: got %v, want %v", idx, name, *got, *want)
	}
}

// TestFileRoundTrip writes seven diverse records through a Ring over a
// File backend, closes, and reads them back strictly: values equal,
// one line per record, deterministic field order.
func TestFileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "records.jsonl")

	f, err := NewFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewRing(f, RingOptions{Capacity: 8, BatchSize: 3, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	want := diverseRecords()
	for _, rec := range want {
		if err := r.Write(rec); err != nil {
			t.Fatalf("write %s: %v", rec.EventID, err)
		}
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got, err := ReadJSONL(path)
	if err != nil {
		t.Fatalf("ReadJSONL: %v", err)
	}
	sameRecords(t, got, want)

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSuffix(raw, []byte("\n")), []byte("\n"))
	if len(lines) != len(want) {
		t.Fatalf("file holds %d lines, want %d", len(lines), len(want))
	}
	for i, line := range lines {
		if !bytes.HasPrefix(line, []byte(`{"schema_version":`)) {
			t.Fatalf("line %d does not start with schema_version (field order not deterministic): %s", i+1, line)
		}
	}
}

// TestFileAppendAcrossRestart simulates a process restart: a second
// producer opens the same path and appends after the first producer's
// content, without overwriting it.
func TestFileAppendAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "records.jsonl")

	first := []string{"a", "b", "c"}
	second := []string{"d", "e"}

	f, err := NewFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewRing(f, RingOptions{Capacity: 4, BatchSize: 2, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range first {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if err := r.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}

	// Restart: a fresh sink over the same path.
	f2, err := NewFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := NewRing(f2, RingOptions{Capacity: 4, BatchSize: 2, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range second {
		if err := r2.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if err := r2.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}

	got, err := ReadJSONL(path)
	if err != nil {
		t.Fatalf("ReadJSONL: %v", err)
	}
	expectIDs(t, idsOf(got), append(append([]string{}, first...), second...))
}

func TestReadJSONLStrict(t *testing.T) {
	dir := t.TempDir()

	validLine := func(id string) []byte {
		b, err := json.Marshal(rec(id))
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	write := func(name string, data []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, data, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	// Zero-byte file: no records, no error.
	p := write("empty.jsonl", nil)
	recs, err := ReadJSONL(p)
	if err != nil {
		t.Fatalf("empty file: %v", err)
	}
	if len(recs) != 0 {
		t.Fatalf("empty file: got %d records, want 0", len(recs))
	}

	// Two valid lines decode to both records.
	two := append(append(append([]byte{}, validLine("x")...), '\n'), validLine("y")...)
	two = append(two, '\n')
	p = write("two.jsonl", two)
	recs, err = ReadJSONL(p)
	if err != nil {
		t.Fatalf("two lines: %v", err)
	}
	expectIDs(t, idsOf(recs), []string{"x", "y"})

	// Unknown field: refused by strict decoding.
	line := validLine("rogue")
	rogue := append(append([]byte{}, line[:len(line)-1]...), []byte(`,"rogue_field":1}`)...)
	p = write("rogue.jsonl", append(rogue, '\n'))
	if _, err := ReadJSONL(p); !errors.Is(err, billing.ErrMalformedPayload) {
		t.Fatalf("unknown field: got %v, want ErrMalformedPayload", err)
	}

	// Two objects on one line: trailing data after the record.
	concat := append(append(append([]byte{}, validLine("m")...), validLine("n")...), '\n')
	p = write("concat.jsonl", concat)
	if _, err := ReadJSONL(p); !errors.Is(err, billing.ErrMalformedPayload) {
		t.Fatalf("two objects on one line: got %v, want ErrMalformedPayload", err)
	}

	// Interior empty line.
	blank := append(append(append([]byte{}, validLine("a")...), '\n', '\n'), validLine("b")...)
	blank = append(blank, '\n')
	p = write("blank.jsonl", blank)
	if _, err := ReadJSONL(p); !errors.Is(err, ErrMalformedLine) {
		t.Fatalf("interior empty line: got %v, want ErrMalformedLine", err)
	}

	// Missing final newline.
	p = write("notrail.jsonl", validLine("solo"))
	if _, err := ReadJSONL(p); !errors.Is(err, ErrMalformedLine) {
		t.Fatalf("missing final newline: got %v, want ErrMalformedLine", err)
	}

	// Garbage line.
	p = write("garbage.jsonl", []byte("not json at all\n"))
	if _, err := ReadJSONL(p); !errors.Is(err, billing.ErrMalformedPayload) {
		t.Fatalf("garbage line: got %v, want ErrMalformedPayload", err)
	}
}

// TestFileAppendAfterCloseRefused checks the backend's closed contract
// directly.
func TestFileAppendAfterCloseRefused(t *testing.T) {
	dir := t.TempDir()
	f, err := NewFile(filepath.Join(dir, "records.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := f.Append([]billing.UsageRecord{rec("a")}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Append after Close: got %v, want ErrClosed", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("second Close: got %v, want nil (idempotent)", err)
	}
}
