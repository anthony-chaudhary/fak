package billing_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// validRecord builds a fully-populated v1 record that satisfies every
// closed-contract rule; the rejection table mutates exactly one rule at a
// time from this base.
func validRecord(t *testing.T) billing.UsageRecord {
	t.Helper()
	requested := time.Date(2026, 11, 3, 8, 30, 0, 0, time.UTC)
	completed := requested.Add(42 * time.Millisecond)
	prompt, completion, cached := int64(1200), int64(300), int64(256)
	return billing.UsageRecord{
		SchemaVersion:    billing.SchemaVersionV1,
		EventID:          "evt-01",
		RequestID:        "req-01",
		TenantID:         "tenant-opaque-1",
		KeyID:            "key-opaque-1",
		Model:            "model-a",
		ModelRevision:    "rev-7",
		Backend:          "upstream-x",
		Route:            "route-main",
		RequestedAt:      requested,
		CompletedAt:      &completed,
		PromptTokens:     &prompt,
		CompletionTokens: &completion,
		CachedTokens:     &cached,
		Classification:   billing.UsageRecorded,
		PriceBookVersion: "pbv-stamp-1",
		WitnessSHA:       "witness-hex",
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestValidateAcceptsClosedContract(t *testing.T) {
	base := validRecord(t)
	if err := base.Validate(); err != nil {
		t.Fatalf("fully populated record must validate: %v", err)
	}

	zero := int64(0)
	cases := []struct {
		name string
		rec  billing.UsageRecord
	}{
		{"recorded with all three token fields", base},
		{"recorded with a single reported token field", func() billing.UsageRecord {
			r := base
			r.PromptTokens, r.CachedTokens = nil, nil
			return r
		}()},
		{"missing with no token fields", func() billing.UsageRecord {
			r := base
			r.Classification = billing.UsageMissing
			r.PromptTokens, r.CompletionTokens, r.CachedTokens = nil, nil, nil
			return r
		}()},
		{"zero_known with explicit zero token fields", func() billing.UsageRecord {
			r := base
			r.Classification = billing.UsageZeroKnown
			r.PromptTokens, r.CompletionTokens, r.CachedTokens = &zero, &zero, &zero
			return r
		}()},
		{"zero_known with no token fields", func() billing.UsageRecord {
			r := base
			r.Classification = billing.UsageZeroKnown
			r.PromptTokens, r.CompletionTokens, r.CachedTokens = nil, nil, nil
			return r
		}()},
		{"not_applicable with no token fields", func() billing.UsageRecord {
			r := base
			r.Classification = billing.UsageNotApplicable
			r.PromptTokens, r.CompletionTokens, r.CachedTokens = nil, nil, nil
			return r
		}()},
	}
	for _, tc := range cases {
		if err := tc.rec.Validate(); err != nil {
			t.Errorf("%s: unexpected validation failure: %v", tc.name, err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestValidateRejectsOpenContractViolations(t *testing.T) {
	neg := int64(-1)
	cases := []struct {
		name   string
		mutate func(*billing.UsageRecord)
		want   error
	}{
		{"schema version zero", func(r *billing.UsageRecord) { r.SchemaVersion = 0 }, billing.ErrUnknownSchemaVersion},
		{"schema version two", func(r *billing.UsageRecord) { r.SchemaVersion = 2 }, billing.ErrUnknownSchemaVersion},
		{"missing event id", func(r *billing.UsageRecord) { r.EventID = "" }, billing.ErrMissingEventID},
		{"missing model", func(r *billing.UsageRecord) { r.Model = "" }, billing.ErrMissingModel},
		{"unknown classification", func(r *billing.UsageRecord) { r.Classification = "weird" }, billing.ErrUnknownClassification},
		{"zero requested_at", func(r *billing.UsageRecord) { r.RequestedAt = time.Time{} }, billing.ErrMissingRequestedAt},
		{"recorded without any token field", func(r *billing.UsageRecord) {
			r.PromptTokens, r.CompletionTokens, r.CachedTokens = nil, nil, nil
		}, billing.ErrInconsistentTokens},
		{"missing with a reported token", func(r *billing.UsageRecord) {
			r.Classification = billing.UsageMissing
		}, billing.ErrInconsistentTokens},
		{"zero_known with a nonzero token", func(r *billing.UsageRecord) {
			r.Classification = billing.UsageZeroKnown
		}, billing.ErrInconsistentTokens},
		{"not_applicable with a reported token", func(r *billing.UsageRecord) {
			r.Classification = billing.UsageNotApplicable
		}, billing.ErrInconsistentTokens},
		{"negative prompt tokens", func(r *billing.UsageRecord) {
			r.PromptTokens = &neg
		}, billing.ErrInconsistentTokens},
		{"cached exceeds prompt", func(r *billing.UsageRecord) {
			bigger := int64(2000)
			r.CachedTokens = &bigger
		}, billing.ErrInconsistentTokens},
	}
	for _, tc := range cases {
		rec := validRecord(t)
		tc.mutate(&rec)
		err := rec.Validate()
		if err == nil {
			t.Errorf("%s: expected validation failure, got nil", tc.name)
			continue
		}
		if !errors.Is(err, tc.want) {
			t.Errorf("%s: errors.Is = false for %v, got %v", tc.name, tc.want, err)
		}
		if !errors.Is(err, billing.ErrInvalidRecord) {
			t.Errorf("%s: sentinel must wrap ErrInvalidRecord, got %v", tc.name, err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestValidateNilRecord(t *testing.T) {
	var r *billing.UsageRecord
	if err := r.Validate(); !errors.Is(err, billing.ErrInvalidRecord) {
		t.Fatalf("nil record must fail closed with ErrInvalidRecord, got %v", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestClassificationVocabularyIsClosed(t *testing.T) {
	if len(billing.Classifications) != 4 {
		t.Fatalf("classification vocabulary has %d entries, want 4", len(billing.Classifications))
	}
	want := map[billing.UsageClassification]bool{
		billing.UsageRecorded:      true,
		billing.UsageMissing:       true,
		billing.UsageZeroKnown:     true,
		billing.UsageNotApplicable: true,
	}
	for _, c := range billing.Classifications {
		if !c.Valid() {
			t.Errorf("advertised classification %q does not validate", c)
		}
		if !want[c] {
			t.Errorf("unexpected vocabulary member %q", c)
		}
		delete(want, c)
	}
	for c := range want {
		t.Errorf("vocabulary missing advertised member %q", c)
	}
	if billing.UsageClassification("bogus").Valid() {
		t.Fatal("unknown classification validated")
	}
	// The wire values are part of the closed contract: third parties
	// serialize and compare them.
	wire := map[billing.UsageClassification]string{
		billing.UsageRecorded:      "recorded",
		billing.UsageMissing:       "missing",
		billing.UsageZeroKnown:     "zero_known",
		billing.UsageNotApplicable: "not_applicable",
	}
	for c, s := range wire {
		if string(c) != s {
			t.Errorf("wire value for %v = %q, want %q", c, string(c), s)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestJSONRoundTrip(t *testing.T) {
	full := validRecord(t)
	data, err := json.Marshal(&full)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	back, err := billing.DecodeUsageRecord(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(back, full) {
		t.Fatalf("round trip mismatch:\n got %+v\nwant %+v", back, full)
	}

	// A minimal required-fields-only record round-trips too.
	minimal := billing.UsageRecord{
		SchemaVersion:  billing.SchemaVersionV1,
		EventID:        "evt-min",
		Model:          "model-a",
		RequestedAt:    time.Date(2026, 11, 3, 9, 0, 0, 0, time.UTC),
		Classification: billing.UsageMissing,
	}
	data, err = json.Marshal(&minimal)
	if err != nil {
		t.Fatalf("marshal minimal: %v", err)
	}
	back, err = billing.DecodeUsageRecord(data)
	if err != nil {
		t.Fatalf("decode minimal: %v", err)
	}
	if !reflect.DeepEqual(back, minimal) {
		t.Fatalf("minimal round trip mismatch:\n got %+v\nwant %+v", back, minimal)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestJSONWireKeysArePinned(t *testing.T) {
	rec := validRecord(t)
	data, err := json.Marshal(&rec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, key := range []string{
		`"schema_version":1`,
		`"event_id":"evt-01"`,
		`"classification":"recorded"`,
		`"price_book_version":"pbv-stamp-1"`,
		`"prompt_tokens":1200`,
	} {
		if !bytes.Contains(data, []byte(key)) {
			t.Fatalf("serialized record missing pinned key %q:\n%s", key, data)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDecodeRefusesUndeclaredFields(t *testing.T) {
	// The structural privacy invariant on the wire: bodies and
	// credentials cannot ride along in a payload that decodes into a
	// record, because the decoder refuses anything outside the v1 schema.
	payloads := []string{
		`{"schema_version":1,"event_id":"e","model":"m","requested_at":"2026-11-03T08:30:00Z","classification":"missing","prompt":"body text"}`,
		`{"schema_version":1,"event_id":"e","model":"m","requested_at":"2026-11-03T08:30:00Z","classification":"missing","response_body":"x"}`,
		`{"schema_version":1,"event_id":"e","model":"m","requested_at":"2026-11-03T08:30:00Z","classification":"missing","api_key":"secret-value"}`,
		`{"schema_version":1,"event_id":"e","model":"m","requested_at":"2026-11-03T08:30:00Z","classification":"missing","authorization":"Bearer x"}`,
	}
	for i, p := range payloads {
		_, err := billing.DecodeUsageRecord([]byte(p))
		if !errors.Is(err, billing.ErrMalformedPayload) {
			t.Errorf("payload %d (%s): want ErrMalformedPayload, got %v", i, p, err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDecodeRefusesMalformedInput(t *testing.T) {
	rec := validRecord(t)
	valid, err := json.Marshal(&rec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	cases := []struct {
		name    string
		payload []byte
	}{
		{"truncated JSON", []byte(`{"schema_version":`)},
		{"empty input", []byte("")},
		{"two records in one payload", append(append([]byte{}, valid...), '{')},
		{"trailing junk after record", append(valid, []byte("junk")...)},
	}
	for _, tc := range cases {
		_, err := billing.DecodeUsageRecord(tc.payload)
		if !errors.Is(err, billing.ErrMalformedPayload) {
			t.Errorf("%s: want ErrMalformedPayload, got %v", tc.name, err)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSeenDedupSemantics(t *testing.T) {
	s := billing.NewSeen()
	first := validRecord(t)
	if err := s.Record(first); err != nil {
		t.Fatalf("first record: %v", err)
	}
	if err := s.Record(first); !errors.Is(err, billing.ErrAlreadyRecorded) {
		t.Fatalf("replayed event id must return ErrAlreadyRecorded, got %v", err)
	}

	second := validRecord(t)
	second.EventID = "evt-02"
	if err := s.Record(second); err != nil {
		t.Fatalf("distinct event id: %v", err)
	}
	if s.Len() != 2 {
		t.Fatalf("Len = %d, want 2", s.Len())
	}

	// A zero-value Seen is ready to use and dedups independently.
	var z billing.Seen
	if err := z.Record(second); err != nil {
		t.Fatalf("zero-value Seen first record: %v", err)
	}
	if err := z.Record(second); !errors.Is(err, billing.ErrAlreadyRecorded) {
		t.Fatalf("zero-value Seen replay must return ErrAlreadyRecorded, got %v", err)
	}

	// An empty event ID is rejected, not indexed.
	noID := first
	noID.EventID = ""
	if err := s.Record(noID); !errors.Is(err, billing.ErrMissingEventID) {
		t.Fatalf("empty event id must return ErrMissingEventID, got %v", err)
	}
	if s.Len() != 2 {
		t.Fatalf("empty event id was indexed: Len = %d, want 2", s.Len())
	}

	// A nil Seen fails closed instead of silently disabling dedup.
	var n *billing.Seen
	if err := n.Record(first); !errors.Is(err, billing.ErrInvalidRecord) {
		t.Fatalf("nil Seen must fail closed with ErrInvalidRecord, got %v", err)
	}
	if n.Len() != 0 {
		t.Fatalf("nil Seen Len = %d, want 0", n.Len())
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestSeenConcurrentRecord(t *testing.T) {
	s := billing.NewSeen()
	const goroutines, perGoroutine = 4, 25
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				rec := validRecord(t)
				rec.EventID = fmt.Sprintf("g%d-%d", g, i)
				if err := s.Record(rec); err != nil {
					t.Error(err)
					return
				}
			}
		}(g)
	}
	wg.Wait()
	if s.Len() != goroutines*perGoroutine {
		t.Fatalf("Len = %d, want %d", s.Len(), goroutines*perGoroutine)
	}
}
