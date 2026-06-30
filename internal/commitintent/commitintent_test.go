package commitintent

import (
	"errors"
	"reflect"
	"testing"
	"time"
)

const (
	baseA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	baseB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

func TestSubmitRecordRoundTrip(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 123, time.UTC)
	intent := Intent{
		ID:      "issue-1788-a",
		BaseSHA: baseA,
		Paths:   []string{`internal\commitintent\commitintent.go`, "./internal/commitintent/doc.go"},
		Subject: "feat(commitintent): add submit queue (fak commitintent)",
		Metadata: StampMetadata{
			Issue:      1788,
			Generation: "gen/now",
			Source:     "worker",
			Requester:  "orchestrator",
			Labels:     []string{"commit-lane", "first-slice"},
		},
	}
	queue, rec, err := Submit(NewQueue(), now, intent)
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if queue.NextSequence != 2 || len(queue.Records) != 1 {
		t.Fatalf("queue after submit = %+v", queue)
	}
	if rec.Intent.PathDigest == "" {
		t.Fatal("submit must stamp a path digest")
	}
	b, err := MarshalRecord(rec)
	if err != nil {
		t.Fatalf("MarshalRecord: %v", err)
	}
	got, err := ParseRecord(b)
	if err != nil {
		t.Fatalf("ParseRecord: %v", err)
	}
	if !reflect.DeepEqual(got, rec) {
		t.Fatalf("round trip mismatch\n got: %#v\nwant: %#v", got, rec)
	}
}

func TestPathDigestStable(t *testing.T) {
	a := PathDigest([]string{"internal/commitintent/doc.go", `internal\commitintent\commitintent.go`, "./internal/commitintent/doc.go"})
	b := PathDigest([]string{`.\internal\commitintent\commitintent.go`, "internal/commitintent/doc.go"})
	if a == "" {
		t.Fatal("PathDigest returned empty digest")
	}
	if a != b {
		t.Fatalf("digest should be order, duplicate, and separator stable: %s != %s", a, b)
	}
}

func TestValidationMissingAndStaleFields(t *testing.T) {
	valid := Intent{
		ID:      "issue-1788-valid",
		BaseSHA: baseA,
		Paths:   []string{"internal/commitintent/commitintent.go"},
		Subject: "feat(commitintent): add submit queue (fak commitintent)",
	}
	tests := []struct {
		name string
		mut  func(Intent) Intent
		want error
	}{
		{
			name: "missing id",
			mut: func(in Intent) Intent {
				in.ID = ""
				return in
			},
			want: ErrMissingField,
		},
		{
			name: "missing base",
			mut: func(in Intent) Intent {
				in.BaseSHA = ""
				return in
			},
			want: ErrMissingField,
		},
		{
			name: "missing paths",
			mut: func(in Intent) Intent {
				in.Paths = nil
				return in
			},
			want: ErrMissingField,
		},
		{
			name: "missing subject",
			mut: func(in Intent) Intent {
				in.Subject = ""
				return in
			},
			want: ErrMissingField,
		},
		{
			name: "missing stamp",
			mut: func(in Intent) Intent {
				in.Subject = "feat(commitintent): add submit queue"
				return in
			},
			want: ErrMissingField,
		},
		{
			name: "stale base",
			mut:  func(in Intent) Intent { return in },
			want: ErrStaleBase,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			intent := tt.mut(valid)
			var err error
			if tt.want == ErrStaleBase {
				err = ValidateIntentForBase(intent, baseB)
			} else {
				err = ValidateIntent(intent)
			}
			if !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestDrainDeterministicOrdering(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	mk := func(id string, seq int64, base string, offset time.Duration) SubmitRecord {
		intent := Intent{
			ID:      id,
			BaseSHA: base,
			Paths:   []string{"internal/commitintent/" + id + ".go"},
			Subject: "feat(commitintent): add submit queue (fak commitintent)",
		}
		intent, err := NormalizeIntent(intent)
		if err != nil {
			t.Fatalf("NormalizeIntent(%s): %v", id, err)
		}
		return SubmitRecord{
			Schema:      RecordSchema,
			Sequence:    seq,
			SubmittedAt: now.Add(offset),
			State:       StatePending,
			Intent:      intent,
		}
	}
	records := []SubmitRecord{
		mk("c", 3, baseA, 0),
		mk("stale", 2, baseB, 0),
		mk("a", 1, baseA, time.Second),
		mk("b", 1, baseA, 0),
	}
	plan := Drain(records, baseA, 0)
	got := ids(plan.Ready)
	want := []string{"b", "a", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ready ids = %v, want %v", got, want)
	}
	if gotStale := ids(plan.Stale); !reflect.DeepEqual(gotStale, []string{"stale"}) {
		t.Fatalf("stale ids = %v", gotStale)
	}
	if len(plan.Invalid) != 0 {
		t.Fatalf("invalid = %+v", plan.Invalid)
	}
}

func ids(records []SubmitRecord) []string {
	out := make([]string, 0, len(records))
	for _, rec := range records {
		out = append(out, rec.Intent.ID)
	}
	return out
}
