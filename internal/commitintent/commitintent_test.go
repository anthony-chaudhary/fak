package commitintent

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"sync"
	"sync/atomic"
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
		ID:         "issue-1788-a",
		BaseSHA:    baseA,
		Paths:      []string{`internal\commitintent\commitintent.go`, "./internal/commitintent/doc.go"},
		DiffDigest: "SHA256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		Subject:    "feat(commitintent): add submit queue (fak commitintent)",
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
	if rec.Intent.DiffDigest != "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" {
		t.Fatalf("diff digest was not normalized: %q", rec.Intent.DiffDigest)
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

func TestStoreSubmitPersistsGitignoredQueue(t *testing.T) {
	root := t.TempDir()
	store := Store{
		Dir: DefaultQueueDir(root),
		Now: func() time.Time { return time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC) },
	}
	intent := Intent{
		ID:      "issue-1788-store",
		BaseSHA: baseA,
		Paths:   []string{"internal/commitintent/store.go"},
		Subject: "feat(commitintent): persist submit queue (#1788) (fak commitintent)",
	}
	q, rec, err := store.Submit(intent)
	if err != nil {
		t.Fatalf("Store.Submit: %v", err)
	}
	if q.NextSequence != 2 || rec.Sequence != 1 {
		t.Fatalf("queue sequence = %+v rec=%+v", q, rec)
	}
	if got, want := store.Dir, filepath.Join(root, ".fak", StoreDirName); got != want {
		t.Fatalf("store dir = %q, want %q", got, want)
	}
	if _, err := os.Stat(filepath.Join(store.Dir, StoreQueueFile)); err != nil {
		t.Fatalf("queue file was not persisted: %v", err)
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded.Records) != 1 || loaded.Records[0].Intent.ID != "issue-1788-store" {
		t.Fatalf("loaded queue = %+v", loaded)
	}
}

func TestStoreConcurrentSubmitOrdering(t *testing.T) {
	var tick int64
	store := Store{
		Dir: filepath.Join(t.TempDir(), ".fak", StoreDirName),
		Now: func() time.Time {
			n := atomic.AddInt64(&tick, 1)
			return time.Date(2026, 6, 30, 12, 0, 0, int(n), time.UTC)
		},
	}
	const n = 20
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			suffix := fmt.Sprintf("%02d", i)
			_, _, err := store.Submit(Intent{
				ID:      "issue-1788-concurrent-" + suffix,
				BaseSHA: baseA,
				Paths:   []string{"internal/commitintent/path-" + suffix + ".go"},
				Subject: "feat(commitintent): persist submit queue (#1788) (fak commitintent)",
			})
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent submit: %v", err)
		}
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded.Records) != n {
		t.Fatalf("record count = %d, want %d", len(loaded.Records), n)
	}
	seqs := make([]int, 0, n)
	for _, rec := range loaded.Records {
		seqs = append(seqs, int(rec.Sequence))
	}
	sort.Ints(seqs)
	for i, seq := range seqs {
		if seq != i+1 {
			t.Fatalf("sequences = %v, want contiguous 1..%d", seqs, n)
		}
	}
}

func TestStoreDrainPlansStaleBase(t *testing.T) {
	store := Store{
		Dir: filepath.Join(t.TempDir(), ".fak", StoreDirName),
		Now: func() time.Time { return time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC) },
	}
	for _, tc := range []struct {
		id   string
		base string
	}{
		{"ready", baseA},
		{"stale", baseB},
	} {
		if _, _, err := store.Submit(Intent{
			ID:      tc.id,
			BaseSHA: tc.base,
			Paths:   []string{"internal/commitintent/" + tc.id + ".go"},
			Subject: "feat(commitintent): persist submit queue (#1788) (fak commitintent)",
		}); err != nil {
			t.Fatalf("submit %s: %v", tc.id, err)
		}
	}
	plan, err := store.Drain(baseA, 0)
	if err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if got := ids(plan.Ready); !reflect.DeepEqual(got, []string{"ready"}) {
		t.Fatalf("ready = %v", got)
	}
	if got := ids(plan.Stale); !reflect.DeepEqual(got, []string{"stale"}) {
		t.Fatalf("stale = %v", got)
	}
}

func TestStoreMarkStatesPersistsDoneIntent(t *testing.T) {
	store := Store{
		Dir: filepath.Join(t.TempDir(), ".fak", StoreDirName),
		Now: func() time.Time { return time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC) },
	}
	for _, id := range []string{"done", "pending"} {
		if _, _, err := store.Submit(Intent{
			ID:      id,
			BaseSHA: baseA,
			Paths:   []string{"internal/commitintent/" + id + ".go"},
			Subject: "feat(commitintent): persist submit queue (#1788) (fak commitintent)",
		}); err != nil {
			t.Fatalf("submit %s: %v", id, err)
		}
	}
	if _, err := store.MarkStates(map[string]State{"done": StateDone}); err != nil {
		t.Fatalf("MarkStates: %v", err)
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	states := map[string]State{}
	for _, rec := range loaded.Records {
		states[rec.Intent.ID] = rec.State
	}
	if states["done"] != StateDone || states["pending"] != StatePending {
		t.Fatalf("states = %+v", states)
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
