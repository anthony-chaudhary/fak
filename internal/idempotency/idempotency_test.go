package idempotency

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fixedClock returns a clock whose now advances only when the test bumps it, so
// window expiry is deterministic.
type fixedClock struct{ t time.Time }

func (c *fixedClock) now() time.Time      { return c.t }
func (c *fixedClock) add(d time.Duration) { c.t = c.t.Add(d) }

// TestRetryAfterHangReplaysWithoutDuplicate is the #2093 acceptance witness: a
// retried keyed op (issue-create) AFTER a simulated hang returns the original
// result without applying the op a second time, and a genuinely new op with a
// fresh key proceeds. `applied` is the durable side effect the op would double if
// dedup failed (the "issues filed" ledger).
func TestRetryAfterHangReplaysWithoutDuplicate(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	var applied []string // every real issue-create appends here

	issueCreate := func(title string) func() (string, error) {
		return func() (string, error) {
			applied = append(applied, title)
			return fmt.Sprintf("created issue #%d: %s", len(applied), title), nil
		}
	}

	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	// Attempt 1: the op lands and records. (In the hang case its signal is then
	// lost, so the caller never sees this result.)
	key := Key("issue-create", "epic-2093-file-child")
	res1, replayed1, err := store.Do(key, "issue-create", issueCreate("idempotency keys"))
	if err != nil {
		t.Fatalf("attempt 1: %v", err)
	}
	if replayed1 {
		t.Fatal("attempt 1 must apply, not replay")
	}

	// Retry after the hang, in a FRESH store (a new process re-reading the ledger).
	store2, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	res2, replayed2, err := store2.Do(key, "issue-create", issueCreate("idempotency keys"))
	if err != nil {
		t.Fatalf("retry: %v", err)
	}
	if !replayed2 {
		t.Error("retry after hang must REPLAY the recorded result, not re-apply")
	}
	if res2 != res1 {
		t.Errorf("retry returned %q, want the original result %q", res2, res1)
	}
	if len(applied) != 1 {
		t.Fatalf("op double-applied: %d issues filed, want exactly 1 (%v)", len(applied), applied)
	}

	// A genuinely new op with a fresh key proceeds.
	freshKey := Key("issue-create", "epic-2093-file-sibling")
	res3, replayed3, err := store2.Do(freshKey, "issue-create", issueCreate("timeout partial-state child"))
	if err != nil {
		t.Fatalf("fresh op: %v", err)
	}
	if replayed3 {
		t.Error("a fresh key must proceed, not replay")
	}
	if len(applied) != 2 {
		t.Fatalf("fresh op did not apply: %d issues filed, want 2 (%v)", len(applied), applied)
	}
	if res3 == res1 {
		t.Errorf("fresh op returned the prior result %q; want a distinct result", res3)
	}
}

// TestWindowExpiryLetsReusedKeyProceed proves dedup is time-bounded: once the
// window lapses, the same key is treated as a genuinely new op and applies again.
func TestWindowExpiryLetsReusedKeyProceed(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	clk := &fixedClock{t: time.Unix(1_700_000_000, 0)}
	window := time.Hour

	store, err := OpenClock(ledger, window, clk.now)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	var calls int
	apply := func() (string, error) { calls++; return fmt.Sprintf("r%d", calls), nil }

	key := Key("push", "tok")
	if _, replayed, err := store.Do(key, "push", apply); err != nil || replayed {
		t.Fatalf("first apply: replayed=%v err=%v", replayed, err)
	}
	// Within the window: replays.
	clk.add(window - time.Minute)
	if _, replayed, err := store.Do(key, "push", apply); err != nil || !replayed {
		t.Fatalf("in-window call: replayed=%v err=%v, want replayed", replayed, err)
	}
	if calls != 1 {
		t.Fatalf("in-window call re-applied: calls=%d, want 1", calls)
	}
	// Past the window: proceeds again.
	clk.add(2 * window)
	if _, replayed, err := store.Do(key, "push", apply); err != nil || replayed {
		t.Fatalf("post-window call: replayed=%v err=%v, want a fresh apply", replayed, err)
	}
	if calls != 2 {
		t.Fatalf("post-window call did not re-apply: calls=%d, want 2", calls)
	}
}

func TestMarkerThenErrorBlocksRetryUntilProvenApplied(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	key := Key("issue-create", "tok")
	boom := errors.New("hang")
	var markers int
	if _, _, err := store.Do(key, "issue-create", func() (string, error) {
		markers++
		return "", boom
	}); !errors.Is(err, ErrUnknownApplied) || !errors.Is(err, boom) {
		t.Fatalf("Do error = %v, want UNKNOWN_APPLIED wrapping apply error", err)
	}
	rec, ok, err := store.Status(key)
	if err != nil || !ok || rec.State != StateUnknownApplied {
		t.Fatalf("Status = (%+v, %v, %v), want UNKNOWN_APPLIED", rec, ok, err)
	}
	if _, _, err := store.Do(key, "issue-create", func() (string, error) {
		markers++
		return "duplicate", nil
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("retry error = %v, want UNKNOWN_APPLIED", err)
	}
	if markers != 1 {
		t.Fatalf("ambiguous retry applied %d times, want 1", markers)
	}
	if _, err := store.Resolve(key, func(Record) (Resolution, string, error) {
		return ResolutionApplied, "created issue #1", nil
	}); err != nil {
		t.Fatalf("Resolve applied: %v", err)
	}
	res, replayed, err := store.Do(key, "issue-create", func() (string, error) {
		markers++
		return "duplicate", nil
	})
	if err != nil || !replayed || res != "created issue #1" {
		t.Fatalf("resolved replay = (%q, %v, %v)", res, replayed, err)
	}
	if markers != 1 {
		t.Fatalf("proven-applied key re-executed: markers=%d", markers)
	}
}

func TestPendingIntentExistsBeforeApplyAndSurvivesReopen(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	key := Key("push", "intent")
	if _, _, err := store.Do(key, "push", func() (string, error) {
		b, readErr := os.ReadFile(ledger)
		if readErr != nil {
			t.Fatalf("read intent during apply: %v", readErr)
		}
		if !strings.Contains(string(b), `"state":"PENDING"`) {
			t.Fatalf("ledger before apply lacks PENDING intent: %s", b)
		}
		return "ok", nil
	}); err != nil {
		t.Fatalf("Do: %v", err)
	}

	crashKey := Key("push", "crash-after-intent")
	if err := store.appendLocked(Record{
		Key: crashKey, Op: "push", State: StatePending, UpdatedAt: time.Now().UnixNano(),
	}); err != nil {
		t.Fatalf("append crash intent: %v", err)
	}
	reopened, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	calls := 0
	if _, _, err := reopened.Do(crashKey, "push", func() (string, error) {
		calls++
		return "duplicate", nil
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("crash retry error = %v, want UNKNOWN_APPLIED", err)
	}
	if calls != 0 {
		t.Fatalf("crash-after-intent retry called apply %d times", calls)
	}
}

func TestApplySuccessThenLedgerErrorStaysUnknown(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	realAppend := store.appendLocked
	ledgerErr := errors.New("success record fsync failed")
	store.appendRecord = func(rec Record) error {
		if rec.State == StateApplied {
			return ledgerErr
		}
		return realAppend(rec)
	}
	key := Key("provision", "tok")
	calls := 0
	if _, _, err := store.Do(key, "provision", func() (string, error) {
		calls++
		return "vm-1", nil
	}); !errors.Is(err, ErrUnknownApplied) || !errors.Is(err, ledgerErr) {
		t.Fatalf("Do error = %v, want UNKNOWN_APPLIED wrapping ledger failure", err)
	}
	reopened, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if _, _, err := reopened.Do(key, "provision", func() (string, error) {
		calls++
		return "vm-2", nil
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("retry error = %v, want UNKNOWN_APPLIED", err)
	}
	if calls != 1 {
		t.Fatalf("post-apply ledger failure re-executed: calls=%d", calls)
	}
}

func TestProvenAbsentAllowsOneFreshApply(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	key := Key("append", "tok")
	calls := 0
	if _, _, err := store.Do(key, "append", func() (string, error) {
		calls++
		return "", errors.New("transport lost")
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("ambiguous apply: %v", err)
	}
	if _, err := store.Resolve(key, func(Record) (Resolution, string, error) {
		return ResolutionAbsent, "", nil
	}); err != nil {
		t.Fatalf("Resolve absent: %v", err)
	}
	if rec, ok, err := store.Status(key); err != nil || !ok || rec.State != StateProvenAbsent {
		t.Fatalf("resolved status = (%+v, %v, %v), want PROVEN_ABSENT", rec, ok, err)
	}
	res, replayed, err := store.Do(key, "append", func() (string, error) {
		calls++
		return "appended", nil
	})
	if err != nil || replayed || res != "appended" || calls != 2 {
		t.Fatalf("retry after proven absent = (%q, %v, %v), calls=%d", res, replayed, err, calls)
	}
}

func TestStillUnknownResolutionRetainsBlock(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	key := Key("provision", "readback-unknown")
	calls := 0
	if _, _, err := store.Do(key, "provision", func() (string, error) {
		calls++
		return "", errors.New("response lost")
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("ambiguous apply: %v", err)
	}
	if _, err := store.Resolve(key, func(Record) (Resolution, string, error) {
		return ResolutionUnknown, "", nil
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("unknown read-back error = %v, want UNKNOWN_APPLIED", err)
	}
	if _, _, err := store.Do(key, "provision", func() (string, error) {
		calls++
		return "duplicate", nil
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("retry after unknown read-back = %v", err)
	}
	if calls != 1 {
		t.Fatalf("unknown read-back allowed reapply: calls=%d", calls)
	}
}

func TestUnknownNeverExpiresIntoReapplication(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	clk := &fixedClock{t: time.Unix(1_700_000_000, 0)}
	store, err := OpenClock(ledger, time.Hour, clk.now)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	key := Key("push", "unknown-expiry")
	calls := 0
	if _, _, err := store.Do(key, "push", func() (string, error) {
		calls++
		return "", errors.New("connection reset")
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("ambiguous apply: %v", err)
	}
	clk.add(10 * time.Hour)
	if _, _, err := store.Do(key, "push", func() (string, error) {
		calls++
		return "duplicate", nil
	}); !errors.Is(err, ErrUnknownApplied) {
		t.Fatalf("expired unknown error = %v", err)
	}
	if calls != 1 {
		t.Fatalf("expired unknown re-applied: calls=%d", calls)
	}
}

func TestConcurrentAmbiguousApplyRunsOnce(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	store1, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open store1: %v", err)
	}
	store2, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open store2: %v", err)
	}
	key := Key("issue-create", "concurrent-unknown")
	var calls int32
	apply := func() (string, error) {
		atomic.AddInt32(&calls, 1)
		return "", errors.New("response lost")
	}
	results := make(chan error, 2)
	go func() { _, _, err := store1.Do(key, "issue-create", apply); results <- err }()
	go func() { _, _, err := store2.Do(key, "issue-create", apply); results <- err }()
	for i := 0; i < 2; i++ {
		if err := <-results; !errors.Is(err, ErrUnknownApplied) {
			t.Fatalf("concurrent Do error = %v, want UNKNOWN_APPLIED", err)
		}
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("concurrent ambiguous apply ran %d times, want 1", got)
	}
}

func TestLegacySuccessRecordStillReplays(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	key := Key("issue-create", "legacy")
	legacy := fmt.Sprintf(`{"key":%q,"op":"issue-create","result":"issue #1","applied_at":%d}`+"\n", key, time.Now().UnixNano())
	if err := os.WriteFile(ledger, []byte(legacy), 0o644); err != nil {
		t.Fatalf("write legacy ledger: %v", err)
	}
	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	calls := 0
	res, replayed, err := store.Do(key, "issue-create", func() (string, error) {
		calls++
		return "duplicate", nil
	})
	if err != nil || !replayed || res != "issue #1" || calls != 0 {
		t.Fatalf("legacy replay = (%q, %v, %v), calls=%d", res, replayed, err, calls)
	}
}

// TestKeyIsStableAndSeparated pins the key derivation: identical (op,token) maps
// to one key; a changed op or token maps elsewhere; and the separator keeps
// ("ab","c") distinct from ("a","bc").
func TestKeyIsStableAndSeparated(t *testing.T) {
	if Key("issue-create", "t") != Key("issue-create", "t") {
		t.Error("Key is not stable for identical inputs")
	}
	if Key("issue-create", "t") == Key("push", "t") {
		t.Error("different ops must not collide")
	}
	if Key("issue-create", "a") == Key("issue-create", "b") {
		t.Error("different tokens must not collide")
	}
	if Key("ab", "c") == Key("a", "bc") {
		t.Error("op/token boundary must be separated")
	}
}

// TestConcurrentApplyRunsOnce is the cross-process concurrency witness: two stores
// that BOTH opened before either applied — each holding an empty open-time snapshot,
// which is exactly the race — collapse a concurrent same-key Do to ONE apply. The
// loser blocks on the ledger lock, re-reads under it, finds the winner's fresh
// record, and replays it.
//
// This is the dgx-bridge double-dispatch shape: two workers pick up the same nonce
// and would both run the expensive op because neither has recorded yet. It is
// distinct from TestRetryAfterHangReplaysWithoutDuplicate, which covers the
// SEQUENTIAL reopen (a fresh Open re-reads the ledger and so never missed).
//
// Two *Store values in one process are a faithful stand-in for two processes here:
// each opens its own lock fd, and flock contends across separate open file
// descriptions on both unix and Windows. Their in-process sync.Mutexes are distinct,
// so only the file lock can serialize them.
func TestConcurrentApplyRunsOnce(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")

	// Both stores open BEFORE either applies: neither's snapshot has the record.
	store1, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open store1: %v", err)
	}
	store2, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open store2: %v", err)
	}

	var calls int32
	started := make(chan struct{}) // closed once the winner is inside apply
	release := make(chan struct{}) // closes to let the winner finish
	var once sync.Once
	apply := func() (string, error) {
		atomic.AddInt32(&calls, 1)
		once.Do(func() { close(started) })
		<-release
		return "created issue #1", nil
	}

	key := Key("issue-create", "concurrent-dispatch")
	type outcome struct {
		res      string
		replayed bool
		err      error
	}
	results := make(chan outcome, 2)
	do := func(s *Store) {
		res, replayed, err := s.Do(key, "issue-create", apply)
		results <- outcome{res, replayed, err}
	}

	go do(store1)
	<-started // store1 now holds the lock, parked inside apply

	go do(store2) // must block acquiring the ledger lock, NOT apply
	// Give the loser time to reach the lock and poll it, so releasing the winner
	// genuinely exercises the blocked path rather than a lucky ordering.
	time.Sleep(10 * lockPoll)
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("apply ran %d times while the lock was held; the second Do must block, not apply", got)
	}
	close(release)

	var applied, replayed int
	for i := 0; i < 2; i++ {
		got := <-results
		if got.err != nil {
			t.Fatalf("Do: %v", got.err)
		}
		if got.res != "created issue #1" {
			t.Errorf("Do returned %q, want the single applied result", got.res)
		}
		if got.replayed {
			replayed++
		} else {
			applied++
		}
	}

	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("op double-applied: apply ran %d times, want exactly 1", got)
	}
	if applied != 1 || replayed != 1 {
		t.Errorf("got %d applied / %d replayed, want exactly 1 each", applied, replayed)
	}
}

func TestBoundIdentityRestart(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	token := "token-abc"
	ident := RequestIdentity{
		Operation:     "issue-create",
		Resource:      "issues/repo-1",
		Principal:     "agent-worker-1",
		PayloadDigest: "sha256:fedcba9876543210",
	}
	key := Key(ident.Operation, token)

	calls := 0
	apply := func() (string, error) {
		calls++
		return fmt.Sprintf("created issue #%d", calls), nil
	}

	// 1. Apply one bound identity via DoBound. Verify applied (not replayed).
	res1, replayed1, err := store.DoBound(key, ident, apply)
	if err != nil {
		t.Fatalf("first apply: %v", err)
	}
	if replayed1 {
		t.Fatal("first apply must not be replayed")
	}
	if res1 != "created issue #1" {
		t.Fatalf("first apply result = %q, want %q", res1, "created issue #1")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1", calls)
	}

	// 2. Close and reopen store in a new Store instance.
	store2, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}

	// 3. Retry with the exact same identity and token: verify replayed (replayed=true), result matches, apply() was NOT called.
	res2, replayed2, err := store2.DoBound(key, ident, apply)
	if err != nil {
		t.Fatalf("replayed retry: %v", err)
	}
	if !replayed2 {
		t.Fatal("retry with exact same identity must be replayed")
	}
	if res2 != res1 {
		t.Fatalf("retry result = %q, want %q", res2, res1)
	}
	if calls != 1 {
		t.Fatalf("apply() called during replay: calls = %d, want 1", calls)
	}

	// 4. Retry with changed payload digest (or changed resource/principal): verify returns ErrIdentityConflict, apply() was NOT called.
	for _, tc := range []struct {
		name  string
		ident RequestIdentity
	}{
		{
			name: "changed payload digest",
			ident: RequestIdentity{
				Operation:     ident.Operation,
				Resource:      ident.Resource,
				Principal:     ident.Principal,
				PayloadDigest: "sha256:0123456789abcdef",
			},
		},
		{
			name: "changed resource",
			ident: RequestIdentity{
				Operation:     ident.Operation,
				Resource:      "issues/repo-2",
				Principal:     ident.Principal,
				PayloadDigest: ident.PayloadDigest,
			},
		},
		{
			name: "changed principal",
			ident: RequestIdentity{
				Operation:     ident.Operation,
				Resource:      ident.Resource,
				Principal:     "agent-worker-2",
				PayloadDigest: ident.PayloadDigest,
			},
		},
		{
			name: "changed operation",
			ident: RequestIdentity{
				Operation:     "issue-update",
				Resource:      ident.Resource,
				Principal:     ident.Principal,
				PayloadDigest: ident.PayloadDigest,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := store2.DoBound(key, tc.ident, apply)
			if !errors.Is(err, ErrIdentityConflict) {
				t.Fatalf("expected ErrIdentityConflict, got %v", err)
			}
			if calls != 1 {
				t.Fatalf("apply() was called during conflict: calls = %d, want 1", calls)
			}
		})
	}

	// 5. In an existing store with an unbound legacy record (written with regular Do): calling DoBound on that key returns ErrLegacyBindingRequired.
	legacyToken := "legacy-token-xyz"
	legacyKey := Key("issue-create", legacyToken)
	legacyRes, legacyReplayed, err := store2.Do(legacyKey, "issue-create", func() (string, error) {
		return "legacy-created", nil
	})
	if err != nil || legacyReplayed || legacyRes != "legacy-created" {
		t.Fatalf("legacy Do failed: res=%q replayed=%v err=%v", legacyRes, legacyReplayed, err)
	}

	_, _, err = store2.DoBound(legacyKey, ident, apply)
	if !errors.Is(err, ErrLegacyBindingRequired) {
		t.Fatalf("DoBound on legacy unbound record: got %v, want ErrLegacyBindingRequired", err)
	}
	if calls != 1 {
		t.Fatalf("apply() was called during legacy binding check: calls = %d, want 1", calls)
	}
}

func TestRequestIdentity(t *testing.T) {
	var zero RequestIdentity
	if !zero.IsZero() {
		t.Error("expected zero RequestIdentity to report IsZero == true")
	}

	ident := RequestIdentity{
		Operation:     "deploy",
		Resource:      "service/auth",
		Principal:     "agent-1",
		PayloadDigest: "hash123",
	}
	if ident.IsZero() {
		t.Error("expected non-zero RequestIdentity to report IsZero == false")
	}

	d1 := ident.Digest()
	d2 := ident.Digest()
	if d1 == "" || d1 != d2 {
		t.Fatalf("Digest not deterministic: %q vs %q", d1, d2)
	}

	identDiff := ident
	identDiff.PayloadDigest = "hash456"
	if identDiff.Digest() == d1 {
		t.Error("different identity must produce different digest")
	}
}
