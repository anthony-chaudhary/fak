package idempotency

import (
	"errors"
	"fmt"
	"path/filepath"
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

// TestApplyErrorIsNotRecorded proves a failed op stays retryable: an apply error
// records nothing, so the next attempt with the same key runs the op instead of
// replaying a phantom "success".
func TestApplyErrorIsNotRecorded(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "idem.jsonl")
	store, err := Open(ledger, DefaultWindow)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	key := Key("issue-create", "tok")
	boom := errors.New("hang")
	if _, _, err := store.Do(key, "issue-create", func() (string, error) { return "", boom }); !errors.Is(err, boom) {
		t.Fatalf("Do surfaced %v, want the apply error", err)
	}
	if _, ok := store.Lookup(key); ok {
		t.Fatal("a failed apply must not record a dedup entry")
	}
	res, replayed, err := store.Do(key, "issue-create", func() (string, error) { return "ok", nil })
	if err != nil || replayed || res != "ok" {
		t.Fatalf("retry after failure: res=%q replayed=%v err=%v, want a fresh apply", res, replayed, err)
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
