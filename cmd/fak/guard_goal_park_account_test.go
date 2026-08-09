package main

// Account-scoped goal park. A `fak guard` park records one ACCOUNT's long
// Retry-After wall on a goal; before this it was lane-scoped and account-blind,
// so ONE account's 1h 429 disabled the whole lane for every account until the
// far-future timestamp elapsed — while the dispatcher kept dispatching into the
// lane and guard killed each child mid-tool_use, no report and no commit. These
// tests pin the two halves that made that possible: the reader must scope the
// wall to the walled account, and the dispatcher must NAME that account in the
// child env so a record can be attributed at all.

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/goalpark"
)

// goalParkTestRoot makes a throwaway module root and chdirs into it, so
// repoRoot() — and therefore goalParkStore() — resolves to a temp .fak/goal-park
// instead of the live one in this checkout.
func goalParkTestRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module goalparktest\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	if got := repoRoot(); got != dir {
		// EvalSymlinks-level differences (macOS /var vs /private/var) would make the
		// store path disagree with the fixture; fail loudly rather than silently
		// touching another directory.
		if resolved, err := filepath.EvalSymlinks(dir); err != nil || got != resolved {
			t.Fatalf("repoRoot()=%q want %q", got, dir)
		}
	}
	return dir
}

// parkGoalForTest writes a live park through the SAME store the guard writes to.
func parkGoalForTest(t *testing.T, goal, lane, account string, until time.Time) {
	t.Helper()
	now := until.Add(-time.Hour)
	rec := goalpark.Record{
		Goal: goal, Lane: lane, Account: account, Reason: "LONG_RETRY_AFTER",
		ParkedAt: now.Unix(), ParkedUntil: until.Unix(),
		Command: []string{"claude", "-p", "--model", "claude-opus-4-8"},
	}
	if err := goalParkStore().Park(rec); err != nil {
		t.Fatalf("park %s/%s: %v", goal, account, err)
	}
}

// (a) A park recorded for account A must not park account B on the same lane.
// Before the fix guardGoalParked ignored rec.Account entirely, so every case
// below reported parked=true and short-circuited rotation.rotateAfterExit.
func TestGuardGoalParkWallsOnlyTheRateLimitedAccount(t *testing.T) {
	goalParkTestRoot(t)
	t.Setenv("DISPATCH_GOAL", "")
	t.Setenv("DISPATCH_LANE", "quality")
	parkGoalForTest(t, "quality", "quality", "seat-a", time.Now().Add(time.Hour))

	for _, tc := range []struct {
		name    string
		account string
		want    bool
	}{
		{"the account that hit the 429 waits", "seat-a", true},
		{"a sibling account keeps the lane", "seat-b", false},
		{"an unattributed worker keeps the lane", "", false},
	} {
		t.Setenv("DISPATCH_ACCOUNT", tc.account)
		rec, parked := guardGoalParked()
		if parked != tc.want {
			t.Errorf("%s: DISPATCH_ACCOUNT=%q parked=%v want %v (rec=%+v)", tc.name, tc.account, parked, tc.want, rec)
		}
	}

	// An unattributed RECORD — every park written before DISPATCH_ACCOUNT was
	// wired carried a blank account — must wall nobody at all.
	parkGoalForTest(t, "quality", "quality", "", time.Now().Add(time.Hour))
	for _, account := range []string{"seat-a", "seat-b", ""} {
		t.Setenv("DISPATCH_ACCOUNT", account)
		if _, parked := guardGoalParked(); parked {
			t.Errorf("a park record with a blank account walled %q", account)
		}
	}
}

// (b) A park record must carry the account that was actually walled. This drives
// the real wiring end to end: dispatchWorkerEnv builds the child env, the child's
// guard derives its park identity from it, and the reader matches that identity.
func TestDispatchedWorkerParkRecordCarriesItsAccount(t *testing.T) {
	root := goalParkTestRoot(t)
	// A stale identity in the tick process must never leak into an unrelated child.
	t.Setenv("DISPATCH_ACCOUNT", "stale-tick-identity")

	env, err := dispatchWorkerEnv("claude", "gateway", root, filepath.Join(root, "runs"),
		dispatchtick.Account{Tag: "seat-a", Dir: filepath.Join(root, "accounts", "a")}, "gateway", "")
	if err != nil {
		t.Fatal(err)
	}
	if env["DISPATCH_ACCOUNT"] != "seat-a" {
		t.Fatalf("child env DISPATCH_ACCOUNT=%q want \"seat-a\" — a blank account is why every live park record was unattributed", env["DISPATCH_ACCOUNT"])
	}

	// Boot the child's guard on exactly that env and let it write its park record.
	t.Setenv("DISPATCH_GOAL", env["DISPATCH_GOAL"])
	t.Setenv("DISPATCH_LANE", env["DISPATCH_LANE"])
	t.Setenv("DISPATCH_ACCOUNT", env["DISPATCH_ACCOUNT"])
	goal, account := parkGoalIdentity()
	if goal != "gateway" || account != "seat-a" {
		t.Fatalf("park identity=(%q,%q) want (\"gateway\",\"seat-a\")", goal, account)
	}
	parkGoalForTest(t, goal, "gateway", account, time.Now().Add(time.Hour))

	rec, err := goalParkStore().Load("gateway")
	if err != nil {
		t.Fatal(err)
	}
	if rec.Account != "seat-a" {
		t.Fatalf("park record account=%q want \"seat-a\"", rec.Account)
	}
	if _, parked := guardGoalParked(); !parked {
		t.Fatal("the walled account was not recognized by its own park record")
	}
	t.Setenv("DISPATCH_ACCOUNT", "seat-b")
	if _, parked := guardGoalParked(); parked {
		t.Fatal("seat-b was walled by seat-a's park")
	}
}

// An account the chooser could not tag is still attributable by its config dir;
// a genuinely anonymous one must clear the variable rather than inherit the
// dispatcher's own identity and mislabel someone else's wall.
func TestDispatchWorkerEnvAccountIdentityFallbacks(t *testing.T) {
	root := t.TempDir()
	t.Setenv("DISPATCH_ACCOUNT", "stale-tick-identity")
	for _, tc := range []struct {
		name    string
		account dispatchtick.Account
		want    string
	}{
		{"tag is the identity of record", dispatchtick.Account{Tag: "seat-a", Dir: filepath.Join(root, "b")}, "seat-a"},
		{"an untagged seat falls back to its config dir", dispatchtick.Account{Dir: filepath.Join(root, "seat-c")}, "seat-c"},
		{"an anonymous seat inherits nothing", dispatchtick.Account{}, ""},
	} {
		env, err := dispatchWorkerEnv("claude", "gateway", root, filepath.Join(root, "runs"), tc.account, "gateway", "")
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if env["DISPATCH_ACCOUNT"] != tc.want {
			t.Errorf("%s: DISPATCH_ACCOUNT=%q want %q", tc.name, env["DISPATCH_ACCOUNT"], tc.want)
		}
	}
}

// (c) A park must become claimable and retire, not sit until its far-future
// timestamp. Before the fix nothing in the product ever called ClaimDue, so
// claimed_at stayed 0 in every record and the promised resume never happened.
func TestGuardGoalParkRetiresWhenTheWaitElapses(t *testing.T) {
	goalParkTestRoot(t)
	t.Setenv("DISPATCH_GOAL", "quality")
	t.Setenv("DISPATCH_LANE", "quality")
	t.Setenv("DISPATCH_ACCOUNT", "seat-a")
	parkGoalForTest(t, "quality", "quality", "seat-a", time.Now().Add(-time.Minute))

	rec, parked := guardGoalParked()
	if parked {
		t.Fatal("a park whose wait already elapsed still walled its account")
	}
	if rec.ClaimedAt == 0 {
		t.Fatalf("a due park was not claimed, so it never retires: %+v", rec)
	}
	if rec.ClaimedBy != "fak-guard/seat-a" {
		t.Fatalf("claim does not name the supervisor that resumed it: %q", rec.ClaimedBy)
	}
	// The claim is durable — a later reader sees the retired park, not a live one.
	stored, err := goalParkStore().Load("quality")
	if err != nil {
		t.Fatal(err)
	}
	if stored.ClaimedAt == 0 || stored.ClaimedBy != "fak-guard/seat-a" {
		t.Fatalf("claim did not survive to disk: %+v", stored)
	}
	if _, parked = guardGoalParked(); parked {
		t.Fatal("a retired park came back")
	}
}
