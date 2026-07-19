package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// usagelog_gitop_donecondition_test.go is the consolidated closure witness for
// #4225 ("measure commit/sweep/sync latency by terminal outcome"). The recorder
// (runObservedGitOperation/recordUsage) and the outcome-split fold
// (usagelog.FoldRows.ByOperationOutcome) are each exercised in isolation in
// usagelog_record_test.go; this test binds the issue's Done condition into the
// single deterministic artifact its Closure binding names, and closes the one
// rung the isolated tests leave uncovered — that a git-op row carrying SECRET
// argv persists no raw argv anywhere on disk.
//
// It asserts, end to end over the authoritative recorder + fold:
//   - an injected clock records a 37ms `sync push` exit-3 (refused) row;
//   - the fold keeps the refused push p50 (37ms) separate from the successful
//     push p50 (900ms), so a fast refusal cannot flatter push velocity; and
//   - the persisted rows leak no raw argv — a secret repo path never becomes an
//     operation label, never lands in Row.Args, and never appears anywhere in
//     the serialized journal; only the salted ArgsDigest commits to the args.
func TestUsageGitOpsDoneConditionWitness(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	t.Setenv("FAK_USAGE_LOG_PATH", path)
	t.Setenv("FAK_USAGE_LOG", "")

	const secret = "secret/customer-token.txt"
	base := time.Unix(1_700_000_000, 0)
	oldNow := recordUsageNow
	t.Cleanup(func() { recordUsageNow = oldNow })

	// A refused push that exits fast (37ms) carrying secret-bearing argv...
	recordUsageNow = func() time.Time { return base.Add(37 * time.Millisecond) }
	refusedArgv := []string{"push", "--repo", secret}
	if code := runObservedGitOperation(base, gitOperationName("sync", refusedArgv), refusedArgv, func() int {
		return syncExitRefused
	}); code != syncExitRefused {
		t.Fatalf("refused push code = %d, want %d", code, syncExitRefused)
	}

	// ...and a successful push that takes far longer (900ms).
	recordUsageNow = func() time.Time { return base.Add(900 * time.Millisecond) }
	okArgv := []string{"push"}
	if code := runObservedGitOperation(base, gitOperationName("sync", okArgv), okArgv, func() int {
		return 0
	}); code != 0 {
		t.Fatalf("success push code = %d, want 0", code)
	}

	rows, err := usagelog.ReadRows(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2: %+v", len(rows), rows)
	}

	// The refused row: 37ms, exit-3, composite `sync push` operation label — the
	// literal row the issue's Done condition names.
	if refused := rows[0]; refused.Verb != "sync push" || refused.ExitCode != syncExitRefused || refused.DurationMS != 37 {
		t.Fatalf("refused row = %+v, want 37ms exit-3 sync push", refused)
	}

	// Leaks no raw argv: the secret never lands on disk in any form.
	for _, r := range rows {
		if r.Verb != "sync push" {
			t.Fatalf("operation label = %q, want closed `sync push` (raw argv must never become a label)", r.Verb)
		}
		if len(r.Args) != 0 {
			t.Fatalf("Row.Args = %v, want empty (raw argv must never be persisted)", r.Args)
		}
		if r.ArgsDigest == "" {
			t.Fatal("ArgsDigest empty, want a salted commitment to the args")
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), secret) {
		t.Fatalf("serialized usage journal leaked the raw secret %q:\n%s", secret, raw)
	}

	// The fold keeps the refused push p50 separate from the successful push p50.
	stats := usagelog.FoldRows(rows, usagelog.FoldOptions{}).ByOperationOutcome
	got := map[usagelog.TerminalOutcome]int64{}
	for _, s := range stats {
		if s.Operation != "sync push" {
			t.Fatalf("operation = %q, want sync push", s.Operation)
		}
		got[s.Outcome] = s.P50MS
	}
	if got[usagelog.OutcomeRefused] != 37 || got[usagelog.OutcomeSuccess] != 900 {
		t.Fatalf("fold p50 by outcome = %v, want refused=37 success=900 kept separate", got)
	}
}
