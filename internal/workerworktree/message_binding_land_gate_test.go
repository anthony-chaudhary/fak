package workerworktree

import (
	"os"
	"path/filepath"
	"testing"
)

// TestMessageBindingLandGate witnesses the MESSAGE_RACE guard (#2196): a land whose
// commit message cites a DIFFERENT issue than the work is for is refused; a matching
// reference clears; and every uncertainty (no intent, no reference, unparseable)
// fails open. The intended issue is threaded (intendedIssue > 0) on the bot path and
// falls back to the durable Intent (intendedIssue == 0) on the public-native path.
func TestMessageBindingLandGate(t *testing.T) {
	// A GitRunner the gate never calls (it only reads the intent + message file),
	// but supplied for signature consistency.
	noopGit := func(root string, args []string) (int, string) { return 0, "" }

	t.Run("mismatch refuses", func(t *testing.T) {
		wt := t.TempDir()
		// The operator's intent is for #4574 (the suite-split work).
		if err := SaveIntent(wt, "base-sha", "feat(quality): split the suite under #4574 (fak quality)", []string{"internal/quality/suite.go"}); err != nil {
			t.Fatalf("SaveIntent: %v", err)
		}
		// But the commit message that will actually be committed cites #5098 — the
		// exact MESSAGE_RACE: a stale base-tip subject bound to the wrong issue.
		msgFile := filepath.Join(t.TempDir(), "msg.txt")
		if err := os.WriteFile(msgFile, []byte("feat(model): bind polymodel.SpecDecode to a live path (#5098) (fak model)\n"), 0o600); err != nil {
			t.Fatalf("write msg: %v", err)
		}
		res, fired := messageBindingLandGate(wt, msgFile, 0, noopGit)
		if !fired {
			t.Fatalf("expected the gate to fire on a mismatched issue reference")
		}
		if res.OK {
			t.Fatalf("expected a refusal, got OK")
		}
		if res.Code != LandResultMessageBindingMismatch {
			t.Fatalf("expected code %q, got %q", LandResultMessageBindingMismatch, res.Code)
		}
		if !res.Preserved {
			t.Fatalf("expected the worktree to be preserved on refusal")
		}
	})

	t.Run("match clears", func(t *testing.T) {
		wt := t.TempDir()
		if err := SaveIntent(wt, "base-sha", "feat(quality): split the suite under #4574 (fak quality)", []string{"internal/quality/suite.go"}); err != nil {
			t.Fatalf("SaveIntent: %v", err)
		}
		msgFile := filepath.Join(t.TempDir(), "msg.txt")
		if err := os.WriteFile(msgFile, []byte("feat(quality): split the suite under #4574 (fak quality)\n"), 0o600); err != nil {
			t.Fatalf("write msg: %v", err)
		}
		if res, fired := messageBindingLandGate(wt, msgFile, 0, noopGit); fired {
			t.Fatalf("expected the gate to clear on a matching issue reference, got %+v", res)
		}
	})

	t.Run("no intent fails open", func(t *testing.T) {
		wt := t.TempDir() // no intent written
		msgFile := filepath.Join(t.TempDir(), "msg.txt")
		if err := os.WriteFile(msgFile, []byte("feat(model): some work (#5098) (fak model)\n"), 0o600); err != nil {
			t.Fatalf("write msg: %v", err)
		}
		if res, fired := messageBindingLandGate(wt, msgFile, 0, noopGit); fired {
			t.Fatalf("expected fail-open with no intent, got %+v", res)
		}
	})

	t.Run("no issue reference fails open", func(t *testing.T) {
		wt := t.TempDir()
		if err := SaveIntent(wt, "base-sha", "feat(quality): split the suite under #4574 (fak quality)", []string{"internal/quality/suite.go"}); err != nil {
			t.Fatalf("SaveIntent: %v", err)
		}
		msgFile := filepath.Join(t.TempDir(), "msg.txt")
		// A subject with no closing reference: nothing to bind.
		if err := os.WriteFile(msgFile, []byte("chore(dispatch): land worker worktree diff\n"), 0o600); err != nil {
			t.Fatalf("write msg: %v", err)
		}
		if res, fired := messageBindingLandGate(wt, msgFile, 0, noopGit); fired {
			t.Fatalf("expected fail-open with no issue reference, got %+v", res)
		}
	})

	t.Run("unreadable message fails open", func(t *testing.T) {
		wt := t.TempDir()
		if err := SaveIntent(wt, "base-sha", "feat(quality): split the suite under #4574 (fak quality)", []string{"internal/quality/suite.go"}); err != nil {
			t.Fatalf("SaveIntent: %v", err)
		}
		if res, fired := messageBindingLandGate(wt, filepath.Join(t.TempDir(), "does-not-exist.txt"), 0, noopGit); fired {
			t.Fatalf("expected fail-open with an unreadable message, got %+v", res)
		}
	})

	// Threaded path (the bot path): the intended issue is supplied directly, no
	// public Intent is written. The gate must bind the final message to it.
	t.Run("threaded mismatch refuses", func(t *testing.T) {
		wt := t.TempDir() // no intent written — the bot path never writes one
		msgFile := filepath.Join(t.TempDir(), "msg.txt")
		// The final message cites #5098 but the work is for #4574.
		if err := os.WriteFile(msgFile, []byte("feat(model): bind polymodel.SpecDecode to a live path (#5098) (fak model)\n"), 0o600); err != nil {
			t.Fatalf("write msg: %v", err)
		}
		res, fired := messageBindingLandGate(wt, msgFile, 4574, noopGit)
		if !fired {
			t.Fatalf("expected the gate to fire on a threaded mismatch")
		}
		if res.OK {
			t.Fatalf("expected a refusal, got OK")
		}
		if res.Code != LandResultMessageBindingMismatch {
			t.Fatalf("expected code %q, got %q", LandResultMessageBindingMismatch, res.Code)
		}
	})

	t.Run("threaded match clears", func(t *testing.T) {
		wt := t.TempDir() // no intent written
		msgFile := filepath.Join(t.TempDir(), "msg.txt")
		if err := os.WriteFile(msgFile, []byte("feat(quality): split the suite under #4574 (fak quality)\n"), 0o600); err != nil {
			t.Fatalf("write msg: %v", err)
		}
		if res, fired := messageBindingLandGate(wt, msgFile, 4574, noopGit); fired {
			t.Fatalf("expected the gate to clear on a threaded match, got %+v", res)
		}
	})

	t.Run("threaded wins over intent", func(t *testing.T) {
		wt := t.TempDir()
		// The durable intent says #4574, but the threaded intended issue is #5098
		// (the authoritative value from the private lander). The final message
		// cites #5098, so it must clear against the threaded value even though it
		// would mismatch the intent.
		if err := SaveIntent(wt, "base-sha", "feat(quality): split the suite under #4574 (fak quality)", []string{"internal/quality/suite.go"}); err != nil {
			t.Fatalf("SaveIntent: %v", err)
		}
		msgFile := filepath.Join(t.TempDir(), "msg.txt")
		if err := os.WriteFile(msgFile, []byte("feat(model): bind polymodel.SpecDecode to a live path (#5098) (fak model)\n"), 0o600); err != nil {
			t.Fatalf("write msg: %v", err)
		}
		if res, fired := messageBindingLandGate(wt, msgFile, 5098, noopGit); fired {
			t.Fatalf("expected the threaded intended issue to win over the intent, got %+v", res)
		}
	})

	t.Run("threaded zero falls back to intent", func(t *testing.T) {
		wt := t.TempDir()
		// intendedIssue == 0 means "not supplied" -> fall back to the durable
		// Intent (#4574). The message cites #5098, so it must refuse.
		if err := SaveIntent(wt, "base-sha", "feat(quality): split the suite under #4574 (fak quality)", []string{"internal/quality/suite.go"}); err != nil {
			t.Fatalf("SaveIntent: %v", err)
		}
		msgFile := filepath.Join(t.TempDir(), "msg.txt")
		if err := os.WriteFile(msgFile, []byte("feat(model): bind polymodel.SpecDecode to a live path (#5098) (fak model)\n"), 0o600); err != nil {
			t.Fatalf("write msg: %v", err)
		}
		res, fired := messageBindingLandGate(wt, msgFile, 0, noopGit)
		if !fired || res.OK {
			t.Fatalf("expected the zero-threaded fallback to the intent to refuse, got fired=%v res=%+v", fired, res)
		}
	})
}
