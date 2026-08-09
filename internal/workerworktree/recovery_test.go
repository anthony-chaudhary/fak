package workerworktree

import (
	"strings"
	"testing"
)

func TestRecoveryRefValidatesWorkerAndSHA(t *testing.T) {
	got, err := RecoveryRef(`/tmp/fak-worker-wt-tools-abc`, "abcdef1234567")
	if err != nil || got != RecoveryRefPrefix+"fak-worker-wt-tools-abc/abcdef1234567" {
		t.Fatalf("RecoveryRef = %q, %v", got, err)
	}
	if _, err := RecoveryRef(`/tmp/fak-worker-wt-tools-abc`, "not-a-sha"); err == nil {
		t.Fatal("invalid sha accepted")
	}
}

func TestAnchorRecoveryEntryCreatesRefloggedRef(t *testing.T) {
	g := newFakeGit().reply("update-ref", 0, "")
	ref, err := AnchorRecoveryEntry("/repo", "/tmp/fak-worker-wt-tools-abc", "abcdef1234567", g.run)
	if err != nil {
		t.Fatal(err)
	}
	calls := g.callsWithPrefix("update-ref", "--create-reflog")
	if len(calls) != 1 || calls[0][3] != recoveryReflogMessage || calls[0][4] != ref || calls[0][5] != "abcdef1234567" {
		t.Fatalf("anchor call = %v", calls)
	}
}

func TestRecoveryEntriesClassifiesLandedAndRecoverable(t *testing.T) {
	g := newFakeGit()
	g.reply("for-each-ref", 0, strings.Join([]string{
		RecoveryRefPrefix + "fak-worker-wt-a-1/aaa1111\x00aaa1111",
		RecoveryRefPrefix + "fak-worker-wt-b-2/bbb2222\x00bbb2222",
	}, "\n"))
	g.replyOnce("merge-base", 0, "").replyOnce("merge-base", 1, "")
	items, err := RecoveryEntries("/repo", g.run)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 || items[0].State != "LANDED" || items[1].State != "RECOVERABLE" {
		t.Fatalf("candidates = %+v", items)
	}
}

func TestDeleteRecoveryRefRefusesUnlandedByDefault(t *testing.T) {
	g := newFakeGit().reply("rev-parse", 0, "abc1234\n").reply("merge-base", 1, "").reply("update-ref", 0, "")
	ref := RecoveryRefPrefix + "fak-worker-wt-tools-x/abc1234"
	if err := DeleteRecoveryRef("/repo", ref, false, g.run); err == nil || !strings.Contains(err.Error(), "not landed") {
		t.Fatalf("DeleteRecoveryRef error = %v", err)
	}
	if len(g.callsWithPrefix("update-ref", "-d")) != 0 {
		t.Fatal("unlanded ref was deleted")
	}
	if err := DeleteRecoveryRef("/repo", ref, true, g.run); err != nil {
		t.Fatalf("forced cleanup: %v", err)
	}
}

func TestIsolatedLandAnchorsCandidateBeforeTrunkCAS(t *testing.T) {
	t.Setenv(IsolatedLandRetryEnv, "1")
	g := newFakeGit().
		reply("config", 0, "Test User\ntest-at-example.invalid\n").
		reply("symbolic-ref", 0, "refs/heads/main\n").
		reply("rev-parse", 0, "old111122223333444455556666777788889999\n").
		reply("read-tree", 0, "").
		reply("apply", 0, "").
		reply("write-tree", 0, "tree11112222333344445555666677778888999\n").
		reply("commit-tree", 0, "new222233334444555566667777888899990000\n").
		reply("update-ref", 0, "").
		reply("checkout", 0, "")
	res, handled := landIsolated("/repo", "/tmp/fak-worker-wt-tools-abc", "patch", writeMsg(t, "fix: durable"), []string{"x"}, g.run, g.runEnv)
	if !handled || !res.OK || !strings.Contains(res.Detail, "recovery-ref=") {
		t.Fatalf("land = %+v handled=%v", res, handled)
	}
	anchors := g.envCallsWithPrefix("update-ref", "--create-reflog")
	cas := g.callsWithPrefix("update-ref", "refs/heads/main")
	if len(anchors) != 1 || len(cas) != 1 {
		t.Fatalf("anchor=%v cas=%v", anchors, cas)
	}
}
