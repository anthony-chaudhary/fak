package safesync

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExecutePacket_SuccessfulDisjointWithReadback(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	// Remote commit on origin
	writeFile(t, filepath.Join(origin, "remote_feat.txt"), "remote line\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote feature commit")

	// Local commit on clone
	writeFile(t, filepath.Join(clone, "local_feat.txt"), "local line\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local feature commit")

	git(t, clone, "fetch", "origin")

	opts := PacketOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	if pkt.Disposition != DispositionSafeDisjoint {
		t.Fatalf("disposition = %s, want safe-disjoint", pkt.Disposition)
	}
	if !pkt.Dispatchable {
		t.Fatalf("dispatchable = false, want true")
	}

	execOpts := ExecuteOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	receipt, err := ExecutePacket(context.Background(), pkt, execOpts)
	if err != nil {
		t.Fatalf("ExecutePacket failed: %v", err)
	}

	if receipt == nil {
		t.Fatal("receipt is nil")
	}
	if receipt.Schema != ExecuteReceiptSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, ExecuteReceiptSchema)
	}
	if receipt.Status != ExecuteStatusExecuted {
		t.Errorf("status = %q, want %q", receipt.Status, ExecuteStatusExecuted)
	}
	if !receipt.Pushed {
		t.Errorf("pushed = %v, want true", receipt.Pushed)
	}
	if !receipt.LocalCommitsContained {
		t.Errorf("local_commits_contained = %v, want true", receipt.LocalCommitsContained)
	}
	if !receipt.PeerBytesPreserved {
		t.Errorf("peer_bytes_preserved = %v, want true", receipt.PeerBytesPreserved)
	}
	if receipt.NewHEAD == "" {
		t.Error("new_head is empty")
	}

	// Verify origin now has the merge commit as HEAD
	originHead, err := rev(context.Background(), RealRunner, origin, "HEAD")
	if err != nil {
		t.Fatalf("rev origin HEAD: %v", err)
	}
	if originHead != receipt.NewHEAD {
		t.Errorf("origin HEAD %s != receipt NewHEAD %s", originHead, receipt.NewHEAD)
	}

	// Independent graph readback check: local commit is an ancestor of origin/work
	for _, c := range pkt.LocalCommits {
		mbRes := RealRunner(context.Background(), clone, "merge-base", "--is-ancestor", c.SHA, "origin/work")
		if mbRes.Code != 0 {
			t.Errorf("expected local commit %s to be ancestor of origin/work", c.SHA)
		}
	}
}

func TestExecutePacket_StaleTargetRefRefusal(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "remote1.txt"), "remote 1\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote 1")

	writeFile(t, filepath.Join(clone, "local1.txt"), "local 1\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local 1")

	git(t, clone, "fetch", "origin")

	opts := PacketOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	// Move target ref on origin and fetch to make packet stale
	writeFile(t, filepath.Join(origin, "remote2.txt"), "remote 2\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote 2")
	git(t, clone, "fetch", "origin")

	execOpts := ExecuteOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	receipt, err := ExecutePacket(context.Background(), pkt, execOpts)
	if err == nil {
		t.Fatal("expected error on stale target ref, got nil")
	}
	if !strings.Contains(err.Error(), ReasonTargetMoved) {
		t.Errorf("expected error to contain %s, got %v", ReasonTargetMoved, err)
	}
	if receipt != nil && receipt.Status != ExecuteStatusRefused {
		t.Errorf("expected receipt status REFUSED, got %s", receipt.Status)
	}
}

func TestExecutePacket_LocalHEADChangedRefusal(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "remote_a.txt"), "remote a\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote a")

	writeFile(t, filepath.Join(clone, "local_a.txt"), "local a\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local a")

	git(t, clone, "fetch", "origin")

	opts := PacketOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	// Advance local HEAD after packet generation
	writeFile(t, filepath.Join(clone, "local_b.txt"), "local b\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local b")

	execOpts := ExecuteOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	receipt, err := ExecutePacket(context.Background(), pkt, execOpts)
	if err == nil {
		t.Fatal("expected error on local HEAD change, got nil")
	}
	if !strings.Contains(err.Error(), ReasonPathspecRace) {
		t.Errorf("expected error to contain %s, got %v", ReasonPathspecRace, err)
	}
	if receipt != nil && receipt.Status != ExecuteStatusRefused {
		t.Errorf("expected receipt status REFUSED, got %s", receipt.Status)
	}
}

func TestExecutePacket_DirtyPathsDriftedRefusal(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "r.txt"), "r\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "r")

	writeFile(t, filepath.Join(clone, "l.txt"), "l\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "l")

	git(t, clone, "fetch", "origin")

	opts := PacketOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	// Create dirty file after packet was generated with clean tree
	writeFile(t, filepath.Join(clone, "drift_dirty.txt"), "drift\n")

	execOpts := ExecuteOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	receipt, err := ExecutePacket(context.Background(), pkt, execOpts)
	if err == nil {
		t.Fatal("expected error on dirty paths drift, got nil")
	}
	if !strings.Contains(err.Error(), ReasonDirtyWriteOverlap) {
		t.Errorf("expected error to contain %s, got %v", ReasonDirtyWriteOverlap, err)
	}
	if receipt != nil && receipt.Status != ExecuteStatusRefused {
		t.Errorf("expected receipt status REFUSED, got %s", receipt.Status)
	}
}

func TestExecutePacket_NonDispatchableRefusal(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	// Cause content collision on shared.txt
	writeFile(t, filepath.Join(origin, "shared.txt"), "origin modification\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "origin collision")

	writeFile(t, filepath.Join(clone, "shared.txt"), "clone modification\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "clone collision")

	git(t, clone, "fetch", "origin")

	opts := PacketOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	if pkt.Dispatchable {
		t.Fatalf("expected dispatchable = false for semantic conflict, got true")
	}
	if pkt.Disposition != DispositionSemanticConflictReview {
		t.Fatalf("disposition = %s, want semantic-conflict-review", pkt.Disposition)
	}

	execOpts := ExecuteOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	_, err = ExecutePacket(context.Background(), pkt, execOpts)
	if err == nil {
		t.Fatal("expected error for non-dispatchable packet, got nil")
	}
	if !strings.Contains(err.Error(), ReasonDivergedOverlap) {
		t.Errorf("expected error to contain %s, got %v", ReasonDivergedOverlap, err)
	}

	// Test wait-for-owner disposition
	pkt.Disposition = DispositionWaitForOwner
	pkt.Dispatchable = false
	_, err = ExecutePacket(context.Background(), pkt, execOpts)
	if err == nil {
		t.Fatal("expected error for wait-for-owner packet, got nil")
	}
	if !strings.Contains(err.Error(), ReasonLeaseOwnerUnavailable) {
		t.Errorf("expected error to contain %s, got %v", ReasonLeaseOwnerUnavailable, err)
	}
}

func TestExecutePacket_PeerBytePreservation(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "remote_work.txt"), "remote work\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote work")

	writeFile(t, filepath.Join(clone, "local_work.txt"), "local work\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local work")

	// Create a peer uncommitted file in clone
	peerContent := "peer uncommitted work: critical bytes that must never be lost\n"
	writeFile(t, filepath.Join(clone, "peer_dirty.txt"), peerContent)

	git(t, clone, "fetch", "origin")

	opts := PacketOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	if !pkt.Dispatchable || pkt.Disposition != DispositionSafeDisjoint {
		t.Fatalf("expected dispatchable safe-disjoint, got %s (dispatchable=%v)", pkt.Disposition, pkt.Dispatchable)
	}

	execOpts := ExecuteOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	receipt, err := ExecutePacket(context.Background(), pkt, execOpts)
	if err != nil {
		t.Fatalf("ExecutePacket failed: %v", err)
	}

	if !receipt.PeerBytesPreserved {
		t.Errorf("peer_bytes_preserved = false, want true")
	}

	// Verify peer_dirty.txt content is unchanged on disk
	actualBytes, err := os.ReadFile(filepath.Join(clone, "peer_dirty.txt"))
	if err != nil {
		t.Fatalf("read peer_dirty.txt: %v", err)
	}
	if string(actualBytes) != peerContent {
		t.Errorf("peer bytes changed! got %q, want %q", string(actualBytes), peerContent)
	}
}

func TestExecutePacket_OwnerAuthorizedPathSuspend(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	// Add multiline suspend_target.txt to origin and pull to clone
	writeFile(t, filepath.Join(origin, "suspend_target.txt"), "top\nmiddle\nbottom\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "add suspend_target")

	git(t, clone, "checkout", "HEAD", "--", ".")
	git(t, clone, "pull", "--ff-only", "origin", "work")

	// Remote commits to suspend_target.txt modifying top
	writeFile(t, filepath.Join(origin, "suspend_target.txt"), "top-upstream\nmiddle\nbottom\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "upstream shared edit")

	// Clone has uncommitted modification to suspend_target.txt modifying bottom
	writeFile(t, filepath.Join(clone, "suspend_target.txt"), "top\nmiddle\nbottom-local\n")
	// And an unrelated peer uncommitted file
	peerContent := "unrelated peer work\n"
	writeFile(t, filepath.Join(clone, "peer_unrelated.txt"), peerContent)

	git(t, clone, "fetch", "origin")

	opts := PacketOptions{
		Repo:         clone,
		Remote:       "origin",
		Branch:       "work",
		SuspendPaths: []string{"suspend_target.txt"},
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	if pkt.Disposition != DispositionOwnerAuthorizedPathSuspend {
		t.Fatalf("disposition = %s, want owner-authorized-path-suspend", pkt.Disposition)
	}
	if !pkt.Dispatchable {
		t.Fatalf("dispatchable = false, want true")
	}

	execOpts := ExecuteOptions{
		Repo:         clone,
		Remote:       "origin",
		Branch:       "work",
		SuspendPaths: []string{"suspend_target.txt"},
	}
	receipt, err := ExecutePacket(context.Background(), pkt, execOpts)
	if err != nil {
		t.Fatalf("ExecutePacket failed: %v", err)
	}

	if receipt.Status != ExecuteStatusExecuted {
		t.Errorf("status = %s, want EXECUTED", receipt.Status)
	}
	if !receipt.PeerBytesPreserved {
		t.Errorf("peer_bytes_preserved = false, want true")
	}

	// Verify unrelated peer file is preserved
	actualPeer, err := os.ReadFile(filepath.Join(clone, "peer_unrelated.txt"))
	if err != nil {
		t.Fatalf("read peer_unrelated.txt: %v", err)
	}
	if string(actualPeer) != peerContent {
		t.Errorf("unrelated peer content altered: got %q, want %q", string(actualPeer), peerContent)
	}
}
