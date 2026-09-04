package safesync

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func TestBuildPacketDisjointDivergence(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "remote_file.txt"), "remote content\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote file commit")

	writeFile(t, filepath.Join(clone, "local_file.txt"), "local content\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local file commit")

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

	if pkt.Schema != PacketSchema {
		t.Errorf("schema = %q, want %q", pkt.Schema, PacketSchema)
	}
	if pkt.Disposition != DispositionSafeDisjoint {
		t.Errorf("disposition = %q, want %q", pkt.Disposition, DispositionSafeDisjoint)
	}
	if !pkt.Dispatchable {
		t.Errorf("dispatchable = %v, want true", pkt.Dispatchable)
	}
	if !pkt.MergePreview.Clean {
		t.Errorf("merge preview clean = %v, want true", pkt.MergePreview.Clean)
	}
	if len(pkt.LocalCommits) != 1 {
		t.Fatalf("local commits count = %d, want 1", len(pkt.LocalCommits))
	}
	if len(pkt.RemoteCommits) != 1 {
		t.Fatalf("remote commits count = %d, want 1", len(pkt.RemoteCommits))
	}
	if len(pkt.RequiredWitnesses) == 0 {
		t.Errorf("expected required witnesses")
	}
}

func TestBuildPacketContentCollision(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "shared.txt"), "remote conflict line\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote edit shared")

	writeFile(t, filepath.Join(clone, "shared.txt"), "local conflict line\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local edit shared")

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

	if pkt.Disposition != DispositionSemanticConflictReview {
		t.Errorf("disposition = %q, want %q", pkt.Disposition, DispositionSemanticConflictReview)
	}
	if pkt.Dispatchable {
		t.Errorf("dispatchable = %v, want false", pkt.Dispatchable)
	}
	if pkt.MergePreview.Clean {
		t.Errorf("merge preview clean = %v, want false", pkt.MergePreview.Clean)
	}
	if len(pkt.MergePreview.Conflicts) == 0 {
		t.Fatalf("expected merge preview conflicts on shared.txt")
	}
	foundShared := false
	for _, c := range pkt.MergePreview.Conflicts {
		if c == "shared.txt" {
			foundShared = true
			break
		}
	}
	if !foundShared {
		t.Errorf("conflicts = %v, want shared.txt", pkt.MergePreview.Conflicts)
	}
	if info, ok := pkt.PathOwnership["shared.txt"]; !ok || info.Active {
		t.Errorf("shared.txt active = %v, want false (no active lease held)", info.Active)
	}
}

func TestBuildPacketActivePeerLease(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "shared.txt"), "remote line\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote edit shared")

	writeFile(t, filepath.Join(clone, "shared.txt"), "local line\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local edit shared")

	git(t, clone, "fetch", "origin")

	store := leaseref.NewInDir(clone)
	_, err := store.Acquire(context.Background(), leaseref.Record{
		ID:         "shared",
		TreeGlobs:  []string{"shared.txt"},
		Holder:     "peer-agent",
		SessionID:  "peer-session-xyz",
		AcquiredAt: time.Now().Unix(),
		TTLSeconds: 3600,
	})
	if err != nil {
		t.Fatalf("acquire peer lease: %v", err)
	}

	opts := PacketOptions{
		Repo:    clone,
		Remote:  "origin",
		Branch:  "work",
		Session: "my-session",
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	if pkt.Disposition != DispositionWaitForOwner {
		t.Errorf("disposition = %q, want %q", pkt.Disposition, DispositionWaitForOwner)
	}
	if pkt.Dispatchable {
		t.Errorf("dispatchable = %v, want false", pkt.Dispatchable)
	}
	info, ok := pkt.PathOwnership["shared.txt"]
	if !ok {
		t.Fatalf("missing PathOwnership for shared.txt")
	}
	if !info.Active {
		t.Errorf("shared.txt active = %v, want true", info.Active)
	}
	if info.Owner != "peer-agent" {
		t.Errorf("shared.txt owner = %q, want peer-agent", info.Owner)
	}
	if info.SessionID != "peer-session-xyz" {
		t.Errorf("shared.txt session = %q, want peer-session-xyz", info.SessionID)
	}
}

func TestBuildPacketDeterministicJSON(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "remote_b.txt"), "b\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote b")

	writeFile(t, filepath.Join(clone, "local_a.txt"), "a\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local a")

	git(t, clone, "fetch", "origin")

	fixedTime := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	nowFn := func() time.Time { return fixedTime }

	opts := PacketOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
		Now:    nowFn,
	}

	pkt1, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket 1: %v", err)
	}
	pkt2, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket 2: %v", err)
	}

	b1, err := json.MarshalIndent(pkt1, "", "  ")
	if err != nil {
		t.Fatalf("marshal pkt1: %v", err)
	}
	b2, err := json.MarshalIndent(pkt2, "", "  ")
	if err != nil {
		t.Fatalf("marshal pkt2: %v", err)
	}

	if !bytes.Equal(b1, b2) {
		t.Fatalf("JSON marshaling is non-deterministic:\nFirst:\n%s\nSecond:\n%s", string(b1), string(b2))
	}

	var decoded ReconciliationPacket
	if err := json.Unmarshal(b1, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.Schema != PacketSchema {
		t.Errorf("schema = %q, want %q", decoded.Schema, PacketSchema)
	}
	if decoded.GeneratedAt != fixedTime.Format(time.RFC3339) {
		t.Errorf("generated_at = %q, want %q", decoded.GeneratedAt, fixedTime.Format(time.RFC3339))
	}
}

func TestBuildPacketOwnerAuthorizedPathSuspend(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(origin, "incoming.txt"), "incoming line\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote add incoming")

	git(t, clone, "fetch", "origin")

	writeFile(t, filepath.Join(clone, "incoming.txt"), "dirty local edit\n")

	opts := PacketOptions{
		Repo:         clone,
		Remote:       "origin",
		Branch:       "work",
		SuspendPaths: []string{"incoming.txt"},
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	if pkt.Disposition != DispositionOwnerAuthorizedPathSuspend {
		t.Errorf("disposition = %q, want %q", pkt.Disposition, DispositionOwnerAuthorizedPathSuspend)
	}
	if !pkt.Dispatchable {
		t.Errorf("dispatchable = %v, want true", pkt.Dispatchable)
	}
}

func TestBuildPacketTrivialSuperset(t *testing.T) {
	_, clone := setupTestOriginAndClone(t)

	writeFile(t, filepath.Join(clone, "ahead.txt"), "ahead line\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local ahead")

	opts := PacketOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
	}
	pkt, err := BuildReconciliationPacket(context.Background(), opts)
	if err != nil {
		t.Fatalf("BuildReconciliationPacket: %v", err)
	}

	if pkt.Disposition != DispositionTrivialSuperset {
		t.Errorf("disposition = %q, want %q", pkt.Disposition, DispositionTrivialSuperset)
	}
	if !pkt.Dispatchable {
		t.Errorf("dispatchable = %v, want true", pkt.Dispatchable)
	}
}
