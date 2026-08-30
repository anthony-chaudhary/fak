package selfinstall

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAdvanceInstallIdentityDigestEqualRefreshPreservesAppVersion(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "fak.self-update-identity.json")
	activePath := filepath.Join(dir, "fak")
	artifact := []byte("same verified artifact")
	if err := os.WriteFile(activePath, artifact, 0o755); err != nil {
		t.Fatal(err)
	}
	artifactDigest := digestBytes(artifact)
	buildDigest := digestBytes([]byte("build inputs"))
	first := StateUpdate{
		SelectedSourceCommit: cacheTestCommitA,
		ArtifactSourceCommit: cacheTestCommitA,
		BuildInputDigest:     buildDigest,
		ArtifactDigest:       artifactDigest,
		ArtifactSize:         int64(len(artifact)),
		AppVersion:           "1.2.3",
	}
	state, err := AdvanceInstallIdentity(statePath, InstallIdentity{}, first, activePath, filepath.Join(dir, "prior"), true)
	if err != nil {
		t.Fatal(err)
	}

	second := first
	second.SelectedSourceCommit = cacheTestCommitB
	second.AppVersion = "9.9.9-metadata-must-not-win"
	state, err = AdvanceInstallIdentity(statePath, state, second, activePath, filepath.Join(dir, "prior"), false)
	if err != nil {
		t.Fatal(err)
	}
	if state.MetadataGeneration != 2 || state.SelectedSourceCommit != cacheTestCommitB {
		t.Fatalf("metadata refresh = %+v", state)
	}
	if state.AppVersion != "1.2.3" {
		t.Fatalf("digest-equal app version = %q, want preserved 1.2.3", state.AppVersion)
	}
	active, ok := VerifiedArtifact(state, artifactDigest)
	if !ok || active.ArtifactSourceCommit != cacheTestCommitA || active.AppVersion != "1.2.3" {
		t.Fatalf("active verified slot = %+v ok=%v", active, ok)
	}

	persisted, err := ReadInstallIdentity(statePath)
	if err != nil {
		t.Fatal(err)
	}
	if persisted.MetadataGeneration != 2 || persisted.SelectedSourceCommit != cacheTestCommitB ||
		persisted.ArtifactDigest != artifactDigest || persisted.AppVersion != "1.2.3" {
		t.Fatalf("persisted identity = %+v", persisted)
	}
}

func TestAdvanceInstallIdentityActivationKeepsRollbackAsVerifiedSlot(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "identity.json")
	activePath := filepath.Join(dir, "fak")
	rollbackPath := filepath.Join(dir, "fak.self-update-prior")
	oldArtifact := []byte("old verified artifact")
	newArtifact := []byte("new verified artifact")
	oldDigest := digestBytes(oldArtifact)
	newDigest := digestBytes(newArtifact)
	buildDigest := digestBytes([]byte("build inputs"))

	prior, err := AdvanceInstallIdentity(statePath, InstallIdentity{}, StateUpdate{
		SelectedSourceCommit: cacheTestCommitA,
		ArtifactSourceCommit: cacheTestCommitA,
		BuildInputDigest:     buildDigest,
		ArtifactDigest:       oldDigest,
		ArtifactSize:         int64(len(oldArtifact)),
		AppVersion:           "1.0.0",
	}, activePath, rollbackPath, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(rollbackPath, oldArtifact, 0o755); err != nil {
		t.Fatal(err)
	}
	state, err := AdvanceInstallIdentity(statePath, prior, StateUpdate{
		SelectedSourceCommit: cacheTestCommitB,
		ArtifactSourceCommit: cacheTestCommitB,
		BuildInputDigest:     digestBytes([]byte("new build inputs")),
		ArtifactDigest:       newDigest,
		ArtifactSize:         int64(len(newArtifact)),
		AppVersion:           "2.0.0",
	}, activePath, rollbackPath, true)
	if err != nil {
		t.Fatal(err)
	}
	if state.AppVersion != "2.0.0" || state.CurrentDigest != newDigest {
		t.Fatalf("activated identity = %+v", state)
	}
	rollback, ok := VerifiedArtifact(state, oldDigest)
	if !ok || rollback.Path != rollbackPath || rollback.AppVersion != "1.0.0" {
		t.Fatalf("rollback slot = %+v ok=%v", rollback, ok)
	}
	if _, ok := VerifiedArtifact(state, "1.0.0"); ok {
		t.Fatal("ambiguous app version resolved as a rollback slot")
	}
}
