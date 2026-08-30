package selfinstall

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdvanceInstallIdentityPersistsCanonicalBuildInputDigest(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "identity.json")
	buildDigest := buildInputDigest([]byte("build inputs"))
	state, err := AdvanceInstallIdentity(statePath, InstallIdentity{}, StateUpdate{
		SelectedSourceCommit: cacheTestCommitA,
		ArtifactSourceCommit: cacheTestCommitA,
		BuildInputDigest:     buildDigest,
		ArtifactDigest:       digestBytes([]byte("artifact")),
		ArtifactSize:         int64(len("artifact")),
		AppVersion:           "1.2.3",
	}, filepath.Join(dir, "fak"), filepath.Join(dir, "prior"), true)
	if err != nil {
		t.Fatal(err)
	}
	if state.BuildInputDigest != buildDigest {
		t.Fatalf("build-input digest = %q, want %q", state.BuildInputDigest, buildDigest)
	}
	persisted, err := ReadInstallIdentity(statePath)
	if err != nil {
		t.Fatal(err)
	}
	if persisted.BuildInputDigest != buildDigest {
		t.Fatalf("persisted build-input digest = %q, want %q", persisted.BuildInputDigest, buildDigest)
	}
}

func TestAdvanceInstallIdentityRejectsMalformedBuildInputDigest(t *testing.T) {
	digest := digestBytes([]byte("build inputs"))
	tests := []struct {
		name   string
		digest string
	}{
		{name: "missing prefix", digest: digest},
		{name: "wrong prefix", digest: "sha512:" + digest},
		{name: "uppercase prefix", digest: "SHA256:" + digest},
		{name: "short digest", digest: "sha256:" + digest[:len(digest)-1]},
		{name: "non hex digest", digest: "sha256:" + strings.Repeat("g", len(digest))},
		{name: "uppercase digest", digest: "sha256:" + strings.ToUpper(digest)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := AdvanceInstallIdentity(filepath.Join(t.TempDir(), "identity.json"), InstallIdentity{}, StateUpdate{
				SelectedSourceCommit: cacheTestCommitA,
				ArtifactSourceCommit: cacheTestCommitA,
				BuildInputDigest:     tt.digest,
				ArtifactDigest:       digestBytes([]byte("artifact")),
				ArtifactSize:         int64(len("artifact")),
				AppVersion:           "1.2.3",
			}, "fak", "prior", true)
			if err == nil || !strings.Contains(err.Error(), "build-input digest is not SHA-256") {
				t.Fatalf("AdvanceInstallIdentity error = %v, want malformed build-input digest rejection", err)
			}
		})
	}
}

func TestAdvanceInstallIdentityDigestEqualRefreshPreservesAppVersion(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "fak.self-update-identity.json")
	activePath := filepath.Join(dir, "fak")
	artifact := []byte("same verified artifact")
	if err := os.WriteFile(activePath, artifact, 0o755); err != nil {
		t.Fatal(err)
	}
	artifactDigest := digestBytes(artifact)
	buildDigest := buildInputDigest([]byte("build inputs"))
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
	buildDigest := buildInputDigest([]byte("build inputs"))

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
		BuildInputDigest:     buildInputDigest([]byte("new build inputs")),
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

func TestAdvanceInstallIdentityRejectsSignedGenerationRollbackAndFreeze(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "identity.json")
	activePath := filepath.Join(dir, "slot")
	body := []byte("verified slot")
	if err := os.WriteFile(activePath, body, 0o755); err != nil {
		t.Fatal(err)
	}
	base := StateUpdate{
		SignedMetadataGeneration: 5, SelectedSourceCommit: cacheTestCommitA, ArtifactSourceCommit: cacheTestCommitA,
		BuildInputDigest: buildInputDigest([]byte("inputs")), ArtifactDigest: digestBytes(body),
		ArtifactSize: int64(len(body)), AppVersion: "2.0.0",
	}
	state, err := AdvanceInstallIdentity(statePath, InstallIdentity{}, base, activePath, filepath.Join(dir, "prior"), true)
	if err != nil {
		t.Fatal(err)
	}
	rollback := base
	rollback.SignedMetadataGeneration = 4
	if _, err := AdvanceInstallIdentity(statePath, state, rollback, activePath, filepath.Join(dir, "prior"), false); err == nil {
		t.Fatal("metadata generation rollback accepted")
	}
	freeze := base
	freeze.AppVersion = "2.0.1"
	if _, err := AdvanceInstallIdentity(statePath, state, freeze, activePath, filepath.Join(dir, "prior"), false); err == nil {
		t.Fatal("same-generation changed identity accepted")
	}
	replay, err := AdvanceInstallIdentity(statePath, state, base, activePath, filepath.Join(dir, "prior"), false)
	if err != nil || replay.SignedMetadataGeneration != 5 || replay.MetadataGeneration != 1 {
		t.Fatalf("identical generation replay = %+v, %v", replay, err)
	}
}

func buildInputDigest(body []byte) string {
	return "sha256:" + digestBytes(body)
}
