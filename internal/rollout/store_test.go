package rollout

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStoreInstallLoadAndActivateWithoutOverwritingArtifacts(t *testing.T) {
	store := NewStore(t.TempDir())
	artifactA := []byte("fak generation a")
	artifactB := []byte("fak generation b")
	generationA := generationForArtifact("gen-a", artifactA)
	generationB := generationForArtifact("gen-b", artifactB)

	if err := store.Install(generationA, artifactA); err != nil {
		t.Fatalf("install generation A: %v", err)
	}
	if err := store.Install(generationB, artifactB); err != nil {
		t.Fatalf("install generation B: %v", err)
	}
	for _, activation := range []PointerName{PointerStable, PointerCanary, PointerLastKnownGood} {
		if err := store.Activate(activation, generationA); err != nil {
			t.Fatalf("activate %s: %v", activation, err)
		}
	}

	if err := store.Activate(PointerStable, generationB); err != nil {
		t.Fatalf("switch stable activation: %v", err)
	}
	if err := store.Activate(PointerLastKnownGood, generationA); err != nil {
		t.Fatalf("retain LKG activation: %v", err)
	}

	stable, err := store.Active(PointerStable)
	if err != nil {
		t.Fatalf("read stable activation: %v", err)
	}
	if stable != generationB {
		t.Fatalf("stable = %#v, want %#v", stable, generationB)
	}
	lkg, err := store.Active(PointerLastKnownGood)
	if err != nil {
		t.Fatalf("read LKG activation: %v", err)
	}
	if lkg != generationA {
		t.Fatalf("LKG = %#v, want %#v", lkg, generationA)
	}

	gotA, err := store.Artifact(generationA)
	if err != nil {
		t.Fatalf("read generation A after pointer changes: %v", err)
	}
	gotB, err := store.Artifact(generationB)
	if err != nil {
		t.Fatalf("read generation B after pointer changes: %v", err)
	}
	if !bytes.Equal(gotA, artifactA) || !bytes.Equal(gotB, artifactB) {
		t.Fatalf("activation changed generation bytes: A=%q B=%q", gotA, gotB)
	}
}

func TestStoreRejectsDigestMismatchBeforeWriting(t *testing.T) {
	root := t.TempDir()
	store := NewStore(root)
	artifact := []byte("artifact")
	generation := generationForArtifact("gen-a", []byte("different artifact"))

	err := store.Install(generation, artifact)
	if err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("Install error = %v, want digest mismatch", err)
	}
	entries, readErr := os.ReadDir(root)
	if readErr != nil {
		t.Fatalf("read store root: %v", readErr)
	}
	if len(entries) != 0 {
		t.Fatalf("digest mismatch wrote store entries: %v", entries)
	}
}

func TestStoreRejectsGenerationIdentityMismatch(t *testing.T) {
	root := t.TempDir()
	store := NewStore(root)
	artifact := []byte("artifact")
	generation := generationForArtifact("gen-a", artifact)
	if err := store.Install(generation, artifact); err != nil {
		t.Fatalf("install: %v", err)
	}

	manifestPath := store.manifestPath(generation.ID)
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	data = bytes.Replace(data, []byte(`"gen-a"`), []byte(`"gen-b"`), 1)
	if err := os.Chmod(manifestPath, 0o644); err != nil {
		t.Fatalf("chmod manifest: %v", err)
	}
	if err := os.WriteFile(manifestPath, data, 0o444); err != nil {
		t.Fatalf("corrupt manifest: %v", err)
	}

	_, err = store.Load(generation.ID)
	if err == nil || !strings.Contains(err.Error(), "identity mismatch") {
		t.Fatalf("Load error = %v, want identity mismatch", err)
	}
}

func TestStoreRejectsArtifactCorruption(t *testing.T) {
	store := NewStore(t.TempDir())
	artifact := []byte("artifact")
	generation := generationForArtifact("gen-a", artifact)
	if err := store.Install(generation, artifact); err != nil {
		t.Fatalf("install: %v", err)
	}

	digestHex, err := parseDigest(generation.Digest)
	if err != nil {
		t.Fatalf("parse digest: %v", err)
	}
	objectPath := store.objectPath(digestHex)
	if err := os.Chmod(objectPath, 0o644); err != nil {
		t.Fatalf("chmod artifact: %v", err)
	}
	if err := os.WriteFile(objectPath, []byte("corrupt"), 0o444); err != nil {
		t.Fatalf("corrupt artifact: %v", err)
	}

	_, err = store.Artifact(generation)
	if err == nil || !strings.Contains(err.Error(), "artifact corruption") {
		t.Fatalf("Artifact error = %v, want corruption refusal", err)
	}
	if err := store.Activate(PointerStable, generation); err == nil || !strings.Contains(err.Error(), "artifact corruption") {
		t.Fatalf("Activate error = %v, want corruption refusal", err)
	}
}

func TestStoreRejectsGenerationIDReuseAndKeepsOriginal(t *testing.T) {
	store := NewStore(t.TempDir())
	originalArtifact := []byte("original")
	replacementArtifact := []byte("replacement")
	original := generationForArtifact("gen-a", originalArtifact)
	replacement := generationForArtifact("gen-a", replacementArtifact)
	if err := store.Install(original, originalArtifact); err != nil {
		t.Fatalf("install original: %v", err)
	}

	err := store.Install(replacement, replacementArtifact)
	if err == nil || !strings.Contains(err.Error(), "immutable file already exists") {
		t.Fatalf("replacement Install error = %v, want immutable refusal", err)
	}
	got, err := store.Artifact(original)
	if err != nil {
		t.Fatalf("read original: %v", err)
	}
	if !bytes.Equal(got, originalArtifact) {
		t.Fatalf("original bytes = %q, want %q", got, originalArtifact)
	}
}

func TestStorePointerNamePointerIsAlwaysCompleteDuringSwitches(t *testing.T) {
	store := NewStore(t.TempDir())
	artifactA := []byte("a")
	artifactB := []byte("b")
	generationA := generationForArtifact("gen-a", artifactA)
	generationB := generationForArtifact("gen-b", artifactB)
	if err := store.Install(generationA, artifactA); err != nil {
		t.Fatalf("install A: %v", err)
	}
	if err := store.Install(generationB, artifactB); err != nil {
		t.Fatalf("install B: %v", err)
	}
	if err := store.Activate(PointerStable, generationA); err != nil {
		t.Fatalf("activate initial stable: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		for i := 0; i < 100; i++ {
			generation := generationA
			if i%2 == 1 {
				generation = generationB
			}
			if err := store.Activate(PointerStable, generation); err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()

	for {
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("switch activation: %v", err)
			}
			return
		default:
			active, err := store.Active(PointerStable)
			if err != nil {
				t.Fatalf("read activation during switch: %v", err)
			}
			if active != generationA && active != generationB {
				t.Fatalf("partial activation = %#v", active)
			}
		}
	}
}

func TestStoreRejectsUnknownPointerName(t *testing.T) {
	store := NewStore(t.TempDir())
	generation := generationForArtifact("gen-a", []byte("artifact"))
	if err := store.Activate(PointerName("other"), generation); err == nil || !strings.Contains(err.Error(), "unknown activation") {
		t.Fatalf("Activate error = %v, want unknown activation", err)
	}
	if _, err := store.Active(PointerName("other")); err == nil || !strings.Contains(err.Error(), "unknown activation") {
		t.Fatalf("Active error = %v, want unknown activation", err)
	}
}

func generationForArtifact(id string, artifact []byte) Generation {
	digest := sha256.Sum256(artifact)
	return Generation{ID: id, Digest: fmt.Sprintf("sha256:%x", digest)}
}

func TestStoreManifestPathDoesNotUseGenerationIDAsPath(t *testing.T) {
	store := NewStore(t.TempDir())
	artifact := []byte("artifact")
	generation := generationForArtifact(filepath.Join("..", "escape"), artifact)
	if err := store.Install(generation, artifact); err != nil {
		t.Fatalf("install path-like ID: %v", err)
	}
	if _, err := store.Load(generation.ID); err != nil {
		t.Fatalf("load path-like ID: %v", err)
	}
}
