package disambiguation

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewAdmittedIndexAcceptsPublicOwnerFixture(t *testing.T) {
	manifests := PublicManifests{Leaves: []string{"canon"}, Lanes: []string{"canon"}}
	if _, err := NewAdmittedIndex(selfCheckEntries(), manifests); err != nil {
		t.Fatalf("accepted fixture: %v", err)
	}
}

func TestNewAdmittedIndexRejectsAbsentOwnerLeafAndLane(t *testing.T) {
	manifests := PublicManifests{Leaves: []string{"canon"}, Lanes: []string{"canon"}}
	tests := []struct {
		name, want string
		mutate     func([]Entry)
	}{
		{"leaf", "owner leaf", func(entries []Entry) { entries[0].Owner.Leaf = "missing" }},
		{"lane", "dispatch lane", func(entries []Entry) { entries[0].Owner.Lane = "missing" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entries := selfCheckEntries()
			tt.mutate(entries)
			if _, err := NewAdmittedIndex(entries, manifests); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %q rejection", err, tt.want)
			}
		})
	}
}

func TestLoadPublicManifestsUsesRepositoryContracts(t *testing.T) {
	root := t.TempDir()
	for _, dir := range []string{"internal/canon", "cmd/fak"} {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(dir)), 0755); err != nil {
			t.Fatal(err)
		}
	}
	manifest := "[lanes]\nconcurrent = [\n  \"canon\",\n]\nexclusive = [\"cmd\"]\n[lanes.trees]\ncanon = [\"internal/canon/**\"]\n"
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(manifest), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadPublicManifests(root)
	if err != nil {
		t.Fatal(err)
	}
	if !contains(got.Leaves, "canon") || !contains(got.Leaves, "fak") {
		t.Fatalf("leaves=%v", got.Leaves)
	}
	if !contains(got.Lanes, "canon") || !contains(got.Lanes, "cmd") {
		t.Fatalf("lanes=%v", got.Lanes)
	}
}

func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}
