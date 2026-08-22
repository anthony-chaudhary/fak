package ultracodeborrow

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func repositoryRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func checkedArtifact(t *testing.T) Artifact {
	t.Helper()
	root := repositoryRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "notes", "CONCEPT-STUDY-ULTRACODE-WORKFLOWS-2026-08-21.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckPublicText("companion", raw); err != nil {
		t.Fatal(err)
	}
	artifact, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	note, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(artifact.Note)))
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckPublicText("note", note); err != nil {
		t.Fatal(err)
	}
	return artifact
}

func TestCompanionSatisfiesIssue8484(t *testing.T) {
	artifact := checkedArtifact(t)
	if got, want := len(artifact.Mechanisms), len(RequiredMechanisms()); got != want {
		t.Fatalf("mechanisms=%d want=%d", got, want)
	}
}

func TestCheckerRejectsMissingLicenseBenchmarkAndDuplicateOwnership(t *testing.T) {
	base := checkedArtifact(t)
	tests := []struct {
		name string
		edit func(*Artifact)
		want string
	}{
		{name: "license", edit: func(a *Artifact) { a.Sources[0].License.Boundary = "" }, want: "license boundary"},
		{name: "benchmark", edit: func(a *Artifact) { a.Benchmarks[0].Denominators.Cache = "" }, want: "denominators"},
		{name: "owner", edit: func(a *Artifact) { a.ExistingOwners = append(a.ExistingOwners, a.ExistingOwners[0]) }, want: "duplicate ownership"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			copyArtifact := deepCopy(t, base)
			tc.edit(&copyArtifact)
			if err := Validate(copyArtifact); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestPublicTextLeakCheckFailsClosed(t *testing.T) {
	for _, text := range []string{
		"private path C:" + string(rune(92)) + "Users" + string(rune(92)) + "person" + string(rune(92)) + "session.jsonl",
		strings.Repeat(string(rune(92)), 2) + "lab-node" + string(rune(92)) + "share",
		`authorization: bearer-secret-value`,
	} {
		if err := CheckPublicText("fixture", []byte(text)); err == nil {
			t.Fatalf("accepted private fixture %q", text)
		}
	}
}

func deepCopy(t *testing.T, in Artifact) Artifact {
	t.Helper()
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out Artifact
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}
