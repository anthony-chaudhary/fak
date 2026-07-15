package hooks

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func conceptAdmissionFixture(t *testing.T, line string, treatment string, headHas bool) *StagedDiff {
	t.Helper()
	root := t.TempDir()
	data := filepath.Join(root, "tools", "concept_disambiguation_scorecard.data")
	if err := os.MkdirAll(data, 0755); err != nil {
		t.Fatal(err)
	}
	meta := map[string]any{"families": []map[string]any{{"id": "cache", "roots": []string{"cache"}, "ignore": []string{}}}}
	if treatment == "classify" {
		meta["families"].([]map[string]any)[0]["ignore"] = []string{"CacheBurst"}
	}
	mb, _ := json.Marshal(meta)
	os.WriteFile(filepath.Join(data, "_meta.json"), mb, 0600)
	rows := []map[string]any{{"id": "cache-a", "family": "cache", "grounding": "CacheA"}}
	if treatment == "position" {
		rows = append(rows, map[string]any{"id": "cache-burst", "family": "cache", "grounding": "CacheBurst"})
	}
	rb, _ := json.Marshal(map[string]any{"rows": rows})
	os.WriteFile(filepath.Join(data, "rows-cache.json"), rb, 0600)
	d := diffOf(root, map[string][]string{"internal/demo/demo.go": {line}})
	d.ctx = context.Background()
	d.run = func(context.Context, string, ...string) (string, int, error) {
		if headHas {
			return "HEAD:internal/demo/demo.go:CacheBurst", 0, nil
		}
		return "", 1, nil
	}
	return d
}

func TestConceptAdmissionRejectsUntreatedTokenWithRepair(t *testing.T) {
	d := conceptAdmissionFixture(t, "const CacheBurst = 1", "", false)
	start := time.Now()
	got, err := gateConceptAdmission(d)
	elapsed := time.Since(start)
	if err != nil || len(got) != 1 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	detail := got[0].Detail
	for _, want := range []string{"family=cache", "token=CacheBurst", "internal/demo/demo.go:1", "fak concept position", "fak concept classify"} {
		if !strings.Contains(detail, want) {
			t.Fatalf("diagnostic missing %q: %s", want, detail)
		}
	}
	if elapsed > 100*time.Millisecond {
		t.Fatalf("diff-scoped gate latency %s exceeds 100ms fixture budget", elapsed)
	}
}

func TestConceptAdmissionTreatmentsPass(t *testing.T) {
	for _, treatment := range []string{"position", "classify"} {
		t.Run(treatment, func(t *testing.T) {
			got, err := gateConceptAdmission(conceptAdmissionFixture(t, "const CacheBurst = 1", treatment, false))
			if err != nil || len(got) != 0 {
				t.Fatalf("got=%+v err=%v", got, err)
			}
		})
	}
}

func TestConceptAdmissionRenameOrExistingUseIsNotAddition(t *testing.T) {
	got, err := gateConceptAdmission(conceptAdmissionFixture(t, "const CacheBurst = 1", "", true))
	if err != nil || len(got) != 0 {
		t.Fatalf("existing/renamed token falsely admitted: %+v %v", got, err)
	}
}

func TestConceptAdmissionRegisteredEarliestCommitGate(t *testing.T) {
	for _, g := range PreCommitGates() {
		if g.Name == "CONCEPT_ADMISSION" {
			return
		}
	}
	t.Fatal("CONCEPT_ADMISSION is not registered in pre-commit")
}

func gitFixture(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v %s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func TestConceptAdmissionCommittedRangeRepeatsCandidateDiff(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init")
	gitFixture(t, root, "config", "user.email", "fixture@example.com")
	gitFixture(t, root, "config", "user.name", "Fixture")
	data := filepath.Join(root, "tools", "concept_disambiguation_scorecard.data")
	os.MkdirAll(data, 0755)
	os.WriteFile(filepath.Join(data, "_meta.json"), []byte(`{"families":[{"id":"cache","roots":["cache"],"ignore":[]}]}`), 0600)
	os.WriteFile(filepath.Join(data, "rows-cache.json"), []byte(`{"rows":[{"id":"cache-a","family":"cache","grounding":"CacheA"}]}`), 0600)
	os.MkdirAll(filepath.Join(root, "internal", "demo"), 0755)
	os.WriteFile(filepath.Join(root, "internal", "demo", "demo.go"), []byte("package demo\nconst CacheA=1\n"), 0600)
	gitFixture(t, root, "add", ".")
	gitFixture(t, root, "commit", "-m", "base")
	base := gitFixture(t, root, "rev-parse", "HEAD")
	os.WriteFile(filepath.Join(root, "internal", "demo", "demo.go"), []byte("package demo\nconst CacheA=1\nconst CacheBurst=2\n"), 0600)
	gitFixture(t, root, "add", ".")
	gitFixture(t, root, "commit", "-m", "candidate")
	tip := gitFixture(t, root, "rev-parse", "HEAD")
	d, err := ReadRangeDiff(root, base, tip)
	if err != nil {
		t.Fatal(err)
	}
	got, err := CheckConceptAdmission(d)
	if err != nil || len(got) != 1 {
		t.Fatalf("range gate got=%+v err=%v", got, err)
	}
}
