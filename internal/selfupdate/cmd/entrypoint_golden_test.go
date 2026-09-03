package selfupdatecmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

type entryFixture struct {
	Freshness binstamp.Freshness
	Audit     selfinstall.AuditPartition
	Outcome   string
}

type entryFixtureGolden struct {
	Decision selfupdate.CheckPosture `json:"decision"`
	Receipt  selfUpdateReceipt       `json:"receipt"`
}

func entryFixtureBytes(f entryFixture) ([]byte, error) {
	posture := selfupdate.ClassifyCheck(f.Freshness, f.Audit)
	outcome := outcomeCheckOnly
	switch f.Outcome {
	case "", "check":
	case "rollback":
		outcome = outcomeRolledBack
	default:
		return nil, fmt.Errorf("unknown entry fixture outcome %q", f.Outcome)
	}
	receipt := newSelfUpdateReceipt(outcome, "fixture/fak", string(posture.Status))
	receipt.CorrelationID = "fixture"
	receipt.NextCommand = posture.NextCommand
	return json.Marshal(entryFixtureGolden{Decision: posture, Receipt: receipt})
}

func TestExecutableAdaptersShareGoldenDecisionsAndReceipts(t *testing.T) {
	repairable := selfinstall.Audit{Divergent: []selfinstall.Role{selfinstall.RoleWorker}}.Partition()
	attention := selfinstall.Audit{Dirty: []selfinstall.Role{selfinstall.RoleGate}}.Partition()
	fixtures := []struct {
		name string
		in   entryFixture
	}{
		{"current", entryFixture{Freshness: binstamp.Fresh}},
		{"repairable-drift", entryFixture{Freshness: binstamp.Fresh, Audit: repairable}},
		{"audit-only-attention", entryFixture{Freshness: binstamp.Fresh, Audit: attention}},
		{"rollback", entryFixture{Freshness: binstamp.Fresh, Outcome: "rollback"}},
	}
	for _, tc := range fixtures {
		t.Run(tc.name, func(t *testing.T) {
			got, err := entryFixtureBytes(tc.in)
			if err != nil {
				t.Fatal(err)
			}
			golden := filepath.Join("testdata", "entry-"+tc.name+".golden.json")
			want, err := os.ReadFile(golden)
			if err != nil {
				t.Fatalf("read golden: %v", err)
			}
			if string(got) != strings.TrimSpace(string(want)) {
				t.Fatalf("golden mismatch\n got: %s\nwant: %s", got, want)
			}
		})
	}

	root := findRepoRoot(t)
	for _, rel := range []string{filepath.Join("cmd", "fak", "selfupdate.go"), filepath.Join("cmd", "fak-selfupdate", "main.go")} {
		body, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		if !strings.Contains(string(body), "selfupdatecmd.Run(") {
			t.Fatalf("%s does not invoke shared Run", rel)
		}
	}
}

func TestStandaloneCheckInspectsTargetWithoutExecutingIt(t *testing.T) {
	root := findRepoRoot(t)
	standalone := filepath.Join(t.TempDir(), "fak-selfupdate"+exeSuffix())
	build := exec.Command("go", "build", "-o", standalone, "./cmd/fak-selfupdate")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build standalone: %v\n%s", err, out)
	}

	marker := filepath.Join(t.TempDir(), "target-executed")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	check := exec.CommandContext(ctx, standalone, "--check", "--json", "--root", root, "--target", os.Args[0])
	check.Env = append(os.Environ(), selfUpdateProbeHelperEnv+"=1", selfUpdateProbeMarkerEnv+"="+marker)
	if out, err := check.CombinedOutput(); err != nil {
		t.Fatalf("standalone check: %v\n%s", err, out)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("standalone inspection executed stale target; marker err=%v", err)
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
