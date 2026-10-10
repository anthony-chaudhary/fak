package taskrun

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
)

const fixtureControlsOptInEnv = "FAK_AGENTBENCH_FIXTURE_CONTROLS"

// fak-test:runtime fast est=50ms lane=default
func TestEveryFixtureKnownFixStaysInsideOneFunctionBody(t *testing.T) {
	fixtures, err := taskfixture.Suite(taskfixture.SuiteExtended, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fixtures {
		if err := validateCandidateSource(f.BrokenSource, f.FixedSource); err != nil {
			t.Fatalf("%s: known fix violates the candidate boundary: %v", f.ID, err)
		}
	}
}

// fak-test:runtime slow est=9m lane=optin
// Runtime justification: four sandboxed cold `go test` runs per fixture, the
// same visible-control and hidden-verifier paths runOne uses before any model call.
func TestEveryFixtureProvesRedGreenInSandbox(t *testing.T) {
	if os.Getenv(fixtureControlsOptInEnv) != "1" {
		t.Skipf("set %s=1 to run sandboxed red/green controls for every fixture", fixtureControlsOptInEnv)
	}
	requireTaskSandbox(t)
	fixtures, err := taskfixture.Suite(taskfixture.SuiteExtended, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fixtures {
		f := f
		t.Run(f.ID, func(t *testing.T) {
			t.Parallel()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			for _, tc := range []struct {
				name, source string
				pass         bool
			}{{"broken", f.BrokenSource, false}, {"fixed", f.FixedSource, true}} {
				if got := fixtureVisiblePasses(ctx, t, f, tc.source); got != tc.pass {
					t.Fatalf("%s source: visible test passed=%t, want %t", tc.name, got, tc.pass)
				}
				if got := fixtureOraclePasses(ctx, t, f, tc.source); got != tc.pass {
					t.Fatalf("%s source: hidden oracle passed=%t, want %t", tc.name, got, tc.pass)
				}
			}
		})
	}
}

func fixtureVisiblePasses(ctx context.Context, t *testing.T, f taskfixture.Fixture, source string) bool {
	t.Helper()
	root := t.TempDir()
	candidate, visible, oracle := filepath.Join(root, "candidate"), filepath.Join(root, "visible"), filepath.Join(root, "oracle")
	for _, dir := range []string{candidate, visible, oracle} {
		mustMkdir(t, dir)
	}
	if err := seedFixture(candidate, f, source, false); err != nil {
		t.Fatal(err)
	}
	mustWrite(t, filepath.Join(visible, "visible_test.go"), f.VisibleTest)
	result, err := runCandidateTests(ctx, candidate, f.TargetFile, visible, oracle)
	if err != nil {
		t.Fatalf("visible control: %v", err)
	}
	return result.ExitCode == 0
}

func fixtureOraclePasses(ctx context.Context, t *testing.T, f taskfixture.Fixture, source string) bool {
	t.Helper()
	root := t.TempDir()
	verify, oracle := filepath.Join(root, "verify"), filepath.Join(root, "oracle")
	mustMkdir(t, oracle)
	if err := seedFixture(verify, f, source, true); err != nil {
		t.Fatal(err)
	}
	result, err := runSandboxedTests(ctx, verify, oracle, false)
	if err != nil {
		t.Fatalf("hidden verifier: %v", err)
	}
	return result.ExitCode == 0
}
