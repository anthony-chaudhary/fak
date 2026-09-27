package witness

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// budgetContext is a context whose budget a fake runner expires on demand, so a kill
// mid-run is simulated deterministically instead of racing a wall clock on a loaded host.
type budgetContext struct {
	context.Context
	mu   sync.Mutex
	done chan struct{}
	err  error
}

func newBudgetContext() *budgetContext {
	return &budgetContext{Context: context.Background(), done: make(chan struct{})}
}

func (c *budgetContext) Done() <-chan struct{} { return c.done }

func (c *budgetContext) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

func (c *budgetContext) expire(cause error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err == nil {
		c.err = cause
		close(c.done)
	}
}

// midCompileKillOutput is what a `go test -json` killed during compilation leaves behind:
// a truncated build stream with no `run` event and no build-failure marker.
const midCompileKillOutput = "{\"ImportPath\":\"m [m.test]\",\"Action\":\"build-out"

// killCase models how exec.CommandContext reports a context-killed process: Cmd.Wait
// returns an ordinary *exec.ExitError, so commandRunner yields (partial output, code,
// err == nil). Windows TerminateProcess exits 1; a Unix signal kill reports -1.
var killCases = []struct {
	name  string
	cause error
	code  int
}{
	{"deadline exceeded, windows terminate", context.DeadlineExceeded, 1},
	{"canceled, unix signal", context.Canceled, -1},
}

func TestRunSelectedGoTestsContextKillMidCompileIsTimeout(t *testing.T) {
	for _, kc := range killCases {
		t.Run(kc.name, func(t *testing.T) {
			ctx := newBudgetContext()
			run := func(context.Context, string, ...string) (string, int, error) {
				ctx.expire(kc.cause)
				return midCompileKillOutput, kc.code, nil
			}
			got := runSelectedGoTests(ctx, run, t.TempDir(), []string{"./."}, nil, []string{"^TestSelected$"})
			if !got.timedOut {
				t.Fatalf("context-killed run must report timedOut, got %+v", got)
			}
			if got.matched || got.selectedFailed || got.passed {
				t.Fatalf("a killed run carries no test evidence, got %+v", got)
			}
		})
	}
}

// TestSymptomBudgetKillMidCompileAbstainsAsTimeout is the land-refusal regression: the
// budget expiring while `go test` compiles must abstain as a timeout at either ref, never
// refute the witness as "symptom selector matched no executed test".
func TestSymptomBudgetKillMidCompileAbstainsAsTimeout(t *testing.T) {
	requireGoAndGit(t)
	dir := newGoModuleRepo(t)
	writeRepoFile(t, dir, "value.go", "package m\n\nfunc Value() int { return 0 }\n")
	gitIn(t, dir, "add", "value.go")
	gitIn(t, dir, "commit", "-q", "-m", "parent")
	writeRepoFile(t, dir, "value.go", "package m\n\nfunc Value() int { return 1 }\n")
	writeRepoFile(t, dir, "value_test.go", `package m
import "testing"
func TestSelected(t *testing.T) { if Value() != 1 { t.Fatal(Value()) } }
`)
	gitIn(t, dir, "add", "value.go", "value_test.go")
	gitIn(t, dir, "commit", "-q", "-m", "fix(m): value")

	pass := "{\"Action\":\"run\",\"Package\":\"m\",\"Test\":\"TestSelected\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"m\",\"Test\":\"TestSelected\"}\n"
	for _, side := range []struct {
		name   string
		killAt int
	}{
		{"candidate", 1},
		{"parent", 2},
	} {
		for _, kc := range killCases {
			t.Run(side.name+"/"+kc.name, func(t *testing.T) {
				ctx := newBudgetContext()
				calls := 0
				exec := func(_ context.Context, _ string, argv ...string) (string, int, error) {
					calls++
					if calls == side.killAt {
						ctx.expire(kc.cause)
						return midCompileKillOutput, kc.code, nil
					}
					return pass, 0, nil
				}
				got, detail := NewWithRunners(gitRunner, exec, dir).
					WithSymptomTests([]string{"^TestSelected$"}).
					ResolveSymptomWithDetail(ctx, "HEAD", true)
				if got != abi.WitnessAbstain {
					t.Fatalf("outcome=%v detail=%q, want ABSTAIN (a killed run is unproven, never refuted)", got, detail)
				}
				if !strings.Contains(detail, "timed out") || strings.Contains(detail, "matched no") {
					t.Fatalf("detail=%q, want a timeout detail, not a zero-match refutation", detail)
				}
				if calls != side.killAt {
					t.Fatalf("executor calls=%d, want the resolution to stop at the killed %s run", calls, side.name)
				}
			})
		}
	}
	assertRepoClean(t, dir)
}

// TestSymptomBudgetKillAtParentNeverConfirmsPython guards the false-CONFIRM mirror: a
// killed parent run exits non-zero, which the Python-only path would otherwise read as RED.
func TestSymptomBudgetKillAtParentNeverConfirmsPython(t *testing.T) {
	dir := newExecutionRepo(t)
	writeRepoPath(t, dir, "tools/calc.py", "def compute(n):\n    return 0\n")
	gitIn(t, dir, "add", "tools/calc.py")
	gitIn(t, dir, "commit", "-q", "-m", "parent")
	writeRepoPath(t, dir, "tools/calc.py", "def compute(n):\n    return -1 if n < 0 else 0\n")
	writeRepoPath(t, dir, "tools/calc_test.py", "import calc\nassert calc.compute(-1) == -1\n")
	gitIn(t, dir, "add", "tools/calc.py", "tools/calc_test.py")
	gitIn(t, dir, "commit", "-q", "-m", "fix(tools): negatives")

	ctx := newBudgetContext()
	calls := 0
	exec := func(context.Context, string, ...string) (string, int, error) {
		calls++
		if calls == 2 {
			ctx.expire(context.DeadlineExceeded)
			return "", 1, nil
		}
		return "", 0, nil
	}
	got, detail := NewWithRunners(gitRunner, exec, dir).ResolveSymptomWithDetail(ctx, "HEAD", true)
	if got != abi.WitnessAbstain || !strings.Contains(detail, "timed out") {
		t.Fatalf("outcome=%v detail=%q, want ABSTAIN timeout (a killed parent run is not a red)", got, detail)
	}
	if calls != 2 {
		t.Fatalf("executor calls=%d, want candidate and parent", calls)
	}
	assertRepoClean(t, dir)
}
