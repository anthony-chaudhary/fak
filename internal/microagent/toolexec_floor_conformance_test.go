package microagent_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// This file is the #2018 acceptance suite for the ToolExec seam: ONE
// adjudication test suite that runs against ALL registered backends, proving a
// policy-denied action is blocked in every one — the in-process kernel
// adjudication floor holds at EVERY isolation level, not just the trusted
// goroutine path.
//
// It is the microagent-lane complement of the policy-lane witness
// (internal/policy/isolation_allbackends_conformance_test.go, which proves the
// POLICY floor is orthogonal to the isolation dial). Here the witness is the
// RUNTIME seam: a denied action never reaches Backend.Dispatch, whatever the
// backend.

// spyBackend records whether the seam ever dispatched to it. It is the
// structural witness that adjudication happens BEFORE dispatch: on a denied
// action the call count must stay zero.
type spyBackend struct {
	calls int32
}

func (s *spyBackend) Name() string { return "spy" }

func (s *spyBackend) Dispatch(ctx context.Context, act microagent.ToolAction) (microagent.ToolResult, error) {
	atomic.AddInt32(&s.calls, 1)
	return microagent.ToolResult{Ran: true}, nil
}

// TestToolExecSeamAdjudicatesBeforeDispatch proves the seam property #2018
// bullet 1 names: kernel adjudication fires BEFORE dispatch, for ANY backend —
// the backend is only ever handed an already-allowed action. The spy counts
// Dispatch calls; a denied action must leave the count at zero.
func TestToolExecSeamAdjudicatesBeforeDispatch(t *testing.T) {
	t.Run("denied action never reaches Dispatch", func(t *testing.T) {
		spy := &spyBackend{}
		te, err := microagent.NewToolExecBackend(denyKernel(), spy)
		if err != nil {
			t.Fatalf("NewToolExecBackend: %v", err)
		}
		res, err := te.Run(context.Background(), microagent.ToolAction{Tool: "run_shell"})
		if !errors.Is(err, microagent.ErrActionDenied) {
			t.Fatalf("Run on a denied action = %v, want ErrActionDenied", err)
		}
		if got := atomic.LoadInt32(&spy.calls); got != 0 {
			t.Fatalf("Dispatch was called %d time(s) on a DENIED action — adjudication did not gate dispatch", got)
		}
		if res.Ran {
			t.Fatal("Ran=true on a denied action")
		}
		if res.Verdict.Kind != abi.VerdictDeny {
			t.Errorf("verdict kind = %v, want VerdictDeny", res.Verdict.Kind)
		}
	})
	t.Run("allowed action dispatches exactly once", func(t *testing.T) {
		spy := &spyBackend{}
		te, err := microagent.NewToolExecBackend(allowKernel("run_shell"), spy)
		if err != nil {
			t.Fatalf("NewToolExecBackend: %v", err)
		}
		res, err := te.Run(context.Background(), microagent.ToolAction{Tool: "run_shell"})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if got := atomic.LoadInt32(&spy.calls); got != 1 {
			t.Fatalf("Dispatch called %d time(s) on an allowed action, want exactly 1", got)
		}
		if res.Verdict.Kind != abi.VerdictAllow {
			t.Errorf("verdict kind = %v, want VerdictAllow (the seam must stamp the floor verdict on the result)", res.Verdict.Kind)
		}
	})
}

// TestAdjudicationFloorBlocksDeniedActionAcrossAllRegisteredBackends is the
// literal #2018 acceptance: iterate EVERY backend in the registry and prove a
// policy-denied action is blocked in every one. The registered vocabulary is
// pinned exactly, so a newly-registered backend TRIPS this suite until its
// floor coverage is added here — same discipline as the policy-lane pin.
func TestAdjudicationFloorBlocksDeniedActionAcrossAllRegisteredBackends(t *testing.T) {
	want := []string{"goroutine", "subprocess"}
	got := microagent.RegisteredBackends()
	sort.Strings(got)
	if len(got) != len(want) {
		t.Fatalf("registered backends = %v, want exactly %v — a new backend must add its floor-conformance coverage to this suite before registering", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("registered backends = %v, want exactly %v — a new backend must add its floor-conformance coverage to this suite before registering", got, want)
		}
	}

	for _, name := range got {
		t.Run(name, func(t *testing.T) {
			te, err := microagent.NewToolExecFor(name, denyKernel())
			if err != nil {
				t.Fatalf("NewToolExecFor(%q): %v", name, err)
			}
			// A spawn (subprocess) would write this marker via the re-exec
			// helper; its absence is the no-exec witness.
			marker := filepath.Join(t.TempDir(), "should-not-exist")
			t.Setenv(helperModeEnv, "grandchild")
			t.Setenv(helperMarkerEnv, marker)
			act := microagent.ToolAction{
				Tool:    "run_shell", // not in the empty policy's allow set -> default deny
				Path:    os.Args[0],
				Argv:    []string{"-test.run=^TestHelperProcess$"},
				Timeout: 5 * time.Second,
			}
			res, err := te.Run(context.Background(), act)
			if !errors.Is(err, microagent.ErrActionDenied) {
				t.Fatalf("backend %q: Run on a denied action = %v, want ErrActionDenied (denial must fire BEFORE dispatch — an unknown-tool or exec error means dispatch was reached)", name, err)
			}
			if res.Ran {
				t.Fatalf("backend %q: Ran=true on a denied action", name)
			}
			if res.Verdict.Kind != abi.VerdictDeny {
				t.Errorf("backend %q: verdict kind = %v, want VerdictDeny", name, res.Verdict.Kind)
			}
			if _, statErr := os.Stat(marker); statErr == nil {
				t.Fatalf("backend %q: marker exists — the denied action executed anyway", name)
			}
		})
	}
}

// TestGoroutineBackendFloorGatesInProcessFunc proves the floor holds at the
// TRUSTED pole too: an in-process Go function — the cheapest, most tempting
// place to skip adjudication — is still gated. Denied: the func never runs.
// Allowed: it runs in-process and its output is captured.
func TestGoroutineBackendFloorGatesInProcessFunc(t *testing.T) {
	newBackend := func(executed *int32) *microagent.GoroutineBackend {
		gb := microagent.NewGoroutineBackend()
		err := gb.Register("run_shell", func(ctx context.Context, act microagent.ToolAction) (microagent.ToolResult, error) {
			atomic.AddInt32(executed, 1)
			return microagent.ToolResult{Stdout: []byte("in-process-out"), ExitCode: 0}, nil
		})
		if err != nil {
			t.Fatalf("Register: %v", err)
		}
		return gb
	}

	t.Run("denied func never executes", func(t *testing.T) {
		var executed int32
		te, err := microagent.NewToolExecBackend(denyKernel(), newBackend(&executed))
		if err != nil {
			t.Fatalf("NewToolExecBackend: %v", err)
		}
		_, err = te.Run(context.Background(), microagent.ToolAction{Tool: "run_shell"})
		if !errors.Is(err, microagent.ErrActionDenied) {
			t.Fatalf("Run on a denied action = %v, want ErrActionDenied", err)
		}
		if atomic.LoadInt32(&executed) != 0 {
			t.Fatal("the denied in-process func EXECUTED — the goroutine tier skipped the floor")
		}
	})
	t.Run("allowed func runs and is captured", func(t *testing.T) {
		var executed int32
		te, err := microagent.NewToolExecBackend(allowKernel("run_shell"), newBackend(&executed))
		if err != nil {
			t.Fatalf("NewToolExecBackend: %v", err)
		}
		res, err := te.Run(context.Background(), microagent.ToolAction{Tool: "run_shell"})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if atomic.LoadInt32(&executed) != 1 {
			t.Fatalf("func executed %d time(s), want exactly 1", executed)
		}
		if !res.Ran {
			t.Fatal("Ran=false on an allowed in-process action")
		}
		if string(res.Stdout) != "in-process-out" {
			t.Errorf("stdout = %q, want %q", res.Stdout, "in-process-out")
		}
		if res.Verdict.Kind != abi.VerdictAllow {
			t.Errorf("verdict kind = %v, want VerdictAllow", res.Verdict.Kind)
		}
	})
	t.Run("allowed but unregistered tool refuses AFTER the floor", func(t *testing.T) {
		// The ordering discriminator: ErrNoGoTool can only surface once the
		// floor has allowed the action — a denied action must yield
		// ErrActionDenied instead (asserted above), proving denial gates first.
		te, err := microagent.NewToolExecBackend(allowKernel("run_shell"), microagent.NewGoroutineBackend())
		if err != nil {
			t.Fatalf("NewToolExecBackend: %v", err)
		}
		res, err := te.Run(context.Background(), microagent.ToolAction{Tool: "run_shell"})
		if !errors.Is(err, microagent.ErrNoGoTool) {
			t.Fatalf("Run with no registered func = %v, want ErrNoGoTool", err)
		}
		if res.Ran {
			t.Fatal("Ran=true with no registered func")
		}
	})
}

// TestToolExecBackendRefusals pins the fail-closed construction edges: there is
// no way to build a runnable executor without the floor, no way to reach a
// backend the registry does not know, and no silent double-registration.
func TestToolExecBackendRefusals(t *testing.T) {
	gb := microagent.NewGoroutineBackend()
	if _, err := microagent.NewToolExecBackend(nil, gb); !errors.Is(err, microagent.ErrNilFloor) {
		t.Fatalf("NewToolExecBackend(nil floor) = %v, want ErrNilFloor", err)
	}
	if _, err := microagent.NewToolExecBackend(denyKernel(), nil); !errors.Is(err, microagent.ErrNilBackend) {
		t.Fatalf("NewToolExecBackend(nil backend) = %v, want ErrNilBackend", err)
	}
	if _, err := microagent.NewToolExecFor("no-such-backend", denyKernel()); !errors.Is(err, microagent.ErrUnknownBackend) {
		t.Fatalf("NewToolExecFor(unknown) = %v, want ErrUnknownBackend", err)
	}
	if _, err := microagent.NewToolExecFor("goroutine", nil); !errors.Is(err, microagent.ErrNilFloor) {
		t.Fatalf("NewToolExecFor(nil floor) = %v, want ErrNilFloor — the registry must never issue an unadjudicated executor", err)
	}
	if err := microagent.RegisterBackend("subprocess", func() microagent.Backend { return &spyBackend{} }); err == nil {
		t.Fatal("RegisterBackend(duplicate name) succeeded, want a refusal")
	}
	if err := microagent.RegisterBackend("", func() microagent.Backend { return &spyBackend{} }); err == nil {
		t.Fatal("RegisterBackend(empty name) succeeded, want a refusal")
	}
	if err := microagent.RegisterBackend("nil-ctor", nil); err == nil {
		t.Fatal("RegisterBackend(nil constructor) succeeded, want a refusal")
	}
	if err := gb.Register("", nil); err == nil {
		t.Fatal("GoroutineBackend.Register(empty tool) succeeded, want a refusal")
	}
	if err := gb.Register("t", nil); err == nil {
		t.Fatal("GoroutineBackend.Register(nil func) succeeded, want a refusal")
	}
	if err := gb.Register("t", func(ctx context.Context, act microagent.ToolAction) (microagent.ToolResult, error) {
		return microagent.ToolResult{}, nil
	}); err != nil {
		t.Fatalf("GoroutineBackend.Register: %v", err)
	}
	if err := gb.Register("t", func(ctx context.Context, act microagent.ToolAction) (microagent.ToolResult, error) {
		return microagent.ToolResult{}, nil
	}); err == nil {
		t.Fatal("GoroutineBackend.Register(duplicate tool) succeeded, want a refusal")
	}
}
