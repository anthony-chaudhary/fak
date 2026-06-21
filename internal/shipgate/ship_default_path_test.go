package shipgate_test

// This is the issue-#11 witness: the dev-agent floor + the CICD pillars
// (shipgate / witness / plancfi) wired on the REAL default driver chain — not a
// test fake. It blank-imports internal/registrations (the defconfig), so every
// adjudicator the production binary loads is registered, then drives calls through
// a real kernel.Syscall and asserts the four headline behaviors:
//
//   1. a kernel self-modify is DENIED with disposition ESCALATE;
//   2. an unwitnessed ship is REFUSED (UNWITNESSED);
//   3. a git-corroborated ship is ALLOWED (the witness resolver confirms from real
//      git evidence the agent did not author);
//   4. RequireApproval is actually EMITTED by the registered plancfi adjudicator on
//      a plan deviation.
//
// External test package so it can import registrations (which imports shipgate)
// without an import cycle.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/plancfi"
	_ "github.com/anthony-chaudhary/fak/internal/registrations" // the real default driver chain
)

func inline(s string) abi.Ref { return abi.Ref{Kind: abi.RefInline, Inline: []byte(s)} }

// withDevAgentFloor swaps the registered monitor's policy to the dev-agent preset
// for the duration of a test, restoring the bench default afterward.
func withDevAgentFloor(t *testing.T) {
	t.Helper()
	adjudicator.Default.SetPolicy(adjudicator.DevAgentPolicy())
	t.Cleanup(func() { adjudicator.Default.SetPolicy(adjudicator.DefaultPolicy()) })
}

func TestDevAgentDefaultPath(t *testing.T) {
	ctx := context.Background()
	withDevAgentFloor(t)
	k := kernel.New("mock") // the mock engine ships in the defconfig
	k.SetVDSO(false)        // every call must actually be adjudicated, no fast-path

	t.Run("kernel self-modify is denied (ESCALATE)", func(t *testing.T) {
		c := &abi.ToolCall{Tool: "write_file", Args: inline(`{"path":"internal/kernel/kernel.go"}`)}
		r, v := k.Syscall(ctx, c)
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
			t.Fatalf("verdict = %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
		}
		if r.Meta["disposition"] != "ESCALATE" {
			t.Fatalf("disposition = %q, want ESCALATE", r.Meta["disposition"])
		}
	})

	t.Run("unwitnessed ship is refused", func(t *testing.T) {
		// ship_release is allowed at the floor; shipgate lifts it to RequireWitness.
		// With NO claim attached the witness fold abstains => fail-closed UNWITNESSED.
		c := &abi.ToolCall{Tool: "ship_release", Args: inline(`{}`)}
		r, v := k.Syscall(ctx, c)
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonUnwitnessed {
			t.Fatalf("verdict = %v/%s, want Deny/UNWITNESSED", v.Kind, abi.ReasonName(v.Reason))
		}
		if r.Meta["reason"] != "UNWITNESSED" {
			t.Fatalf("deny result reason = %q, want UNWITNESSED", r.Meta["reason"])
		}
	})

	t.Run("git-corroborated ship is allowed", func(t *testing.T) {
		// The claim "ancestor:HEAD" is corroborated by the REAL witness resolver
		// against real git (HEAD is reachable from HEAD) — the agent did not author
		// this evidence. A corroborated ship dispatches to the engine.
		c := &abi.ToolCall{Tool: "ship_release", Args: inline(`{}`),
			Meta: map[string]string{"witness": "ancestor:HEAD"}}
		r, v := k.Syscall(ctx, c)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("verdict = %v/%s, want Allow (git-corroborated ship)", v.Kind, abi.ReasonName(v.Reason))
		}
		if v.By != "witness" {
			t.Fatalf("allow By = %q, want witness (the corroboration opened the gate)", v.By)
		}
		if r == nil || r.Status != abi.StatusOK {
			t.Fatalf("corroborated ship must dispatch + admit, got result %+v", r)
		}
	})

	t.Run("RequireApproval is emitted on a plan deviation", func(t *testing.T) {
		trace := "devagent-plan-1"
		// Approve a plan whose only step is git_status, then deviate to git_diff
		// (allowed by the floor, so the monitor does not mask the escalation).
		plancfi.Default.Declare(trace, plancfi.Plan{Tools: []string{"git_status"}, Mode: plancfi.AllowedSet})
		t.Cleanup(func() { plancfi.Default.Clear(trace) })

		c := &abi.ToolCall{Tool: "git_diff", TraceID: trace, Args: inline(`{}`)}
		r, v := k.Syscall(ctx, c)
		if v.Kind != plancfi.VerdictRequireApproval {
			t.Fatalf("verdict kind = %d, want RequireApproval (%d) from the registered plancfi rung",
				v.Kind, plancfi.VerdictRequireApproval)
		}
		if r.Meta["disposition"] != "ESCALATE" {
			t.Fatalf("disposition = %q, want ESCALATE", r.Meta["disposition"])
		}
	})
}
