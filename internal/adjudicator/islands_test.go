package adjudicator

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestAdjudicatorIslandsWiring(t *testing.T) {
	ctx := context.Background()
	adj := New(Policy{
		Allow: map[string]bool{"Bash": true, "bash": true},
	})

	t.Run("LifecycleFSM and RecoveryAuditLedger wired into Adjudicate", func(t *testing.T) {
		// Verify AdjudicateWithFSM advances through LifecycleFSM
		call := inlineCall("Bash", `{"command":"echo hello"}`)
		call.TraceID = "sess-123"
		call.Meta = map[string]string{"turn": "1"}

		verdict, fsm := adj.AdjudicateWithFSM(ctx, call)
		if fsm == nil {
			t.Fatal("expected non-nil LifecycleFSM from AdjudicateWithFSM")
		}
		if verdict.Kind != abi.VerdictAllow {
			t.Fatalf("expected VerdictAllow for echo, got %v", verdict.Kind)
		}
		if !fsm.IsTerminal() {
			t.Fatalf("expected fsm to be terminal, got state: %s", fsm.State)
		}
		if len(fsm.History) < 2 {
			t.Fatalf("expected at least 2 transitions in FSM history, got: %d", len(fsm.History))
		}
		if fsm.History[0].Event != EventEvaluate {
			t.Fatalf("expected first event to be EventEvaluate, got %s", fsm.History[0].Event)
		}

		// Verify RecoveryAuditLedger is active
		ledger := adj.RecoveryLedger()
		if ledger == nil {
			t.Fatal("expected non-nil RecoveryAuditLedger from Adjudicator")
		}
		if ledger.Len() == 0 {
			t.Fatal("expected RecoveryAuditLedger to record entries")
		}

		// Test refusal recording in ledger
		deniedCall := inlineCall("rm_rf", `{}`)
		deniedCall.TraceID = "sess-123"
		deniedCall.Meta = map[string]string{"turn": "2"}
		adj.SetPolicy(Policy{
			Deny: map[string]abi.ReasonCode{
				"rm_rf": abi.ReasonPolicyBlock,
			},
		})
		vDeny, fsmDeny := adj.AdjudicateWithFSM(ctx, deniedCall)
		if vDeny.Kind != abi.VerdictDeny {
			t.Fatalf("expected VerdictDeny, got %v", vDeny.Kind)
		}
		if fsmDeny.State != StateFinished {
			t.Fatalf("expected finished state, got %s", fsmDeny.State)
		}
		hasDenyEvent := false
		for _, tr := range fsmDeny.History {
			if tr.Event == EventDeny {
				hasDenyEvent = true
				break
			}
		}
		if !hasDenyEvent {
			t.Fatal("expected EventDeny in FSM history")
		}

		if ledger.Len() < 2 {
			t.Fatalf("expected at least 2 ledger entries, got %d", ledger.Len())
		}
	})

	t.Run("HeredocTargets and ParseBashWriteTargets wired into commandWriteTargets", func(t *testing.T) {
		// Heredoc write target
		heredocCmd := "cat << 'EOF' > /tmp/heredoc_test.txt\nsome data\nEOF"
		targets := commandWriteTargets(heredocCmd)
		foundHeredoc := false
		for _, tgt := range targets {
			if strings.Contains(tgt, "heredoc_test.txt") {
				foundHeredoc = true
				break
			}
		}
		if !foundHeredoc {
			t.Fatalf("commandWriteTargets failed to detect heredoc target, got: %v", targets)
		}

		// Subshell AST write target
		subshellCmd := "(echo a; echo b) > /tmp/subshell_ast_test.txt"
		targets = commandWriteTargets(subshellCmd)
		foundSubshell := false
		for _, tgt := range targets {
			if strings.Contains(tgt, "subshell_ast_test.txt") {
				foundSubshell = true
				break
			}
		}
		if !foundSubshell {
			t.Fatalf("commandWriteTargets failed to detect subshell AST write target, got: %v", targets)
		}
	})

	t.Run("CheckShellEditNudge wired into Adjudicate advisoryNotes", func(t *testing.T) {
		adj.SetPolicy(Policy{
			Allow: map[string]bool{"bash": true},
		})
		shellEditCall := inlineCall("bash", `{"command":"sed -i 's/foo/bar/g' file.txt"}`)
		v := adj.Adjudicate(ctx, shellEditCall)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("expected VerdictAllow for bash, got %v", v.Kind)
		}
		adv := v.Meta["advisory_violations"]
		if !strings.Contains(adv, "Prefer structured tool 'Edit' or 'Write'") {
			t.Fatalf("expected shell edit nudge in advisory_violations, got: %q", adv)
		}
	})
}
