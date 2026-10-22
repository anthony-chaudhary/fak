package main

import (
	"bytes"
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestRunSyncLocalOnlyRejectsUnsafeCombinations(t *testing.T) {
	for _, args := range [][]string{
		{"push", "--local-only"}, {"apply", "--local-only"},
		{"reconcile", "--local-only"}, {"reconcile", "--execute", "--apply", "--local-only"},
		{"execute", "--apply", "--local-only"},
	} {
		var out, errb bytes.Buffer
		if code := runSync(&out, &errb, args); code != syncExitUsage {
			t.Fatalf("%v: exit=%d stderr=%s", args, code, errb.String())
		}
	}
}

func TestRunSyncLocalOnlyPropagation(t *testing.T) {
	oldPacket, oldExecute, oldRoute := syncBuildReconciliationPacket, syncExecutePacket, syncRouteReconciliation
	t.Cleanup(func() {
		syncBuildReconciliationPacket, syncExecutePacket, syncRouteReconciliation = oldPacket, oldExecute, oldRoute
	})
	syncBuildReconciliationPacket = func(context.Context, safesync.PacketOptions) (*safesync.ReconciliationPacket, error) {
		return &safesync.ReconciliationPacket{}, nil
	}
	for _, local := range []bool{false, true} {
		called := false
		syncExecutePacket = func(_ context.Context, _ *safesync.ReconciliationPacket, opts safesync.ExecuteOptions) (*safesync.ExecutionReceipt, error) {
			called = true
			if opts.LocalOnly != local {
				t.Errorf("LocalOnly=%v want=%v", opts.LocalOnly, local)
			}
			return &safesync.ExecutionReceipt{Status: safesync.ExecuteStatusExecuted, LocalOnly: local, Pushed: !local}, nil
		}
		args := []string{"execute", "--json"}
		if local {
			args = append(args, "--local-only")
		}
		var out, errb bytes.Buffer
		if code := runSync(&out, &errb, args); code != syncExitOK || !called {
			t.Fatalf("exit=%d called=%v stderr=%s", code, called, errb.String())
		}
		called = false
		syncRouteReconciliation = func(_ context.Context, opts safesync.ReconcileOptions) (safesync.ReconcileAssessment, error) {
			called = true
			if opts.LocalOnly != local {
				t.Errorf("LocalOnly=%v want=%v", opts.LocalOnly, local)
			}
			return safesync.ReconcileAssessment{ExecuteReceipt: &safesync.ExecutionReceipt{Status: safesync.ExecuteStatusExecuted}}, nil
		}
		args = []string{"reconcile", "--execute", "--json"}
		if local {
			args = append(args, "--local-only")
		}
		out.Reset()
		errb.Reset()
		if code := runSync(&out, &errb, args); code != syncExitOK || !called {
			t.Fatalf("exit=%d called=%v stderr=%s", code, called, errb.String())
		}
	}
}
