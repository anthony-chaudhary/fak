package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

// fak-test:runtime fast est=2s
func TestNativeCodeToolsSetupAfterCompletedRegistryReset(t *testing.T) {
	root := t.TempDir()
	const granted = "echo registry-rearm-owned"
	const ungranted = "echo registry-rearm-ungranted"
	opts := agent.CodeToolsOptions{
		Root:                 root,
		Focused:              true,
		ExactAllowedCommands: []string{granted},
	}
	if _, err := agent.ArmCodeToolsWithOptions(opts); err != nil {
		t.Fatalf("initial code-tool setup: %v", err)
	}
	t.Cleanup(func() {
		agent.DisarmCodeTools()
		agent.Configure()
	})

	// A completed reset removes the earlier setup's registrations. Every setup
	// below uses the same workspace, and no reset races with a request.
	abi.ResetForTest()
	if len(abi.Adjudicators()) != 0 || abi.Engine(codetools.EngineBash) != nil {
		t.Fatal("registry reset did not clear the controlled driver set")
	}

	cfg := Config{
		NativeCodeWorkspace:        root,
		NativeExactAllowedCommands: []string{granted},
	}
	t.Run("owned_commands", func(t *testing.T) {
		// Exercise New -> native HTTP handler -> owned loop -> actual Bash engine.
		// A successful transport response alone would also accept unknown-tool
		// failures, so require the command's structured execution result.
		results, _ := runNativeExactCommands(t, cfg, granted, ungranted)
		var executed codetools.BashResult
		if err := json.Unmarshal([]byte(results[0]), &executed); err != nil {
			t.Errorf("decode granted command result: %v; result=%q", err, results[0])
		} else if strings.TrimSpace(executed.Stdout) != "registry-rearm-owned" ||
			executed.ExitCode != 0 || executed.TimedOut {
			t.Errorf("granted command did not execute after registry reset: %s", results[0])
		}

		var refused struct {
			Error *codetools.Refusal `json:"error"`
		}
		if err := json.Unmarshal([]byte(results[1]), &refused); err != nil {
			t.Errorf("decode ungranted command refusal: %v; result=%q", err, results[1])
		} else if refused.Error == nil || refused.Error.Code != codetools.CodeCommandDeny {
			t.Errorf("ungranted command must retain COMMAND_DENY after registry reset: %s", results[1])
		}
	})

	t.Run("repeated_setup_is_idempotent", func(t *testing.T) {
		before := len(abi.Adjudicators())
		for attempt := 1; attempt <= 2; attempt++ {
			if _, err := agent.ArmCodeToolsWithOptions(opts); err != nil {
				t.Fatalf("repeated setup %d: %v", attempt, err)
			}
			if got := len(abi.Adjudicators()); got != before {
				t.Fatalf("setup %d grew adjudication chain from %d to %d", attempt, before, got)
			}
		}
	})

	t.Run("proxy_still_requires_permission", func(t *testing.T) {
		proxyCfg := cfg
		proxyCfg.EngineID, proxyCfg.Model = "localtools", "test-model"
		proxyCfg.VDSO, proxyCfg.Native = true, true
		srv, err := New(proxyCfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(srv.Close)

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		kept, adjs, dropped, served, hits := srv.adjudicateProposedServed(ctx,
			nativeBashCalls(t, granted), "registry-rearm-unowned-proposal")
		if len(kept) != 0 || len(adjs) != 1 || dropped != 1 || served != "" || hits != 0 {
			t.Fatalf("unowned proposal kept=%+v adjudications=%+v dropped=%d served=%q hits=%d",
				kept, adjs, dropped, served, hits)
		}
		if verdict := adjs[0].Verdict; verdict.Reason != "DEFAULT_DENY" || verdict.By != "lease-admission" {
			t.Fatalf("unowned proposal verdict=%+v, want lease-admission DEFAULT_DENY", verdict)
		}
	})
}
