package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ops"
)

// runOpsWithTimeout invokes the ops dispatcher on a separate goroutine and
// enforces a hard deadline. A regression that starts an infinite daemon ticker
// blocks forever; without this guard the test suite would hang CI instead of
// failing loudly, so a timeout is treated as a fatal assertion failure.
func runOpsWithTimeout(t *testing.T, args []string, root string, cfg ops.Config) (int, string, string) {
	t.Helper()

	type result struct {
		rc   int
		out  string
		errb string
	}

	ch := make(chan result, 1)
	go func() {
		var out, errb bytes.Buffer
		rc := runOps(&out, &errb, args)
		ch <- result{rc: rc, out: out.String(), errb: errb.String()}
	}()

	select {
	case r := <-ch:
		return r.rc, r.out, r.errb
	case <-time.After(10 * time.Second):
		t.Fatalf("runOps(%v) did not return within timeout — daemon ticker likely started", args)
		return 0, "", ""
	}
}

// TestOpsDaemonUnknownArgsRefuseBeforeEffects is the ticket witness for the
// fail-closed `fak ops daemon` argument contract: unknown/refused arg shapes
// must be rejected BEFORE any effect (no ticker, no daemon, prompt return),
// and read-only subcommands must never start a daemon.
func TestOpsDaemonUnknownArgsRefuseBeforeEffects(t *testing.T) {
	root := t.TempDir()
	cfg := ops.DefaultConfig()

	t.Run("unknown_arg_refuses_before_effects", func(t *testing.T) {
		rc, _, errb := runOpsWithTimeout(t, []string{"daemon", "bogus"}, root, cfg)
		if rc != 2 {
			t.Fatalf("runOps(daemon bogus) rc = %d, want 2", rc)
		}
		if strings.TrimSpace(errb) == "" {
			t.Fatalf("runOps(daemon bogus) wrote empty stderr, want a non-empty refusal message")
		}
	})

	t.Run("status_never_starts_daemon", func(t *testing.T) {
		rc, _, errb := runOpsWithTimeout(t, []string{"daemon", "status"}, root, cfg)
		if rc != 0 {
			t.Fatalf("runOps(daemon status) rc = %d, want 0 (stderr=%q)", rc, errb)
		}
	})

	t.Run("status_json_never_starts_daemon", func(t *testing.T) {
		rc, _, errb := runOpsWithTimeout(t, []string{"daemon", "status", "--json"}, root, cfg)
		if rc != 0 {
			t.Fatalf("runOps(daemon status --json) rc = %d, want 0 (stderr=%q)", rc, errb)
		}
	})

	t.Run("stop_never_adopts_unowned_pid", func(t *testing.T) {
		rc, out, errb := runOpsWithTimeout(t, []string{"daemon", "stop"}, root, cfg)
		if rc == 0 {
			t.Fatalf("runOps(daemon stop) rc = 0, want non-zero (fail-closed: no owned daemon); stdout=%q stderr=%q", out, errb)
		}
		if strings.TrimSpace(errb) == "" {
			t.Fatalf("runOps(daemon stop) wrote empty stderr, want a non-empty refusal message")
		}
	})

	t.Run("unknown_variant_with_json_refuses", func(t *testing.T) {
		rc, _, errb := runOpsWithTimeout(t, []string{"daemon", "bogus2", "--json"}, root, cfg)
		if rc != 2 {
			t.Fatalf("runOps(daemon bogus2 --json) rc = %d, want 2", rc)
		}
		if strings.TrimSpace(errb) == "" {
			t.Fatalf("runOps(daemon bogus2 --json) wrote empty stderr, want a non-empty refusal message")
		}
	})
}
