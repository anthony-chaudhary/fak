package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ops"
)

func TestOpsDaemonUnknownArgsRefuseBeforeEffects(t *testing.T) {
	cases := []struct {
		name string
		args []string
	}{
		{name: "subcommand_like_positional", args: []string{"status"}},
		{name: "subcommand_like_positional_with_token", args: []string{"status", "now"}},
		{name: "bare_unknown_positional_stop", args: []string{"stop"}},
		{name: "bare_unknown_positional_frobnicate", args: []string{"frobnicate"}},
		{name: "unknown_flag_like_token", args: []string{"--json"}},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			root := t.TempDir()
			cfg := ops.DefaultConfig()

			rc := runOpsDaemon(&stdout, &stderr, root, cfg, tc.args)

			if rc != 2 {
				t.Fatalf("rc = %d, want 2 (fail-closed refusal) for args %q; stderr=%q", rc, tc.args, stderr.String())
			}

			offending := tc.args[0]
			if !strings.Contains(stderr.String(), offending) {
				t.Errorf("stderr must name the offending argument %q; stderr=%q", offending, stderr.String())
			}

			if strings.Contains(stdout.String(), "daemon started") {
				t.Errorf("stdout must not contain the daemon-started banner before refusal; stdout=%q", stdout.String())
			}
			if strings.Contains(stdout.String(), "tick") {
				t.Errorf("stdout must not contain a maintenance-tick completion message before refusal; stdout=%q", stdout.String())
			}
		})
	}
}
