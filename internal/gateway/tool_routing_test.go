package gateway

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func TestCanonicalToolNormalization_Table(t *testing.T) {
	pol := &policy.Runtime{
		Adjudicator: adjudicator.Policy{
			Allow: map[string]bool{
				"exec_command": true,
				"fak_read":     true,
				"dos_verify":   true,
				"read_file":    true,
				"apply_patch":  true,
			},
			AllowPrefix: []string{"search_"},
		},
	}

	tests := []struct {
		input string
		pol   *policy.Runtime
		want  string
	}{
		// 1. OpenAI / Codex prefix "functions."
		{"functions.exec_command", pol, "exec_command"},
		{"functions.apply_patch", pol, "apply_patch"},
		{"functions.search_code", pol, "search_code"},

		// 2. Claude Code MCP prefix "mcp__<server>__<tool>"
		{"mcp__fak__fak_read", pol, "fak_read"},
		{"mcp__dos__dos_verify", pol, "dos_verify"},
		{"mcp__filesystem__read_file", pol, "read_file"},
		{"mcp__fak__search_index", pol, "search_index"},

		// 3. OpenCode double prefix "<server>_<server>_<tool>"
		{"fak_fak_read", pol, "fak_read"},
		{"dos_dos_verify", pol, "dos_verify"},

		// 4. OpenCode single prefix "<server>_<tool>"
		{"bash_exec_command", pol, "exec_command"},

		// Unprefixed canonical tools remain unchanged
		{"exec_command", pol, "exec_command"},
		{"fak_read", pol, "fak_read"},
		{"dos_verify", pol, "dos_verify"},

		// Unknown tools / prefixes not in policy remain unchanged
		{"unknown.other_tool", pol, "unknown.other_tool"},
		{"unregistered_tool", pol, "unregistered_tool"},

		// Nil policy fallback
		{"functions.custom_tool", nil, "custom_tool"},
		{"mcp__srv__some_tool", nil, "some_tool"},
		{"fak_fak_read", nil, "fak_read"},
	}

	for _, tc := range tests {
		got := canonicalToolName(tc.input, tc.pol)
		if got != tc.want {
			t.Errorf("canonicalToolName(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestCanonicalToolNormalization(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterEngine("mock", engine.MockEngine)

	pol := &policy.Runtime{
		Adjudicator: adjudicator.Policy{
			Posture: adjudicator.PostureFailClosed,
			Allow: map[string]bool{
				"exec_command": true,
				"fak_read":     true,
			},
		},
	}

	adj := adjudicator.New(pol.Adjudicator)
	abi.RegisterAdjudicator(0, adj)

	srv, err := New(Config{
		EngineID:      "test",
		Model:         "test-model",
		VDSO:          true,
		PolicyRuntime: pol,
	})
	if err != nil {
		t.Fatalf("New server: %v", err)
	}
	t.Cleanup(srv.Close)

	ctx := context.Background()

	// 1. functions.exec_command resolves against canonical "exec_command" allow rule
	wv, _, err := srv.adjudicate(ctx, "functions.exec_command", `{"command":"echo 1"}`, false, "", "trace-1")
	if err != nil {
		t.Fatalf("adjudicate(functions.exec_command): %v", err)
	}
	if wv.Kind != "ALLOW" {
		t.Errorf("functions.exec_command verdict = %q, want ALLOW", wv.Kind)
	}

	// 2. mcp__fak__fak_read resolves against canonical "fak_read" allow rule
	wv, _, err = srv.adjudicate(ctx, "mcp__fak__fak_read", `{"path":"foo.go"}`, true, "", "trace-2")
	if err != nil {
		t.Fatalf("adjudicate(mcp__fak__fak_read): %v", err)
	}
	if wv.Kind != "ALLOW" {
		t.Errorf("mcp__fak__fak_read verdict = %q, want ALLOW", wv.Kind)
	}

	// 3. OpenCode double-prefixed fak_fak_read resolves against canonical "fak_read"
	wv, _, err = srv.adjudicate(ctx, "fak_fak_read", `{"path":"bar.go"}`, true, "", "trace-3")
	if err != nil {
		t.Fatalf("adjudicate(fak_fak_read): %v", err)
	}
	if wv.Kind != "ALLOW" {
		t.Errorf("fak_fak_read verdict = %q, want ALLOW", wv.Kind)
	}

	// 4. OpenCode single-prefixed bash_exec_command resolves against canonical "exec_command"
	wv, _, err = srv.adjudicate(ctx, "bash_exec_command", `{"command":"pwd"}`, false, "", "trace-4")
	if err != nil {
		t.Fatalf("adjudicate(bash_exec_command): %v", err)
	}
	if wv.Kind != "ALLOW" {
		t.Errorf("bash_exec_command verdict = %q, want ALLOW", wv.Kind)
	}

	// 5. Unregistered tool is denied as expected
	wv, _, err = srv.adjudicate(ctx, "functions.forbidden_action", `{}`, false, "", "trace-5")
	if err != nil {
		t.Fatalf("adjudicate(forbidden_action): %v", err)
	}
	if wv.Kind != "DENY" {
		t.Errorf("functions.forbidden_action verdict = %q, want DENY", wv.Kind)
	}
}
