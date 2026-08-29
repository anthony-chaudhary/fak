package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/repoguard"
)

// TestGuardBuildCacheCleanClassifierParity pins the compiled repo hook and the
// embedded capability floor to the same use-vs-mention decision table. They use
// separate tokenizers intentionally; this witness prevents either surface from
// silently gaining or losing a command shape relative to the other.
func TestGuardBuildCacheCleanClassifierParity(t *testing.T) {
	runtime, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("parse embedded guard policy: %v", err)
	}
	adj := adjudicator.New(runtime.Adjudicator)
	resolver := abi.ActiveResolver()
	if resolver == nil {
		t.Fatal("no Ref resolver registered")
	}

	for _, tc := range []struct {
		command string
		deny    bool
	}{
		{"go clean -cache", true},
		{"go clean -cache -testcache", true},
		{"go clean -testcache -cache", true},
		{"env GOENV=off /usr/local/bin/go clean --cache=true", true},
		{"cd /tmp && go clean -cache", true},
		{"sh -c 'go clean -cache'", true},
		{"go clean -testcache", false},
		{"go clean -cache=false", false},
		{"go test ./...", false},
		{"go vet ./...", false},
		{"echo 'go clean -cache'", false},
		{"rg -n 'go clean -cache' docs", false},
		{"git commit -m 'docs: avoid go clean -cache'", false},
	} {
		t.Run(tc.command, func(t *testing.T) {
			hookDeny := len(repoguard.ClassifyBuildCacheClean(tc.command)) > 0
			args, err := json.Marshal(map[string]string{"command": tc.command})
			if err != nil {
				t.Fatal(err)
			}
			ref, err := resolver.Put(context.Background(), args)
			if err != nil {
				t.Fatal(err)
			}
			verdict := adj.Adjudicate(context.Background(), &abi.ToolCall{Tool: "Bash", Args: ref})
			policyDeny := verdict.Kind == abi.VerdictDeny
			if hookDeny != tc.deny || policyDeny != tc.deny {
				t.Fatalf("hook deny=%t policy deny=%t, want %t (policy reason=%s)",
					hookDeny, policyDeny, tc.deny, abi.ReasonName(verdict.Reason))
			}
		})
	}
}

func TestGuardBuildCacheCleanEmbeddedPolicyAliases(t *testing.T) {
	runtime, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("parse embedded guard policy: %v", err)
	}
	adj := adjudicator.New(runtime.Adjudicator)
	resolver := abi.ActiveResolver()
	if resolver == nil {
		t.Fatal("no Ref resolver registered")
	}
	for _, tool := range []struct{ name, arg string }{
		{"Bash", "command"},
		{"PowerShell", "command"},
		{"shell_command", "command"},
		{"functions.shell_command", "command"},
		{"exec_command", "cmd"},
	} {
		t.Run(tool.name, func(t *testing.T) {
			decide := func(command string) abi.Verdict {
				t.Helper()
				args, err := json.Marshal(map[string]string{tool.arg: command})
				if err != nil {
					t.Fatal(err)
				}
				ref, err := resolver.Put(context.Background(), args)
				if err != nil {
					t.Fatal(err)
				}
				return adj.Adjudicate(context.Background(), &abi.ToolCall{Tool: tool.name, Args: ref})
			}
			for _, command := range []string{"go clean -cache", "go clean -testcache -cache", "env GOENV=off go clean --cache=true"} {
				v := decide(command)
				if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
					t.Errorf("%q = %v/%s, want Deny/POLICY_BLOCK", command, v.Kind, abi.ReasonName(v.Reason))
				}
			}
			for _, command := range []string{"go clean -testcache", "go test ./...", "echo 'go clean -cache'", "rg -n 'go clean -cache' docs"} {
				v := decide(command)
				if v.Kind != abi.VerdictAllow {
					t.Errorf("inert/control %q = %v/%s, want Allow", command, v.Kind, abi.ReasonName(v.Reason))
				}
			}
		})
	}
}
