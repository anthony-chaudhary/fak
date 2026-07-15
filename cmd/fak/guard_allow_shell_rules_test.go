package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func TestGuardApplyAllowOverlayAttachesShellDangerRules(t *testing.T) {
	rt := policy.Runtime{Adjudicator: adjudicator.Policy{Allow: map[string]bool{}}}
	ov := guardAllowOverlay{Allow: []string{"opencode.bash", "mcp__x__bash", "mcp__x__search"}}
	if added := guardApplyAllowOverlay(&rt, ov); added != 3 {
		t.Fatalf("added = %d, want 3", added)
	}
	if len(rt.Adjudicator.ArgPredicates) == 0 {
		t.Fatal("shell overlay names got no danger predicates")
	}
	before := len(rt.Adjudicator.ArgPredicates)
	if added := guardApplyAllowOverlay(&rt, ov); added != 0 {
		t.Errorf("idempotent re-apply added = %d, want 0", added)
	}
	if got := len(rt.Adjudicator.ArgPredicates); got != before {
		t.Fatalf("re-apply duplicated predicates: before=%d after=%d", before, got)
	}

	res := abi.ActiveResolver()
	if res == nil {
		t.Fatal("no active ref resolver")
	}
	decide := func(tool, command string) abi.VerdictKind {
		t.Helper()
		ref, err := res.Put(context.Background(), []byte(`{"command":"`+command+`"}`))
		if err != nil {
			t.Fatalf("put args: %v", err)
		}
		return adjudicator.New(rt.Adjudicator).Adjudicate(context.Background(), &abi.ToolCall{Tool: tool, Args: ref}).Kind
	}
	dangerous := "rm" + " -rf /"
	for _, tool := range []string{"opencode.bash", "mcp__x__bash"} {
		if got := decide(tool, dangerous); got != abi.VerdictDeny {
			t.Errorf("%s danger verdict = %v, want DENY", tool, got)
		}
		if got := decide(tool, "echo ok"); got != abi.VerdictAllow {
			t.Errorf("%s benign verdict = %v, want ALLOW", tool, got)
		}
	}
	for _, pred := range rt.Adjudicator.ArgPredicates {
		if pred.Tool == "mcp__x__search" {
			t.Errorf("non-shell overlay inherited danger predicate: %+v", pred)
		}
	}
}

func TestGuardAllowShellAttachmentOutput(t *testing.T) {
	var out bytes.Buffer
	printGuardAllowShellAttachments(&out, []string{"opencode.bash", "mcp__x__bash", "mcp__x__search"})
	got := out.String()
	for _, want := range []string{"opencode.bash=posix_shell", "mcp__x__bash=posix_shell"} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q: %s", want, got)
		}
	}
	if strings.Contains(got, "search") {
		t.Errorf("non-shell tool rendered as attached: %s", got)
	}
}

func TestLoadGuardCapabilityFloorNamesShellDangerAttachments(t *testing.T) {
	dir := t.TempDir()
	overlayPath := filepath.Join(dir, "allow.json")
	t.Setenv(guardAllowOverlayEnv, overlayPath)
	if err := saveGuardAllowOverlay(overlayPath, guardAllowOverlay{Allow: []string{"opencode.bash", "mcp__x__search"}}); err != nil {
		t.Fatal(err)
	}
	_, source, _, _ := loadGuardCapabilityFloor("")
	if !strings.Contains(source, "inherited shell danger rules attached: opencode.bash=posix_shell") {
		t.Fatalf("floor source does not name inherited shell attachment: %s", source)
	}
	if strings.Contains(source, "search=") {
		t.Fatalf("floor source names non-shell attachment: %s", source)
	}
}
