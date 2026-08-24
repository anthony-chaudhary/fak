package hooks

import (
	"strings"
	"testing"
)

func TestGateNativeFirstRejectsSilentSubstitution(t *testing.T) {
	for _, line := range []string{
		"Qwen3.8 native performance defaults to llama.cpp.",
		"Native falls back to llama-server.",
		"The native backend auto-selects llamacpp.",
	} {
		got, err := checkNativeFirst(&StagedDiff{AddedByFile: map[string][]AddedLine{"docs/x.md": {{File: "docs/x.md", New: 7, Text: line}}}})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0].File != "docs/x.md" || got[0].Line != 7 || !strings.Contains(got[0].Detail, "fak-native") || !strings.Contains(got[0].Detail, line) {
			t.Fatalf("line=%q findings=%+v", line, got)
		}
	}
}

func TestGateNativeFirstAllowsExplicitReferenceUses(t *testing.T) {
	for _, line := range []string{
		"Benchmark fak-native against llama.cpp.",
		"Use llama.cpp explicitly for parity diagnosis.",
		"Study and borrow a llama.cpp kernel.",
		"The interop adapter delegates explicitly to llama-server.",
	} {
		got, err := checkNativeFirst(&StagedDiff{AddedByFile: map[string][]AddedLine{"docs/x.md": {{File: "docs/x.md", New: 1, Text: line}}}})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("line=%q findings=%+v", line, got)
		}
	}
}

func TestGateNativeFirstIsStagedAndOperatorSurfaceScoped(t *testing.T) {
	got, err := checkNativeFirst(&StagedDiff{AddedByFile: map[string][]AddedLine{
		"docs/added.md":    {{New: 1, Text: "Benchmark fak-native against llama.cpp."}},
		"binary/image.png": {{New: 1, Text: "Native defaults to llama.cpp."}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("findings=%+v", got)
	}
}

func TestPreCommitGatesIncludesNativeFirst(t *testing.T) {
	for _, g := range PreCommitGates() {
		if g.Name == "NATIVE_FIRST" {
			if g.ModeEnv != "NATIVEFIRST_HOOK_MODE" || g.EscapeEnv != "ALLOW_NATIVE_SUBSTITUTION" {
				t.Fatalf("gate configuration=%+v", g)
			}
			return
		}
	}
	t.Fatal("NATIVE_FIRST gate missing")
}
