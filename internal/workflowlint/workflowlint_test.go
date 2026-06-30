package workflowlint

import (
	"strings"
	"testing"
)

// The canonical seed must lint FAK-NATIVE by construction — that is the whole point
// of shipping it as the template (epic #1494 / C4 #1502). If this fails, the seed has
// drifted away from the concepts the lint enforces.
func TestSeedTemplateIsNative(t *testing.T) {
	if strings.TrimSpace(SeedTemplate) == "" {
		t.Fatal("SeedTemplate is empty — embed of seed.js failed")
	}
	rep := Lint(SeedTemplate)
	if !rep.Native || rep.Verdict != VerdictNative {
		t.Fatalf("seed must be FAK-NATIVE, got verdict=%q missing=%v", rep.Verdict, rep.Missing)
	}
	if len(rep.Missing) != 0 {
		t.Fatalf("seed missing concept classes: %v", rep.Missing)
	}
}

// A generic ultracode workflow with none of the fak concepts is the FAK-BLIND case
// the lint exists to refuse. All three classes must be reported missing.
func TestFakBlindWorkflowRefused(t *testing.T) {
	blind := `export const meta = { name: 'review-fix', description: 'review and fix' }
phase('Review')
const r = await agent('review the changed files for bugs and fix them', {schema: S})
return { r }`
	rep := Lint(blind)
	if rep.Native || rep.Verdict != VerdictBlind {
		t.Fatalf("fak-blind workflow must be refused, got verdict=%q", rep.Verdict)
	}
	wantMissing := map[string]bool{ClassSelfIndex: true, ClassMemory: true, ClassSharedPath: true}
	if len(rep.Missing) != len(wantMissing) {
		t.Fatalf("want all 3 classes missing, got %v", rep.Missing)
	}
	for _, m := range rep.Missing {
		if !wantMissing[m] {
			t.Fatalf("unexpected missing class %q", m)
		}
	}
}

// Each concept class is detected independently, including the underscore (MCP tool)
// spellings fak_index_* / fak_memory_* and the --driver / dos_arbitrate forms.
func TestPerClassDetection(t *testing.T) {
	cases := []struct {
		name    string
		script  string
		present string // the one class that should be present
	}{
		{"cli self-index", "run `fak index leaf cache`", ClassSelfIndex},
		{"mcp self-index", "call fak_index_leaf via MCP", ClassSelfIndex},
		{"cli memory", "run `fak memory drivers`", ClassMemory},
		{"mcp memory", "call fak_memory_run", ClassMemory},
		{"driver memory", "use --driver recall before work", ClassMemory},
		{"memq memory", "the memq algebra pages it in", ClassMemory},
		{"arbitrate shared-path", "take a dos_arbitrate lease... wait that double counts", ClassSharedPath},
		{"collision shared-path", "if it refuses COLLISION_RISK stop", ClassSharedPath},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep := Lint(tc.script)
			var hit *ClassHit
			for i := range rep.Classes {
				if rep.Classes[i].Key == tc.present {
					hit = &rep.Classes[i]
				}
			}
			if hit == nil || !hit.Present {
				t.Fatalf("class %q should be present in %q; report=%+v", tc.present, tc.script, rep)
			}
			if len(hit.Matched) == 0 {
				t.Fatalf("present class %q must report a witness token", tc.present)
			}
		})
	}
}

// The "lease" keyword is a single shared-path witness; matching must be
// case-insensitive (generated scripts vary in casing).
func TestCaseInsensitive(t *testing.T) {
	rep := Lint("FAK INDEX ... FAK_MEMORY_RUN ... DOS_ARBITRATE LEASE")
	if !rep.Native {
		t.Fatalf("uppercase fak concepts must still lint native, missing=%v", rep.Missing)
	}
}

// Empty input is the conservative refusal, never a panic or a false native.
func TestEmptyIsBlind(t *testing.T) {
	rep := Lint("")
	if rep.Native || len(rep.Missing) != 3 {
		t.Fatalf("empty script must be FAK-BLIND with 3 missing, got %+v", rep)
	}
}
