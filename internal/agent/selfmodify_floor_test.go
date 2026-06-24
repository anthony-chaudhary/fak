package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// guardedCall builds a tool call with inline JSON args (no resolver needed for the
// args read path) — the agent-package analogue of adjudicator's inlineCall, used to
// drive the live adjudicator.Default that Configure() installs.
func guardedCall(tool, jsonArgs string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(jsonArgs)},
		Meta: map[string]string{"readOnlyHint": "false"},
	}
}

// jsonEscapeCmd minimally escapes a shell command for inline JSON test args (only the
// characters these commands contain: backslash and double-quote).
func jsonEscapeCmd(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	return strings.ReplaceAll(s, `"`, `\"`)
}

// TestConfiguredFloorGuardsBashWriteAndWitnessTrees is the agent-arm half of the #386
// self-modify floor. Configure() installs the policy the in-kernel bench/dogfood arm
// runs under (RunArm with fak=true), via adjudicator.Default.SetPolicy. The old inline
// SelfModifyGlobs literal in Configure() ({internal/abi/, internal/kernel/, .dos/})
// silently lagged the #172 Hole 2 extension: it did NOT name the WITNESS machinery
// (internal/architest, internal/shipgate, internal/adjudicator, dos.toml), so on THIS
// arm a self-improving loop could edit its own grader through either the direct-write
// OR the Bash-write path even though the shell-write guard (commandSelfModify) was
// already wired in Adjudicate. Configure() now sources the set from
// adjudicator.DefaultPolicy(); this gate proves it.
//
// It witnesses BOTH holes on the live, configured policy:
//   - Hole 1 (Bash write path): a Bash command writing into a guarded tree (e.g.
//     `sed -i internal/abi/...`) is denied SELF_MODIFY, while a Bash READ of the same
//     tree, and a write outside it, stay allowed.
//   - Hole 2 (witness-tree coverage): the configured SelfModifyGlobs is a SUPERSET of
//     the witness trees — the test FAILS if any witness tree is dropped.
//
// WHY HERE (not only in adjudicator): the in-package adjudicator tests run against
// DefaultPolicy()/DevAgentPolicy(); they do not witness what Configure() actually
// installs. A future edit could revert Configure()'s glob set to a stale subset and
// every adjudicator test would stay green while the configured agent arm quietly lost
// witness-tree coverage. This gate binds the CONFIGURED policy to the floor.
func TestConfiguredFloorGuardsBashWriteAndWitnessTrees(t *testing.T) {
	Configure() // installs the agent-arm policy into adjudicator.Default

	// --- Hole 2: the configured guarded set must cover every witness tree. ---
	// Each witness tree paired with a sample write target inside it; coverage is
	// checked by substring, the same way matchGlob matches at runtime.
	witness := map[string]string{
		"internal/abi/":         "internal/abi/kernel.go",           // the kernel spine
		"internal/kernel/":      "internal/kernel/walk.go",          // the kernel walker
		"internal/adjudicator/": "internal/adjudicator/decide.go",   // the guard itself
		"internal/architest/":   "internal/architest/floor_test.go", // the gates
		"internal/shipgate/":    "internal/shipgate/gate.go",        // the ship harness
		"dos.toml":              "dos.toml",                         // the lane taxonomy
	}
	globs := adjudicator.DefaultPolicy().SelfModifyGlobs // the source Configure() now uses
	covers := func(target string) bool {
		for _, g := range globs {
			if g != "" && strings.Contains(target, g) {
				return true
			}
		}
		return false
	}
	for tree, target := range witness {
		if !covers(target) {
			t.Errorf("configured SelfModifyGlobs does not cover witness tree %q "+
				"(no glob fragment is contained in %q).\nSelfModifyGlobs=%v\n"+
				"This is the #386 floor: %s is part of the self-modify/witness machinery — if the "+
				"in-kernel agent arm can write there, a self-improving loop can grade its own homework. "+
				"Configure() must source SelfModifyGlobs from adjudicator.DefaultPolicy().",
				tree, target, globs, tree)
		}
		// End-to-end on the DIRECT-write path through the live, configured policy.
		v := adjudicator.Default.Adjudicate(context.Background(),
			guardedCall("write_file", `{"path":"`+target+`"}`))
		if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
			t.Errorf("configured policy: direct write to witness tree %q got %v/%s, want Deny/SELF_MODIFY",
				target, v.Kind, abi.ReasonName(v.Reason))
		}
	}

	// --- Hole 1: a Bash command WRITING into a guarded tree is denied SELF_MODIFY,
	// while a READ of, and a write OUTSIDE, the guarded trees stay allowed. ---
	cases := []struct {
		name    string
		command string
		want    bool // want Deny/SELF_MODIFY
	}{
		// Acceptance: "a Bash command writing into internal/abi/ is denied unless elevated."
		{"sed -i into abi", "sed -i 's/x/y/' internal/abi/kernel.go", true},
		{"tee into kernel", "echo hi | tee internal/kernel/walk.go", true},
		{"redirect into dos.toml", "echo '[x]' > dos.toml", true},
		{"git apply into adjudicator", "git apply < /tmp/p.patch internal/adjudicator/decide.go", true},
		{"git checkout guarded shipgate", "git checkout -- internal/shipgate/gate.go", true},
		{"cp over architest", "cp /tmp/evil.go internal/architest/floor_test.go", true},
		// A Bash READ of a guarded file is NOT a self-modify — stays allowed.
		{"cat a guarded file", "cat internal/abi/kernel.go", false},
		{"grep a guarded tree", "grep -r foo internal/kernel/", false},
		// A write OUTSIDE the guarded trees is allowed by this guard.
		{"sed -i a normal file", "sed -i 's/a/b/' README.md", false},
	}
	for _, c := range cases {
		args := `{"command":"` + jsonEscapeCmd(c.command) + `"}`
		v := adjudicator.Default.Adjudicate(context.Background(), guardedCall("Bash", args))
		got := v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonSelfModify
		if got != c.want {
			t.Errorf("%s: Bash command %q got %v/%s, wantSelfModify=%v",
				c.name, c.command, v.Kind, abi.ReasonName(v.Reason), c.want)
		}
	}
}
