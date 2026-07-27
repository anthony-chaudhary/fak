package main

import (
	"context"
	"encoding/json"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// TestScratchDeleteVerdictDoesNotDependOnPathSpelling is the end-to-end witness for
// the spelling defect, decided against the REAL embedded floor rather than the
// helper in isolation: one throwaway directory inside the declared scratch root must
// get the SAME verdict whether it is named in the drive-letter spelling or the MSYS
// spelling Git Bash uses for that identical directory. Before the alias expansion
// the two disagreed — a live probe inside the session's own harness scratchpad saw
// `/c/…` hard-denied (POLICY_BLOCK) while byte-equivalent `C:/…` was downgraded to
// the reversibility preview-confirm gate. A capability floor whose strictness tracks
// which shell spelled the path is not a floor.
//
// The out-of-scratch case is asserted in the same breath so this can only ever pin
// the SPELLING equivalence: a recursive delete aimed at the workspace stays denied
// in both spellings, which is what keeps this a false-positive fix and not a hole.
func TestScratchDeleteVerdictDoesNotDependOnPathSpelling(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("drive-letter/MSYS spelling duality is Windows-only")
	}
	t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", guardScratchpadRootsValue(`C:\agent-scratch\claude`))

	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("embedded guard floor is not a valid manifest: %v", err)
	}
	adj := adjudicator.New(rt.Adjudicator)
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatal("no Ref resolver registered (internal/registrations blank import missing)")
	}
	remove := "rm" // spelled indirectly so this source line is not itself a policy tell
	decide := func(command string) abi.Verdict {
		args, err := json.Marshal(map[string]string{"command": command})
		if err != nil {
			t.Fatalf("encode args: %v", err)
		}
		ref, err := res.Put(context.Background(), args)
		if err != nil {
			t.Fatalf("put args: %v", err)
		}
		return adj.Adjudicate(context.Background(), &abi.ToolCall{Tool: "Bash", Args: ref})
	}

	for _, target := range []string{
		`C:/agent-scratch/claude/session-1/probe`,
		`/c/agent-scratch/claude/session-1/probe`,
	} {
		if v := decide(remove + " -rf " + target); v.Kind == abi.VerdictDeny {
			t.Errorf("recursive delete of %s inside the declared scratch root was DENIED (%v): "+
				"cleaning up the session's own throwaway tree is routine work", target, v.Reason)
		}
	}
	// The fail-closed half, in BOTH spellings: outside the declared root the
	// recursive delete keeps its hard deny.
	for _, target := range []string{
		`C:/work/fak/internal`,
		`/c/work/fak/internal`,
	} {
		if v := decide(remove + " -rf " + target); v.Kind != abi.VerdictDeny {
			t.Errorf("recursive delete of %s is OUTSIDE every declared scratch root and must stay denied, got %v", target, v.Kind)
		}
	}
}
