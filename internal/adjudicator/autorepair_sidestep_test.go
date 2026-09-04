package adjudicator

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The reversibility rung auto-substitutes a bare `git push` with the sanctioned
// `fak sync push` ONLY when the operator opts in (Policy.AutoRepairSidestep, wired
// from FAK_GUARD_AUTOREPAIR=sidestep) AND the call is in the safe subset. These
// tests pin all three axes: the opt-in gate, the safe-subset boundary, and the
// faithful correspondence between the substituted verb and the family remedy table.

func transformCommand(t *testing.T, ctx context.Context, v abi.Verdict) string {
	t.Helper()
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("payload type = %T, want TransformPayload", v.Payload)
	}
	var got map[string]any
	if err := json.Unmarshal(refBytes(ctx, tp.NewArgs), &got); err != nil {
		t.Fatalf("transformed args are not JSON: %v", err)
	}
	cmd, _ := got["command"].(string)
	return cmd
}

func TestAutoRepairSidestepSubstitutesBarePush(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}, AutoRepairSidestep: true})
	ctx := context.Background()

	// The safe subset is a CURRENT-BRANCH push: bare `git push` or `git push <remote>`
	// with no branch. Naming a branch (`git push origin main`) is a specific target and
	// stays on the hold — see TestAutoRepairSidestepRefusesDangerousPush.
	for _, cmd := range []string{"git push", "git push origin"} {
		b, _ := json.Marshal(map[string]any{"command": cmd})
		v := a.Adjudicate(ctx, inlineCall("Bash", string(b)))
		if v.Kind != abi.VerdictTransform {
			t.Fatalf("opt-in safe push %q: got %v/%s, want Transform (auto-substitute)", cmd, v.Kind, abi.ReasonName(v.Reason))
		}
		if got := transformCommand(t, ctx, v); got != "fak sync push" {
			t.Fatalf("substituted command = %q, want %q", got, "fak sync push")
		}
	}

	v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git push"}`))
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("got %v, want Transform", v.Kind)
	}
	// Same-tool rewrite (Bash → Bash): NewTool must stay empty so the client keeps
	// running Bash rather than being told to hop tools.
	if tp := v.Payload.(abi.TransformPayload); tp.NewTool != "" {
		t.Fatalf("NewTool = %q, want empty for a Bash→Bash rewrite", tp.NewTool)
	}
	if v.Meta["reversibility_autorepair"] != "sidestep" {
		t.Fatalf("transform must record the auto-repair, meta=%v", v.Meta)
	}
}

func TestAutoRepairSidestepSubstitutesBarePull(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}, AutoRepairSidestep: true})
	ctx := context.Background()

	for _, cmd := range []string{"git pull", "git pull origin", "git pull origin main"} {
		b, _ := json.Marshal(map[string]any{"command": cmd})
		v := a.Adjudicate(ctx, inlineCall("Bash", string(b)))
		if v.Kind != abi.VerdictTransform {
			t.Fatalf("opt-in safe pull %q: got %v/%s, want Transform (auto-substitute)", cmd, v.Kind, abi.ReasonName(v.Reason))
		}
		if got := transformCommand(t, ctx, v); got != "fak sync apply --fetch" {
			t.Fatalf("substituted command = %q, want %q", got, "fak sync apply --fetch")
		}
		if tp := v.Payload.(abi.TransformPayload); tp.NewTool != "" {
			t.Fatalf("NewTool = %q, want empty for a Bash→Bash rewrite", tp.NewTool)
		}
		if v.Meta["reversibility_autorepair"] != "sidestep" {
			t.Fatalf("transform must record the auto-repair, meta=%v", v.Meta)
		}
	}
}

func TestAutoRepairSidestepPreservesDescription(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}, AutoRepairSidestep: true})
	ctx := context.Background()

	// Only the command is swapped; every other arg (description, and crucially workdir,
	// which selects WHICH repo the push targets) must carry through unchanged.
	v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git push","description":"ship the auth fix","workdir":"C:\\work\\fak"}`))
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("got %v, want Transform", v.Kind)
	}
	tp := v.Payload.(abi.TransformPayload)
	var got map[string]any
	if err := json.Unmarshal(refBytes(ctx, tp.NewArgs), &got); err != nil {
		t.Fatalf("args not JSON: %v", err)
	}
	if got["command"] != "fak sync push" {
		t.Fatalf("command = %v, want fak sync push", got["command"])
	}
	if got["description"] != "ship the auth fix" {
		t.Fatalf("description dropped during substitution: %v", got["description"])
	}
	if got["workdir"] != `C:\work\fak` {
		t.Fatalf("workdir dropped during substitution — would push the wrong repo: %v", got["workdir"])
	}
}

func TestAutoRepairSidestepHoldsWhenDisabled(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}}) // AutoRepairSidestep defaults off
	ctx := context.Background()

	v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"git push"}`))
	if v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("opt-out bare push: got %v/%s, want RequireWitness (hold preserved, default off)", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestAutoRepairSidestepRefusesDangerousPush(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"Bash": true}, AutoRepairSidestep: true})
	ctx := context.Background()

	// Every one of these differs from `fak sync push` (non-force, current-branch,
	// fast-forward-only), so laundering it into that verb would silently change
	// intent — worse than refusing. Each must stay on the preview-confirm hold even
	// with auto-repair ON.
	for _, cmd := range []string{
		"git push --force origin main",
		"git push -f",
		"git push --force-with-lease origin main",
		"git push --delete origin feature",
		"git push -d origin feature",
		"git push --tags",
		"git push --all",
		"git push --mirror",
		"git push --prune origin",
		"git push origin local:remote", // explicit refspec
		"git push -u origin main",      // sets upstream
		"git push origin main",         // names a specific branch, not the current one
		"git add -A && git push",       // a sibling command rides along
	} {
		b, _ := json.Marshal(map[string]any{"command": cmd})
		v := a.Adjudicate(ctx, inlineCall("Bash", string(b)))
		if v.Kind != abi.VerdictRequireWitness {
			t.Errorf("dangerous push %q: got %v/%s, want RequireWitness (must NOT auto-substitute)", cmd, v.Kind, abi.ReasonName(v.Reason))
		}
	}
}

// TestAutoRepairSidestepMatchesRemedyTable pins the substituted verb to the same
// string the human-facing remedy table advertises, so the auto-applied command and
// the advisory hint can never drift apart.
func TestAutoRepairSidestepMatchesRemedyTable(t *testing.T) {
	env := ClassifyReversibility("Bash", map[string]any{"command": "git push origin"})
	want := familyRemedyCommands["git-push"][0]
	if env.RewriteCommand != want {
		t.Fatalf("classifier rewrite = %q, remedy table = %q — the auto-applied verb drifted from the advertised sidestep", env.RewriteCommand, want)
	}
	if env.RewriteTool != "Bash" {
		t.Fatalf("rewrite tool = %q, want Bash", env.RewriteTool)
	}
	// A dangerous push carries NO rewrite even at the classifier layer, so the rung's
	// `env.RewriteCommand != ""` gate is sufficient — it can never see a force push.
	danger := ClassifyReversibility("Bash", map[string]any{"command": "git push --force origin main"})
	if danger.RewriteCommand != "" {
		t.Fatalf("force push carried a rewrite target %q — the safe-subset gate leaked", danger.RewriteCommand)
	}
}
