package adjudicator

import (
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// synthPolicy allows the Bash exec tool and a file-write tool, so the ONLY thing
// that turns a synth-tool exec into a deny is the #543 envelope (synthToolSelfModify
// fires SELF_MODIFY before the affirmative allow). Reuses DefaultPolicy's guarded
// globs as the witness trees.
func synthPolicy() Policy {
	return Policy{
		Allow:           map[string]bool{"Bash": true, "write_file": true},
		SelfModifyGlobs: DefaultPolicy().SelfModifyGlobs,
	}
}

// TestSynthToolEnvelopeBindsGuardedReach is the #543 acceptance: a script the agent
// AUTHORED earlier this run, then EXEC'd against a guarded tree, is denied
// SELF_MODIFY exactly as the equivalent direct write would be — a self-synthesized
// tool inherits no more capability than a direct adjudicated call.
func TestSynthToolEnvelopeBindsGuardedReach(t *testing.T) {
	ctx := context.Background()
	a := New(synthPolicy())

	// 1. Author a helper into the (unguarded) workdir — allowed and recorded.
	if v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"helper.py","content":"x"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("authoring write: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}

	// 2. Exec it reaching a guarded tree -> SELF_MODIFY (the hole this floor closes;
	// today `python helper.py …` carries no write verb so commandSelfModify misses it).
	v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"python helper.py internal/abi/x.go"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Errorf("synth-exec into guarded tree: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}
	if wit, ok := v.Payload.(abi.WitnessPayload); !ok || wit.Claim != "internal/abi/" {
		t.Errorf("witness: got %+v, want bounded glob internal/abi/", v.Payload)
	}

	// 3. The SAME synth-tool reaching NO guarded tree stays allowed — the tool keeps
	// the agent's existing capability, it just cannot launder a guarded write.
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"python helper.py out.txt"}`)); v.Kind != abi.VerdictAllow {
		t.Errorf("synth-exec to unguarded path: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestSynthToolLedgerScoping proves the ledger is load-bearing: a script the agent
// did NOT author this run (a pre-existing script, even one that lives in a guarded
// tree) is a plain script run, not a synth-tool — it must NOT be denied, or every
// `pytest` / `python manage.py` would false-trip.
func TestSynthToolLedgerScoping(t *testing.T) {
	ctx := context.Background()
	a := New(synthPolicy())

	// No authoring write. Running a script that lives in a guarded tree names the
	// guarded glob, but the script is not in the ledger -> stays allowed (mirrors the
	// existing TestSelfModifyGuardsShellWritePath "python <guarded>/gen.py" negative).
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"python internal/abi/gen.py"}`)); v.Kind != abi.VerdictAllow {
		t.Errorf("non-authored script run: got %v/%s, want Allow (ledger must scope the guard)", v.Kind, abi.ReasonName(v.Reason))
	}

	// READING an authored script alongside a guarded path is not an EXEC -> allowed.
	if v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"helper.py"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("authoring write: got %v, want Allow", v.Kind)
	}
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"cat helper.py internal/abi/x.go"}`)); v.Kind != abi.VerdictAllow {
		t.Errorf("reading an authored script (not exec): got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestSynthToolAuthoringSurfaces covers the two ways an agent authors a script that
// the ledger must record: a shell redirect creating a script, and `chmod +x` marking
// an extension-less file executable (then run as `./t`).
func TestSynthToolAuthoringSurfaces(t *testing.T) {
	ctx := context.Background()

	// Shell-write surface: `echo … > gen.sh` authors gen.sh; `sh gen.sh <guarded>`
	// is then a synth-tool reaching a guarded tree.
	a := New(synthPolicy())
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"echo '#!/bin/sh' > gen.sh"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("shell authoring write: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"sh gen.sh internal/shipgate/x.go"}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Errorf("sh of shell-authored script into guarded tree: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}

	// chmod +x surface: `chmod +x ./t` authors the extension-less `t`; `./t <guarded>`
	// is then a direct synth-tool exec reaching a guarded tree.
	b := New(synthPolicy())
	if v := b.Adjudicate(ctx, inlineCall("Bash", `{"command":"chmod +x ./t"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("chmod +x: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := b.Adjudicate(ctx, inlineCall("Bash", `{"command":"./t internal/kernel/x.go"}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Errorf("./t of chmod-authored script into guarded tree: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestSynthToolResetRunClearsLedger proves the per-run scoping: after ResetRun the
// previously-authored script is no longer recognized as a synth-tool.
func TestSynthToolResetRunClearsLedger(t *testing.T) {
	ctx := context.Background()
	a := New(synthPolicy())

	if v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"helper.py"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("authoring write: got %v, want Allow", v.Kind)
	}
	// Before reset: recognized -> denied.
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"python helper.py internal/abi/x.go"}`)); v.Kind != abi.VerdictDeny {
		t.Fatalf("pre-reset synth-exec: got %v, want Deny", v.Kind)
	}
	a.ResetRun()
	// After reset: the ledger is empty, so the same exec is a plain script run again.
	if v := a.Adjudicate(ctx, inlineCall("Bash", `{"command":"python helper.py internal/abi/x.go"}`)); v.Kind != abi.VerdictAllow {
		t.Errorf("post-reset synth-exec: got %v/%s, want Allow (ledger should be cleared)", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestExecsAuthoredScript unit-tests the synth->exec detector directly across the
// invocation forms and the read/flag negatives.
func TestExecsAuthoredScript(t *testing.T) {
	var ledger sync.Map
	rememberAuthored(&ledger, "helper.py")
	rememberAuthored(&ledger, "t") // an extension-less chmod +x target

	cases := []struct {
		cmd  string
		want bool
	}{
		{"python helper.py", true},
		{"python3 ./helper.py", true},
		{"python -u helper.py", true},   // skip the -u flag, first operand is the script
		{"node helper.py --flag", true}, // interpreter form with trailing args
		{"/usr/bin/python3 helper.py", true},
		{"./t", true},                      // direct exec of a chmod-authored binary
		{"sh -c 'echo hi'; ./t now", true}, // second sub-command execs the authored binary
		{"cat helper.py", false},           // a READ of the script, not an exec
		{"python other.py", false},         // a different, non-authored script
		{"pytest helper_test.py", false},   // pytest is neither interpreter nor authored
		{"grep helper.py .", false},        // helper.py as a search target, not exec'd
		{"", false},
	}
	for _, c := range cases {
		if got := execsAuthoredScript(c.cmd, &ledger); got != c.want {
			t.Errorf("execsAuthoredScript(%q) = %v, want %v", c.cmd, got, c.want)
		}
	}
}
