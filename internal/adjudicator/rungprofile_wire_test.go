package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// verdictKey collapses a verdict to the observable tuple a byte-identity check cares
// about: kind, reason, By, the bounded-disclosure claim, and the forensic Meta.
func verdictKey(v abi.Verdict) string {
	claim := ""
	if w, ok := v.Payload.(abi.WitnessPayload); ok {
		claim = w.Claim
	}
	return abi.ReasonName(v.Reason) + "|" + v.By + "|" + claim + "|" +
		v.Meta["posture"] + "|" + v.Meta["would_deny"] + "|" +
		map[bool]string{true: "T", false: "F"}[v.Kind == abi.VerdictAllow]
}

// rungBattery is a fixed set of calls that, between them, exercise every rung in
// Adjudicate: by-name deny, write-shaped self-modify, shell self-modify, an arg
// predicate, redact-transform, affirmative allow, and default-deny.
func rungBattery() []*abi.ToolCall {
	return []*abi.ToolCall{
		inlineCall("git_push", `{}`), // by-name Deny
		inlineCall("write_file", `{"path":"internal/abi/x.go","content":"x"}`), // self-modify (write-shaped)
		inlineCall("Bash", `{"command":"sed -i internal/kernel/k.go"}`),        // shell self-modify
		inlineCall("go_build", `{"password":"hunter2"}`),                       // redact transform
		inlineCall("git_status", `{}`),                                         // affirmative allow
		inlineCall("totally_unknown", `{}`),                                    // default-deny
		inlineCall("read_thing", `{}`),                                         // read-class allow-prefix
		inlineCall("write_file", `{"path":"out/safe.txt","content":"x"}`),      // write, not guarded -> allow? (not allowed name -> default-deny)
	}
}

// TestNilProfileEqualsEmptyProfileByteForByte is the #666 anchor: a nil Profile and an
// explicit zero RungProfile{} (which elides nothing) must produce IDENTICAL verdicts
// for every call — the drop-in guarantee that wiring the profile changed no behavior.
func TestNilProfileEqualsEmptyProfileByteForByte(t *testing.T) {
	ctx := context.Background()
	base := DevAgentPolicy()

	nilProf := base
	nilProf.Profile = nil
	emptyProf := base
	emptyProf.Profile = &RungProfile{} // elides nothing

	aNil := New(nilProf)
	aEmpty := New(emptyProf)

	for _, c := range rungBattery() {
		gn := verdictKey(aNil.Adjudicate(ctx, c))
		ge := verdictKey(aEmpty.Adjudicate(ctx, c))
		if gn != ge {
			t.Errorf("tool %q: nil profile %q != empty profile %q", c.Tool, gn, ge)
		}
	}
}

// TestReadClassElisionPreservesVerdicts proves the elision is SAFE: a profile that
// elides all write-only rungs for the read class produces the SAME verdicts as the
// nil profile across the battery — because riskClass keeps every write-capable call in
// the write class, where nothing is elided.
func TestReadClassElisionPreservesVerdicts(t *testing.T) {
	ctx := context.Background()
	base := DevAgentPolicy()

	nilProf := base
	nilProf.Profile = nil

	elideProf := base
	elideProf.Profile = (&RungProfile{}).elide(classRead,
		rungSelfModify, rungCmdSelfModify, rungSynthTool, rungLintWrite)

	aNil := New(nilProf)
	aElide := New(elideProf)

	for _, c := range rungBattery() {
		gn := verdictKey(aNil.Adjudicate(ctx, c))
		ge := verdictKey(aElide.Adjudicate(ctx, c))
		if gn != ge {
			t.Errorf("tool %q: read-class-elision changed a verdict: nil %q vs elide %q", c.Tool, gn, ge)
		}
	}
}

// TestWriteClassFloorSurvivesReadElision is the security anchor: eliding the write-only
// rungs for the READ class must NOT weaken the write floor — a self-modifying write is
// still denied SELF_MODIFY because riskClass puts it in classWrite, untouched.
func TestWriteClassFloorSurvivesReadElision(t *testing.T) {
	ctx := context.Background()
	p := DevAgentPolicy()
	p.Profile = (&RungProfile{}).elide(classRead,
		rungSelfModify, rungCmdSelfModify, rungSynthTool, rungLintWrite)
	a := New(p)

	v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"internal/abi/x.go","content":"x"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("self-modify write under read-elision: got %v/%s, want Deny/SELF_MODIFY",
			v.Kind, abi.ReasonName(v.Reason))
	}
}
