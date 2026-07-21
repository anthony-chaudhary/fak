package guideddecode

import "testing"

// The token stream a reasoning turn produces: an open think span, some reasoning,
// the reasoning-end marker, then the tool-call envelope the constrainer shapes.
var reasoningStream = []string{"<think>", "weigh", "options", "</think>", `{"name":"`, "get", `"`}

// TestMaskArmInactiveThroughReasoning proves the mask stays inactive for every
// token up to and including the reasoning-end marker's arrival, and active for
// every token after it — the reasoning→answer boundary.
func TestMaskArmInactiveThroughReasoning(t *testing.T) {
	a := NewMaskArm("</think>", InactiveUntilMarker)
	want := []bool{false, false, false, true, true, true, true}
	for i, tok := range reasoningStream {
		got := a.Emit(tok)
		if got != want[i] {
			t.Fatalf("token %d %q: active=%v, want %v", i, tok, got, want[i])
		}
	}
}

// TestMaskArmMarkerTokenArms isolates the flip: the token that completes the
// marker is the one that arms the latch.
func TestMaskArmMarkerTokenArms(t *testing.T) {
	a := NewMaskArm("</think>", InactiveUntilMarker)
	if a.Emit("thinking") {
		t.Fatal("pre-marker token armed the latch")
	}
	if a.Armed() {
		t.Fatal("latch armed before the marker was emitted")
	}
	if !a.Emit("</think>") {
		t.Fatal("marker token did not arm the latch")
	}
	if !a.Armed() {
		t.Fatal("latch reports not armed after the marker")
	}
}

// TestMaskArmOneWay proves the arm never re-disables: once past the reasoning-end
// marker, a later <think>/</think> pair cannot un-constrain the stream.
func TestMaskArmOneWay(t *testing.T) {
	a := NewMaskArm("</think>", InactiveUntilMarker)
	a.Emit("</think>") // arm
	for _, tok := range []string{"<think>", "second", "</think>", "answer"} {
		if !a.Emit(tok) {
			t.Fatalf("latch re-disabled on %q after arming", tok)
		}
	}
}

// TestMaskArmMarkerSplitAcrossTokens proves a marker broken across a token
// boundary is still detected.
func TestMaskArmMarkerSplitAcrossTokens(t *testing.T) {
	a := NewMaskArm("</think>", InactiveUntilMarker)
	if a.Emit("</thi") {
		t.Fatal("partial marker armed the latch early")
	}
	if !a.Emit("nk>") {
		t.Fatal("marker split across two tokens was not detected")
	}
}

// TestMaskArmCaseInsensitive mirrors the harness reasoning strip: an uppercase
// close tag still arms the latch.
func TestMaskArmCaseInsensitive(t *testing.T) {
	a := NewMaskArm("</think>", InactiveUntilMarker)
	if !a.Emit("</THINK>") {
		t.Fatal("uppercase marker did not arm the latch")
	}
}

// TestMaskArmEmptyMarkerDefaults proves an empty marker falls back to the
// qwen3-style close tag.
func TestMaskArmEmptyMarkerDefaults(t *testing.T) {
	a := NewMaskArm("", InactiveUntilMarker)
	if a.Emit("hi") {
		t.Fatal("unexpected arm before default marker")
	}
	if !a.Emit("</think>") {
		t.Fatal("empty marker did not default to the qwen3 close tag")
	}
}

// TestMaskActiveByTokenReasoningStream is the end-to-end whole-stream check: a
// reasoning stream is unconstrained through the close tag, then constrained from
// the marker token onward.
func TestMaskActiveByTokenReasoningStream(t *testing.T) {
	got := MaskActiveByToken(reasoningStream, "</think>", InactiveUntilMarker)
	want := []bool{false, false, false, true, true, true, true}
	if len(got) != len(want) {
		t.Fatalf("length: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("position %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

// TestMaskActiveByTokenNoMarkerDefaults proves the configurable default for a
// stream that never emits a reasoning-end marker: fail-open leaves the whole stream
// inactive, fail-closed leaves it active (byte-identical to constraining today).
func TestMaskActiveByTokenNoMarkerDefaults(t *testing.T) {
	stream := []string{`{"name":"`, "get", `"`}

	open := MaskActiveByToken(stream, "</think>", InactiveUntilMarker)
	for i, v := range open {
		if v {
			t.Fatalf("fail-open default: position %d active, want inactive", i)
		}
	}

	closed := MaskActiveByToken(stream, "</think>", ActiveWhenNoMarker)
	for i, v := range closed {
		if !v {
			t.Fatalf("fail-closed default: position %d inactive, want active", i)
		}
	}
}

// TestMaskActiveByTokenDeterministic proves repeated evaluation of the same stream
// yields the same result (wall-clock-free, no hidden state).
func TestMaskActiveByTokenDeterministic(t *testing.T) {
	a := MaskActiveByToken(reasoningStream, "</think>", InactiveUntilMarker)
	b := MaskActiveByToken(reasoningStream, "</think>", InactiveUntilMarker)
	if len(a) != len(b) {
		t.Fatalf("length drift: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("position %d drifted: %v vs %v", i, a[i], b[i])
		}
	}
}
