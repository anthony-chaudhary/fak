package executionroute

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// roundTrip persists a descriptor the way a real session store would — through
// JSON — and reads it back. Every fixture below routes the ROUND-TRIPPED value,
// so a field that fails to survive serialization changes the verdict and fails
// the test, rather than passing on an in-memory struct the store never sees.
func roundTrip(t *testing.T, d SessionDescriptor) SessionDescriptor {
	t.Helper()
	raw, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("marshal descriptor: %v", err)
	}
	var back SessionDescriptor
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal descriptor: %v", err)
	}
	return back
}

func claudeSession() SessionDescriptor {
	return SessionDescriptor{
		Version:          DescriptorVersion,
		ID:               "session-7",
		Harness:          "claude",
		Wire:             harnessprofile.WireAnthropic,
		ModelFamily:      "claude",
		ToolProtocol:     "anthropic-tools",
		TranscriptFormat: "anthropic-messages",
	}
}

// FIXTURE 1 — same envelope: every axis matches, so the session resumes in place.
func TestRouteCompatResumesInIdenticalEnvelope(t *testing.T) {
	source := claudeSession()
	source.RequiredState = []StateKind{StateMessages, StateToolCalls, StateThinking, StateProviderKV}
	target := claudeSession()
	target.ID = ""

	dec, err := RouteCompat(roundTrip(t, source), roundTrip(t, target))
	if err != nil {
		t.Fatal(err)
	}
	if dec.Verdict != CompatIdentical {
		t.Fatalf("verdict=%q want %q (reason: %s)", dec.Verdict, CompatIdentical, dec.Reason)
	}
	if dec.Action != SessionResume {
		t.Fatalf("action=%q want %q", dec.Action, SessionResume)
	}
	if dec.Refused {
		t.Fatal("identical envelope must not refuse")
	}
	if len(dec.Untranslatable) != 0 {
		t.Fatalf("identical envelope stranded state: %+v", dec.Untranslatable)
	}
	// Every axis is explained, not just the differing ones.
	if len(dec.Axes) != len(compatAxes) {
		t.Fatalf("explained %d axes, want all %d", len(dec.Axes), len(compatAxes))
	}
	for _, c := range dec.Axes {
		if !c.Match {
			t.Fatalf("axis %s reported a difference in an identical envelope: %s", c.Axis, c.Reason)
		}
		if c.Reason == "" {
			t.Fatalf("axis %s carries no explanation", c.Axis)
		}
	}
}

// FIXTURE 2 — compatible cross-harness fork: the harness changes, but the session
// requires only state that survives the crossing, so the fork is eligible.
func TestRouteCompatForksCompatibleCrossHarness(t *testing.T) {
	source := claudeSession()
	source.RequiredState = []StateKind{StateSystemPrompt, StateMessages}
	// A different harness that speaks the same wire, family, and formats — the
	// crossing a session CAN make.
	target := claudeSession()
	target.ID = ""
	target.Harness = "openai-generic"

	dec, err := RouteCompat(roundTrip(t, source), roundTrip(t, target))
	if err != nil {
		t.Fatal(err)
	}
	if dec.Verdict != CompatTranslatable {
		t.Fatalf("verdict=%q want %q (reason: %s)", dec.Verdict, CompatTranslatable, dec.Reason)
	}
	if dec.Action != SessionFork {
		t.Fatalf("action=%q want %q", dec.Action, SessionFork)
	}
	if dec.Refused {
		t.Fatal("a translatable cross-harness fork must not refuse")
	}
	if dec.ID != "session-7" {
		t.Fatalf("fork lost the source id: %q", dec.ID)
	}
	if !dec.Differs(AxisHarness) {
		t.Fatal("harness axis should be reported as differing")
	}
	// The harness axis alone strands nothing: no state kind is bound to it.
	if len(dec.Untranslatable) != 0 {
		t.Fatalf("a harness-only change stranded state: %+v", dec.Untranslatable)
	}
}

// FIXTURE 3 — incompatible state refusal: the fork is refused because required
// state is bound to an axis that changed. This is the failure class the caller
// booleans could not see — Portable=true asserts a move that would silently drop
// the very state the caller declared it could not lose.
func TestRouteCompatRefusesForkWhenRequiredStateCannotTranslate(t *testing.T) {
	source := claudeSession()
	source.RequiredState = []StateKind{StateMessages, StateThinking}
	// A genuinely foreign envelope: different family, wire, protocol, and format.
	target := SessionDescriptor{
		Version:          DescriptorVersion,
		Harness:          "codex",
		Wire:             harnessprofile.WireOpenAIResponses,
		ModelFamily:      "gpt",
		ToolProtocol:     "openai-functions",
		TranscriptFormat: "openai-responses",
	}

	dec, err := RouteCompat(roundTrip(t, source), roundTrip(t, target))
	if err != nil {
		t.Fatal(err)
	}
	if dec.Verdict != CompatIncompatible {
		t.Fatalf("verdict=%q want %q (reason: %s)", dec.Verdict, CompatIncompatible, dec.Reason)
	}
	if !dec.Refused {
		t.Fatal("an untranslatable required state must refuse the fork")
	}
	if dec.Action != SessionStart {
		t.Fatalf("action=%q want %q: a refused fork may only start fresh", dec.Action, SessionStart)
	}
	// The refusal names the state and the axis that stranded it, field by field.
	block, ok := dec.Blocked(StateThinking)
	if !ok {
		t.Fatal("thinking state should be reported as untranslatable")
	}
	if !block.Required {
		t.Fatal("thinking was declared required; the block must say so")
	}
	if block.Axis != AxisModelFamily {
		t.Fatalf("thinking stranded on axis %q want %q", block.Axis, AxisModelFamily)
	}
	if dec.Reason == "" {
		t.Fatal("a refusal must carry a reason")
	}
}

// A state kind that is NOT required is dropped rather than fatal: losing a
// provider cache is a cost, not a correctness failure, so it must not refuse.
func TestRouteCompatDropsUnrequiredStateWithoutRefusing(t *testing.T) {
	source := claudeSession()
	source.RequiredState = []StateKind{StateSystemPrompt}
	target := SessionDescriptor{
		Version:          DescriptorVersion,
		Harness:          "codex",
		Wire:             harnessprofile.WireOpenAIResponses,
		ModelFamily:      "gpt",
		ToolProtocol:     "anthropic-tools",
		TranscriptFormat: "anthropic-messages",
	}

	dec, err := RouteCompat(roundTrip(t, source), roundTrip(t, target))
	if err != nil {
		t.Fatal(err)
	}
	if dec.Verdict != CompatTranslatable {
		t.Fatalf("verdict=%q want %q (reason: %s)", dec.Verdict, CompatTranslatable, dec.Reason)
	}
	if dec.Refused {
		t.Fatal("dropping unrequired state must not refuse the fork")
	}
	block, ok := dec.Blocked(StateProviderKV)
	if !ok {
		t.Fatal("provider cache should be reported as stranded by the wire change")
	}
	if block.Required {
		t.Fatal("provider cache was not required; the block must not be fatal")
	}
}

// The version gate fails CLOSED: a descriptor this build cannot interpret is
// refused outright, never read with unknown fields silently ignored.
func TestSessionDescriptorRefusesUninterpretableVersions(t *testing.T) {
	for _, tc := range []struct {
		name string
		d    SessionDescriptor
	}{
		{"unversioned", SessionDescriptor{ID: "s"}},
		{"future version", SessionDescriptor{Version: DescriptorVersion + 1, ID: "s"}},
		{"unknown state", SessionDescriptor{Version: DescriptorVersion, ID: "s", RequiredState: []StateKind{"telepathy"}}},
		{"duplicate state", SessionDescriptor{Version: DescriptorVersion, ID: "s", RequiredState: []StateKind{StateMessages, StateMessages}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.d.Validate(); err == nil {
				t.Fatal("expected an uninterpretable descriptor to be refused")
			}
			if _, err := RouteCompat(tc.d, claudeSession()); err == nil {
				t.Fatal("expected RouteCompat to refuse an uninterpretable descriptor")
			}
		})
	}
}

func TestRouteCompatRequiresASourceSession(t *testing.T) {
	source := claudeSession()
	source.ID = ""
	if _, err := RouteCompat(source, claudeSession()); err == nil {
		t.Fatal("expected a source descriptor with no ID to fail")
	}
}

// ---------------------------------------------------------------------------
// INTEGRATION — the descriptor channel governs the composed execution envelope.
// ---------------------------------------------------------------------------

// The descriptor OVERRIDES the caller's Portable assertion. This is the point of
// the issue: before, `Portable: true` forked unconditionally; now an incompatible
// move is refused no matter what the caller asserts.
func TestRouteRefusesAssertedPortableForkAcrossIncompatibleEnvelopes(t *testing.T) {
	source := claudeSession()
	source.RequiredState = []StateKind{StateThinking}
	target := SessionDescriptor{
		Version:          DescriptorVersion,
		Harness:          "codex",
		Wire:             harnessprofile.WireOpenAIResponses,
		ModelFamily:      "gpt",
		ToolProtocol:     "openai-functions",
		TranscriptFormat: "openai-responses",
	}

	decision, err := Route(Request{
		Model: modelroute.Subject{Aspect: modelroute.AspectRequest},
		Session: SessionSubject{
			ID:       "session-7",
			Portable: true, // the caller's assertion — the descriptor must overrule it
			Source:   &source,
			Target:   &target,
		},
	}, harnessprofile.Builtins(), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := decision.Session.Action; got != SessionStart {
		t.Fatalf("action=%q want %q: an asserted-portable but untranslatable fork must be refused", got, SessionStart)
	}
	if decision.Session.Compat == nil {
		t.Fatal("a descriptor-routed session must carry its compatibility record")
	}
	if !decision.Session.Compat.Refused {
		t.Fatal("the compatibility record must mark the fork refused")
	}
	if decision.Session.ID != "" {
		t.Fatalf("a refused move named session %q; it carries no prior state", decision.Session.ID)
	}
}

// THE FAILURE CLASS, captured as a differential. The boolean channel and the
// descriptor channel are handed the SAME move — a session whose thinking state is
// required, crossing into a foreign model family. The booleans fork it (they have
// no way to see the state cannot survive); the descriptors refuse it. The fork is
// the defect: it silently drops the state the caller declared it could not lose.
// This test fails on the pre-descriptor routing, where the only reachable answer
// was the fork.
func TestDescriptorRefusesTheForkTheBooleansCannotSee(t *testing.T) {
	source := claudeSession()
	source.RequiredState = []StateKind{StateThinking}
	target := SessionDescriptor{
		Version:          DescriptorVersion,
		Harness:          "codex",
		Wire:             harnessprofile.WireOpenAIResponses,
		ModelFamily:      "gpt",
		ToolProtocol:     "openai-functions",
		TranscriptFormat: "openai-responses",
	}
	subject := SessionSubject{ID: "session-7", Portable: true}

	byBooleans, err := routeSession(subject)
	if err != nil {
		t.Fatal(err)
	}
	if byBooleans.Action != SessionFork {
		t.Fatalf("boolean channel action=%q want %q (the behavior this issue replaces)", byBooleans.Action, SessionFork)
	}

	withDescriptors := subject
	withDescriptors.Source, withDescriptors.Target = &source, &target
	byDescriptor, err := routeSession(withDescriptors)
	if err != nil {
		t.Fatal(err)
	}
	if byDescriptor.Action == SessionFork {
		t.Fatal("descriptor channel forked an untranslatable move: the state the caller required cannot cross a model-family change")
	}
	if byDescriptor.Action != SessionStart || !byDescriptor.Compat.Refused {
		t.Fatalf("descriptor channel action=%q refused=%v want a refused start", byDescriptor.Action, byDescriptor.Compat.Refused)
	}
}

// Compaction stays orthogonal to portability: an eligible resume in a full context
// still refines to compact_resume.
func TestRouteCompactsResumeWhenDescriptorsMatchAndContextIsFull(t *testing.T) {
	source := claudeSession()
	target := claudeSession()
	target.ID = ""

	decision, err := Route(Request{
		Model: modelroute.Subject{Aspect: modelroute.AspectRequest},
		Session: SessionSubject{
			ID:                 "session-7",
			ContextUtilization: .91,
			Source:             &source,
			Target:             &target,
		},
	}, harnessprofile.Builtins(), modelroute.DefaultManifest())
	if err != nil {
		t.Fatal(err)
	}
	if got := decision.Session.Action; got != SessionCompactResume {
		t.Fatalf("action=%q want %q", got, SessionCompactResume)
	}
	if decision.Session.Compat == nil || decision.Session.Compat.Verdict != CompatIdentical {
		t.Fatal("a compacted resume must still record the identical-envelope verdict")
	}
}

// A malformed descriptor pair fails the whole route rather than silently falling
// back to the booleans.
func TestRouteFailsOnUninterpretableDescriptor(t *testing.T) {
	source := claudeSession()
	source.Version = DescriptorVersion + 1
	target := claudeSession()
	target.ID = ""

	_, err := Route(Request{
		Model:   modelroute.Subject{Aspect: modelroute.AspectRequest},
		Session: SessionSubject{ID: "session-7", Source: &source, Target: &target},
	}, harnessprofile.Builtins(), modelroute.DefaultManifest())
	if err == nil {
		t.Fatal("expected an uninterpretable descriptor to fail the route")
	}
}
