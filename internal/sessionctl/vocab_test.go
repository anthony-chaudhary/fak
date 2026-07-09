package sessionctl

import (
	"regexp"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// closedTokenShape is the SCREAMING_SNAKE shape every closed refusal token must take
// — the same shape the #2766 completeness test pins, so a control-op refusal can
// never drift into free text.
var closedTokenShape = regexp.MustCompile(`^[A-Z][A-Z0-9_]+$`)

// closed property vocabularies, restated here so the test FAILS if vocab.go adds a
// property value without adding it to its documented closed set.
var (
	knownCapabilities = []Capability{CapOperatorSend, CapOperatorControl}
	knownBoundaries   = []Boundary{BoundaryNextTurn, BoundaryQuiesce, BoundaryImmediate}
	knownWitnesses    = []WitnessKind{
		WitnessSplice, WitnessDirective, WitnessBoundaryStop,
		WitnessSameTurnWake, WitnessSamplingCap, WitnessSchedulerRead,
	}
)

// TestVocabularyCompleteness is the #2754 done condition: every registered op carries
// all four fixed properties, each drawn from its closed set, with no op missing a
// property and no duplicate op. This is the "no registered op missing a property"
// witness.
func TestVocabularyCompleteness(t *testing.T) {
	seen := map[ControlOp]bool{}
	for _, s := range Vocabulary() {
		if s.Op == "" {
			t.Fatalf("a spec has an empty op token: %+v", s)
		}
		if seen[s.Op] {
			t.Fatalf("op %q registered twice — the vocabulary must be a closed, unambiguous set", s.Op)
		}
		seen[s.Op] = true

		// 1. Capability — present and in the closed set.
		if !slices.Contains(knownCapabilities, s.Capability) {
			t.Fatalf("op %q capability %q not in the closed capability set %v", s.Op, s.Capability, knownCapabilities)
		}
		// 2. Boundary — present and in the closed set.
		if !slices.Contains(knownBoundaries, s.Boundary) {
			t.Fatalf("op %q boundary %q not in the closed boundary set %v", s.Op, s.Boundary, knownBoundaries)
		}
		// 3. Witness — present and in the closed set.
		if !slices.Contains(knownWitnesses, s.Witness) {
			t.Fatalf("op %q witness %q not in the closed witness set %v", s.Op, s.Witness, knownWitnesses)
		}
		// 4. RefusalReasons — non-empty, each a well-formed closed token.
		if len(s.RefusalReasons) == 0 {
			t.Fatalf("op %q declares no refusal reason — every op needs a closed refusal for its illegal-for-state submissions", s.Op)
		}
		for _, tok := range s.RefusalReasons {
			if !closedTokenShape.MatchString(tok) {
				t.Fatalf("op %q refusal token %q is not a closed SCREAMING_SNAKE token", s.Op, tok)
			}
		}
		// Summary is a human/audit aid, but a blank one signals an unfinished row.
		if s.Summary == "" {
			t.Fatalf("op %q has no summary", s.Op)
		}
	}
}

// TestVocabularyCoversShippedOps grounds the spine in the real, SHIPPED control verbs:
// the registered op set must be exactly the live op set at HEAD. A new shipped op must
// register its row here (or this names the gap); the not-yet-shipped add-constraint op
// (#2756) is deliberately absent until its child lands, matching the #2766 table.
func TestVocabularyCoversShippedOps(t *testing.T) {
	// The shipped live op set: `fak session` drive-state verbs + `fak signal` steer +
	// #2755 redirect. Kept in lockstep with internal/agent/loop_control_witness_test.go.
	want := []ControlOp{
		OpSteer, OpRedirect, OpPause, OpResume, OpCancel, OpThrottle, OpBudget, OpPriority,
	}
	got := Ops()
	if len(got) != len(want) {
		t.Fatalf("registered ops = %v, want exactly the shipped set %v", got, want)
	}
	for _, op := range want {
		if _, ok := Spec(op); !ok {
			t.Fatalf("shipped control op %q has no registered spec", op)
		}
	}
	for _, op := range got {
		if !slices.Contains(want, op) {
			t.Fatalf("registered op %q is not in the shipped live set — an unshipped op must not be registered", op)
		}
	}
}

// TestSteerWitnessIsSplice is the specific #2754 requirement: the steer op's
// witness-of-applied IS the splice assertion — the loop-side proof that operator prose
// was spliced into the turn input (the shipped assertion in
// internal/agent/loop_control_witness_test.go steer/applied that this contract
// generalizes). Its capability floor refuses with DEFAULT_DENY.
func TestSteerWitnessIsSplice(t *testing.T) {
	s, ok := Spec(OpSteer)
	if !ok {
		t.Fatal("steer op is not registered")
	}
	if s.Witness != WitnessSplice {
		t.Fatalf("steer witness = %q, want %q (the existing splice assertion)", s.Witness, WitnessSplice)
	}
	if s.Capability != CapOperatorSend {
		t.Fatalf("steer capability = %q, want %q (the a2achan send-right)", s.Capability, CapOperatorSend)
	}
	// The capability floor's closed refusal token must be present — a capless steer
	// send fails closed with DEFAULT_DENY (the #2766 steer/refused witness).
	if !slices.Contains(s.RefusalReasons, abi.ReasonName(abi.ReasonDefaultDeny)) {
		t.Fatalf("steer refusal reasons %v missing the capability-floor token %q", s.RefusalReasons, abi.ReasonName(abi.ReasonDefaultDeny))
	}
}

// TestVocabularyGroundedInClosedTokens is the fidelity cross-check that keeps the
// spine from drifting off the authoritative closed sets. It proves each op's declared
// refusal tokens come from the right source, and — for the drive-state ops — that the
// union of declared tokens is EXACTLY session.ControlRefusalTokens() (nothing invented,
// nothing dangling), mirroring the #2766 vocabulary-complete discipline.
func TestVocabularyGroundedInClosedTokens(t *testing.T) {
	driveTokens := map[string]bool{}
	for _, s := range Vocabulary() {
		switch s.Capability {
		case CapOperatorSend:
			// Injection ops: steer grounds in the abi send-floor codes; redirect in its
			// own closed reasons.
			switch s.Op {
			case OpSteer:
				want := []string{abi.ReasonName(abi.ReasonDefaultDeny), abi.ReasonName(abi.ReasonTrustViolation)}
				if !slices.Equal(s.RefusalReasons, want) {
					t.Fatalf("steer refusal tokens = %v, want the abi send-floor codes %v", s.RefusalReasons, want)
				}
			case OpRedirect:
				want := []string{string(RedirectMalformed), string(RedirectNoRedirectableState)}
				if !slices.Equal(s.RefusalReasons, want) {
					t.Fatalf("redirect refusal tokens = %v, want its closed reasons %v", s.RefusalReasons, want)
				}
			default:
				t.Fatalf("unexpected send-capability op %q", s.Op)
			}
		case CapOperatorControl:
			// Drive-state ops ground in the session control-refusal vocabulary.
			for _, tok := range s.RefusalReasons {
				if !slices.Contains(session.ControlRefusalTokens(), tok) {
					t.Fatalf("drive-state op %q token %q is not in session.ControlRefusalTokens() %v", s.Op, tok, session.ControlRefusalTokens())
				}
				driveTokens[tok] = true
			}
		default:
			// A new capability must extend this grounding logic, not bypass it: without
			// a case its tokens escape both the grounding and the disjointness checks.
			t.Fatalf("op %q has an un-grounded capability %q — add a case to the grounding switch", s.Op, s.Capability)
		}
	}
	// Nothing in the authoritative drive-state set dangles unregistered.
	for _, tok := range session.ControlRefusalTokens() {
		if !driveTokens[tok] {
			t.Fatalf("session control token %q is emitted by ControlRefusalFor but no registered op declares it", tok)
		}
	}
	// Category disjointness: the drive-state control tokens must never collide with the
	// per-tool abi refusal vocabulary (a control-write refusal is not a tool refusal —
	// the #2633 category discipline). steer/redirect tokens are exempt: they legitimately
	// ARE abi send-floor codes / redirect's own reasons.
	for tok := range driveTokens {
		if slices.Contains(abi.ReasonNames(), tok) {
			t.Fatalf("drive-state control token %q collides with the per-tool abi refusal vocabulary", tok)
		}
	}
}

// TestSpecLookupAndCopy pins the accessor contract: Spec round-trips every registered
// op and rejects an unknown one, and the exported slices are copies (a caller cannot
// mutate the closed registry through them).
func TestSpecLookupAndCopy(t *testing.T) {
	for _, op := range Ops() {
		if _, ok := Spec(op); !ok {
			t.Fatalf("Ops() returned %q but Spec(%q) missed", op, op)
		}
	}
	if _, ok := Spec("no-such-op"); ok {
		t.Fatal("Spec reported an unknown op as registered")
	}
	// Vocabulary returns a DEEP copy: mutating a scalar field, the outer slice, OR the
	// nested RefusalReasons slice must not disturb a second read of the registry.
	v := Vocabulary()
	if len(v) == 0 {
		t.Fatal("empty vocabulary")
	}
	v[0].Op = "mutated"
	if len(v[0].RefusalReasons) == 0 {
		t.Fatal("first spec has no refusal reasons to mutate")
	}
	v[0].RefusalReasons[0] = "CLOBBERED"
	// A Spec() result's slice must be independent of the registry too.
	if got, _ := Spec(OpSteer); len(got.RefusalReasons) > 0 {
		got.RefusalReasons[0] = "CLOBBERED"
	}
	fresh := Vocabulary()
	if fresh[0].Op != vocabulary[0].Op {
		t.Fatal("mutating the Vocabulary() slice leaked into the registry")
	}
	for _, tok := range fresh[0].RefusalReasons {
		if tok == "CLOBBERED" {
			t.Fatal("mutating a returned spec's RefusalReasons leaked into the registry (shallow copy)")
		}
	}
	if again, _ := Spec(OpSteer); again.Op != OpSteer {
		t.Fatal("registry op token was corrupted")
	}
}
