package sessionreset

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// relayPolicyInput is a drained transcript whose recent turns and mid-stream body carry a
// unique sentinel, so a test can prove a seed did (or did not) lift verbatim transcript
// bytes. The first user line is the standing objective (sentinel-free), and a leading system
// preamble gives warm_prefix something to describe.
func relayPolicyInput(sentinel string) Input {
	return Input{
		Trace: "relay-leg-1",
		Messages: []Msg{
			{Role: "system", Content: strings.Repeat("You are a coding agent. ", 40)},
			{Role: "user", Content: "Ship the strict relay contributor policy."},
			{Role: "assistant", Content: "Investigating the fold. " + sentinel},
			{Role: "user", Content: "Any progress? " + sentinel},
			{Role: "assistant", Content: "Almost there. " + sentinel},
		},
	}
}

// TestRelayPolicyPointersOnly is the #1901 witness: a relay seed carries the objective pin and
// pointer-class parts only — no verbatim transcript bytes and no growing recap.
func TestRelayPolicyPointersOnly(t *testing.T) {
	const sentinel = "SENTINEL_TRANSCRIPT_BYTES_9f3a1c"
	in := relayPolicyInput(sentinel)

	pin := PinObjective("relay-obj-1", in)
	seed := BuildRelaySeed(pin, in)

	// The objective pin is carried: the fresh leg names which objective it continues.
	if !strings.Contains(seed.Recap, pin.PinID) {
		t.Fatalf("relay seed missing objective pin id %q:\n%s", pin.PinID, seed.Recap)
	}

	// The strict property: NO transcript bytes. The verbatim tail / recap sentinel that a
	// default seed would carry (via verbatim_tail / task_distill / model recap) must be absent.
	if strings.Contains(seed.Recap, sentinel) {
		t.Fatalf("relay seed leaked transcript bytes (%q):\n%s", sentinel, seed.Recap)
	}

	// Every fired part is in the closed pointer-only set.
	allowed := map[string]bool{}
	for _, n := range RelayContributorNames() {
		allowed[n] = true
	}
	sawPin := false
	for _, p := range seed.Parts {
		if !allowed[p.Name] {
			t.Errorf("relay seed carried non-pointer contributor %q (Text=%q)", p.Name, p.Text)
		}
		if p.Name == "objective_pin" {
			sawPin = true
		}
	}
	if !sawPin {
		t.Errorf("relay seed did not fold the objective_pin part; parts=%v", partNames(seed.Parts))
	}

	// The transcript-carrying / growing built-ins are specifically excluded.
	for _, banned := range []string{"verbatim_tail", "task_distill", "model_distill", "durability_facts"} {
		for _, p := range seed.Parts {
			if p.Name == banned {
				t.Errorf("relay seed must not include %q (it carries transcript bytes or grows with the session)", banned)
			}
		}
	}
}

// TestRelayPolicyIsStrictSubsetOfDefault proves the relay policy is a genuine tightening of
// the default seed — the default DOES carry the verbatim tail bytes the relay drops — so the
// default (non-relay) reset policy is provably unchanged by #1901.
func TestRelayPolicyIsStrictSubsetOfDefault(t *testing.T) {
	const sentinel = "SENTINEL_DEFAULT_CARRIES_44b2"
	in := relayPolicyInput(sentinel)

	// The default fold (whole registry — always has the four built-ins from init) keeps the
	// verbatim tail, so it contains the sentinel a relay seed must not.
	def := BuildSeed(in)
	if !strings.Contains(def.Recap, sentinel) {
		t.Fatalf("expected the default seed to carry the verbatim tail bytes %q:\n%s", sentinel, def.Recap)
	}

	relay := BuildRelaySeed(PinObjective("relay-obj-2", in), in)
	if strings.Contains(relay.Recap, sentinel) {
		t.Fatalf("relay seed leaked the tail bytes the default carries (%q):\n%s", sentinel, relay.Recap)
	}
	if len(relay.Recap) >= len(def.Recap) {
		t.Errorf("relay seed (%d bytes) should be strictly smaller than the default seed (%d bytes)",
			len(relay.Recap), len(def.Recap))
	}
}

// TestRelayPolicyZeroPinDropsObjectiveLine confirms the degenerate case is safe: with no pin
// the objective_pin line declines and the relay seed carries only the remaining pointer(s),
// never a panic and never a transcript-shaped fallback.
func TestRelayPolicyZeroPinDropsObjectiveLine(t *testing.T) {
	in := relayPolicyInput("SENTINEL_ZEROPIN_c71e")

	seed := BuildRelaySeed(ctxplan.ObjectivePin{}, in)
	for _, p := range seed.Parts {
		if p.Name == "objective_pin" {
			t.Fatalf("a zero pin must not fold an objective_pin part: %q", p.Text)
		}
		if !partAllowed(p.Name) {
			t.Errorf("relay seed carried non-pointer contributor %q", p.Name)
		}
	}
}

func partNames(parts []Part) []string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		out = append(out, p.Name)
	}
	return out
}

func partAllowed(name string) bool {
	for _, n := range RelayContributorNames() {
		if n == name {
			return true
		}
	}
	return false
}
