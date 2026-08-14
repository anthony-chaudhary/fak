package humanctl

import "testing"

func TestIndexIsUniqueAndResolvable(t *testing.T) {
	seenVerbs := map[Verb]bool{}
	seenNames := map[string]Verb{}
	for _, d := range Index() {
		if seenVerbs[d.Verb] {
			t.Fatalf("duplicate verb %q", d.Verb)
		}
		seenVerbs[d.Verb] = true
		for _, name := range append([]string{string(d.Verb)}, d.Aliases...) {
			n := normalize(name)
			if prior, exists := seenNames[n]; exists && prior != d.Verb {
				t.Fatalf("name %q aliases both %s and %s", name, prior, d.Verb)
			}
			seenNames[n] = d.Verb
			got, ok := Lookup(name)
			if !ok || got.Verb != d.Verb {
				t.Fatalf("Lookup(%q) = %v, %v; want %s", name, got.Verb, ok, d.Verb)
			}
		}
	}
	if len(seenVerbs) < 15 {
		t.Fatalf("index has only %d verbs; want a useful spine", len(seenVerbs))
	}
}

func TestConcernDoesNotInventDiagnosis(t *testing.T) {
	d, ok := Lookup(" this   seems OFF ")
	if !ok || d.Verb != FlagConcern || d.Intent != "record concern without inventing a diagnosis" {
		t.Fatalf("ambiguous concern resolved as %#v, %v", d, ok)
	}
	instruction := Instruction{Verb: d.Verb, Text: "I don't know why"}
	if err := instruction.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestReasonAndTargetRemainLossless(t *testing.T) {
	i := Instruction{
		Verb: Redirect, Strength: StrengthHigh,
		Target: "the smallest end-to-end spine",
		Reason: "the broad taxonomy is outrunning its witness",
		Text:   "keep the useful inventory, but ship one real path first",
	}
	if err := i.Validate(); err != nil {
		t.Fatal(err)
	}
	if i.Target == "" || i.Reason == "" || i.Text == "" {
		t.Fatalf("instruction lost human context: %#v", i)
	}
}

func TestComposeDirectionThenAssurance(t *testing.T) {
	program, err := Compose(
		Instruction{Verb: Reinforce, Reason: "the measured path is promising"},
		Instruction{Verb: Verify, Target: "net-true latency"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(program) != 2 || program[0].EffectiveStrength() != StrengthHigh {
		t.Fatalf("unexpected program %#v", program)
	}
}

func TestComposeRejectsWorkAfterStopOrPause(t *testing.T) {
	for _, terminal := range []Verb{Stop, Pause} {
		_, err := Compose(Instruction{Verb: terminal}, Instruction{Verb: Verify, Target: "result"})
		if err == nil {
			t.Fatalf("Compose accepted work after %s", terminal)
		}
	}
}

func TestRequiredTargetAndStrength(t *testing.T) {
	if err := (Instruction{Verb: Avoid}).Validate(); err == nil {
		t.Fatal("Avoid without a target should fail")
	}
	if err := (Instruction{Verb: Continue, Strength: Strength("maximum-ish")}).Validate(); err == nil {
		t.Fatal("unknown strength should fail")
	}
}

func TestIndexReturnsDefensiveCopy(t *testing.T) {
	first := Index()
	first[0].Aliases[0] = "corrupted"
	if got, ok := Lookup("this seems off"); !ok || got.Verb != FlagConcern {
		t.Fatalf("caller mutation corrupted catalog: %#v, %v", got, ok)
	}
}
