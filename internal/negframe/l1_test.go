package negframe

import "testing"

var testL1Domain = Domain{
	Name:    "transport",
	Members: []string{"walk", "bike", "bus", "train", "car"},
}

func TestL1DeMorganAllowSet(t *testing.T) {
	t.Parallel()
	got := RewriteL1("Do not use walk, bike, car.", testL1Domain)
	if got.Text != "Use only bus, train." || got.Admitted != 1 || got.Refused != 0 {
		t.Fatalf("RewriteL1 = %+v", got)
	}
}

func TestL1EquivalenceGateRefuses(t *testing.T) {
	t.Parallel()
	excluded := []string{"walk", "car"}
	for name, allowed := range map[string][]string{
		"missing member":  {"bike", "bus"},
		"retains banned":  {"walk", "bike", "bus", "train"},
		"invented member": {"bike", "bus", "train", "plane"},
		"duplicate":       {"bike", "bus", "train", "train"},
	} {
		t.Run(name, func(t *testing.T) {
			if L1Equivalent(testL1Domain, excluded, allowed) {
				t.Fatalf("L1Equivalent admitted %v", allowed)
			}
		})
	}
	if !L1Equivalent(testL1Domain, excluded, []string{"train", "bus", "bike"}) {
		t.Fatal("exact complement in a different order was refused")
	}

	got := RewriteL1("Do not use walk, hovercraft.", testL1Domain)
	if got.Text != "Do not use walk, hovercraft." || got.Admitted != 0 || got.Refused != 1 {
		t.Fatalf("ambiguous clause changed: %+v", got)
	}
}

func TestL1Idempotent(t *testing.T) {
	t.Parallel()
	once := RewriteL1("Do not use walk or car.", testL1Domain)
	twice := RewriteL1(once.Text, testL1Domain)
	if twice.Text != once.Text || twice.Admitted != 0 || twice.Refused != 0 {
		t.Fatalf("second pass = %+v; first = %+v", twice, once)
	}
}

func TestL1LeavesUnboundedVerbatim(t *testing.T) {
	t.Parallel()
	input := "Do not use walk, bike, car."
	for name, domain := range map[string]Domain{
		"absent":    {},
		"unnamed":   {Members: []string{"walk", "bike", "car"}},
		"duplicate": {Name: "bad", Members: []string{"walk", "WALK"}},
	} {
		t.Run(name, func(t *testing.T) {
			got := RewriteL1(input, domain)
			if got.Text != input || got.Admitted != 0 {
				t.Fatalf("RewriteL1 = %+v", got)
			}
		})
	}
}

func TestL1CodeFenceVerbatim(t *testing.T) {
	t.Parallel()
	input := "```text\nDo not use walk, bike, car.\n```\nKeep this positive."
	if got := RewriteL1(input, testL1Domain); got.Text != input || got.Admitted != 0 {
		t.Fatalf("fenced input changed: %+v", got)
	}
}

func TestL1RefusesAmbiguousConjunctionAndEmptyAllowSet(t *testing.T) {
	t.Parallel()
	for _, input := range []string{
		"Do not use walk and bike.",
		"Do not use walk, bike, bus, train, car.",
	} {
		got := RewriteL1(input, testL1Domain)
		if got.Text != input || got.Admitted != 0 || got.Refused != 1 {
			t.Fatalf("RewriteL1(%q) = %+v", input, got)
		}
	}
}

func TestNNFEquivalenceGated(t *testing.T) {
	got := RewriteL1("Do not use walk or car.", testL1Domain)
	if got.Text != "Use only bike, bus, train." || got.Admitted != 1 || got.Refused != 0 {
		t.Fatalf("NNF rewrite = %+v", got)
	}
	refused := RewriteL1("Do not use walk or hovercraft.", testL1Domain)
	if refused.Text != "Do not use walk or hovercraft." || refused.Refused != 1 {
		t.Fatalf("NNF fail-closed = %+v", refused)
	}
}

func TestPolarityPreservedByPositiveTransforms(t *testing.T) {
	input := "Never delete records. Do not use walk or car."
	once := RewriteL1(input, testL1Domain)
	twice := RewriteL1(once.Text, testL1Domain)
	if once.Text != "Never delete records. Use only bike, bus, train." || twice.Text != once.Text {
		t.Fatalf("polarity/idempotence once=%q twice=%q", once.Text, twice.Text)
	}
}
