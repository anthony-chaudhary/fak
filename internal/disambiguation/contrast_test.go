package disambiguation

import (
	"strings"
	"testing"
)

func TestContrastValidationRejectsSelfContrast(t *testing.T) {
	entry := contrastEntry("agent kernel", "compute kernel", true, true)
	entry.Contrasts[0].CanonicalTerm = "agent kernel"
	if err := entry.Validate(); err == nil || !strings.Contains(err.Error(), "self-contrast") {
		t.Fatalf("Validate error = %v, want self-contrast rejection", err)
	}
}

func TestIndexRejectsUnknownContrastTarget(t *testing.T) {
	entry := contrastEntry("agent kernel", "missing kernel", false, true)
	if _, err := NewIndex([]Entry{entry}); err == nil || !strings.Contains(err.Error(), "unknown canonical target") {
		t.Fatalf("NewIndex error = %v, want unknown target rejection", err)
	}
}

func TestIndexRejectsAsymmetricRequiredPair(t *testing.T) {
	left := contrastEntry("agent kernel", "compute kernel", true, true)
	right := contrastEntry("compute kernel", "agent kernel", false, true)
	if _, err := NewIndex([]Entry{left, right}); err == nil || !strings.Contains(err.Error(), "asymmetric") {
		t.Fatalf("NewIndex error = %v, want asymmetric pair rejection", err)
	}
}

func TestContrastValidationRejectsEmptyExplanation(t *testing.T) {
	entry := contrastEntry("agent kernel", "compute kernel", false, true)
	entry.Contrasts[0].Explanation = ""
	if err := entry.Validate(); err == nil || !strings.Contains(err.Error(), "explanation is required") {
		t.Fatalf("Validate error = %v, want empty explanation rejection", err)
	}
}

func TestIndexAcceptsSymmetricRequiredForbiddenConflation(t *testing.T) {
	left := contrastEntry("agent kernel", "compute kernel", true, true)
	left.Identity.Aliases = []string{"fused agent kernel"}
	right := contrastEntry("compute kernel", "agent kernel", true, true)
	index, err := NewIndex([]Entry{left, right})
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	got, alias, ok := index.resolve("fused agent kernel")
	if !ok || alias != "fused agent kernel" || got.Identity.CanonicalTerm != "agent kernel" {
		t.Fatalf("resolve alias = (%q, %q, %v), want canonical agent kernel", got.Identity.CanonicalTerm, alias, ok)
	}
	contrast := got.Contrasts[0]
	if contrast.RequiredPair == nil || !*contrast.RequiredPair || contrast.ForbiddenConflation == nil || !*contrast.ForbiddenConflation {
		t.Fatalf("contrast flags = required %v forbidden %v, want true/true", contrast.RequiredPair, contrast.ForbiddenConflation)
	}
}

func contrastEntry(term, target string, required, forbidden bool) Entry {
	entry := SelfTestEntry()
	entry.Identity.CanonicalTerm = term
	entry.Identity.Aliases = []string{}
	entry.Contrasts = []Contrast{{
		CanonicalTerm:       target,
		Explanation:         term + " and " + target + " are distinct concepts.",
		RequiredPair:        boolPointer(required),
		ForbiddenConflation: boolPointer(forbidden),
	}}
	return entry
}
