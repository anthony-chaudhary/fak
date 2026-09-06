package sessionread

import (
	"regexp"
	"slices"
	"strings"
	"testing"
)

var closedTokenShape = regexp.MustCompile(`^[A-Z][A-Z0-9_]+$`)

var (
	knownCapabilities = map[Capability]bool{CapReadSelf: true, CapReadFleet: true}
	knownDisclosures  = map[Disclosure]bool{DisclosureMetadata: true, DisclosureRedacted: true, DisclosureFull: true}
	knownEvidence     = map[Evidence]bool{EvidenceObserved: true, EvidenceWitnessed: true}
)

func TestReadVocabularyCompleteness(t *testing.T) {
	seen := make(map[ReadOp]bool)
	for _, s := range Vocabulary() {
		if s.Op == "" {
			t.Fatalf("registered spec with empty op: %+v", s)
		}
		if seen[s.Op] {
			t.Fatalf("duplicate op registered: %q", s.Op)
		}
		seen[s.Op] = true

		if !knownCapabilities[s.Capability] {
			t.Errorf("op %q: capability %q not in the closed set", s.Op, s.Capability)
		}
		if !knownDisclosures[s.Disclosure] {
			t.Errorf("op %q: disclosure %q not in the closed set", s.Op, s.Disclosure)
		}
		if !knownEvidence[s.Evidence] {
			t.Errorf("op %q: evidence %q not in the closed set", s.Op, s.Evidence)
		}
		if len(s.RefusalReasons) == 0 {
			t.Errorf("op %q: no refusal reasons (every op must name how it refuses)", s.Op)
		}
		for _, tok := range s.RefusalReasons {
			if !closedTokenShape.MatchString(tok) {
				t.Errorf("op %q: refusal token %q is not SCREAMING_SNAKE", s.Op, tok)
			}
		}
		if s.Summary == "" {
			t.Errorf("op %q: empty summary", s.Op)
		}

		if s.Capability == CapReadFleet && !slices.Contains(s.RefusalReasons, ReasonReadScopeDenied) {
			t.Errorf("op %q: fleet capability must declare %s", s.Op, ReasonReadScopeDenied)
		}
	}

	if got, want := len(Ops()), len(Vocabulary()); got != want {
		t.Fatalf("Ops() len %d != Vocabulary() len %d", got, want)
	}
}

func TestReadVocabularyCoversShippedSeams(t *testing.T) {
	shipped := []ReadOp{
		OpContextValue,
		OpContextSpans,
		OpContextRestore,
		OpSessionState,
		OpSessionChanges,
		OpCoherenceChanges,
		OpAuditEvents,
		OpTraceTaint,
		OpRunDigest,
		OpResumeHistory,
	}

	got := Ops()
	if len(got) != len(shipped) {
		t.Fatalf("registry has %d ops, expected %d shipped seams: got %v", len(got), len(shipped), got)
	}
	gotSet := make(map[ReadOp]bool, len(got))
	for _, op := range got {
		gotSet[op] = true
	}
	for _, op := range shipped {
		if !gotSet[op] {
			t.Errorf("shipped read seam %q missing from the registry", op)
		}
		if _, ok := Spec(op); !ok {
			t.Errorf("shipped read seam %q has no Spec()", op)
		}
	}
}

func TestReadRefusalTokensGrounded(t *testing.T) {
	const readNamespace = "READ_"

	closed := ReadRefusalTokens()
	closedSet := make(map[string]bool, len(closed))
	for _, tok := range closed {
		if !closedTokenShape.MatchString(tok) {
			t.Errorf("closed read-refusal token %q is not SCREAMING_SNAKE", tok)
		}
		if !strings.HasPrefix(tok, readNamespace) {
			t.Errorf("read-refusal token %q is outside namespace %q", tok, readNamespace)
		}
		closedSet[tok] = true
	}

	declared := make(map[string]bool)
	for _, s := range Vocabulary() {
		for _, tok := range s.RefusalReasons {
			declared[tok] = true
			if !closedSet[tok] {
				t.Errorf("op %q declares token %q not in ReadRefusalTokens()", s.Op, tok)
			}
		}
	}
	for tok := range closedSet {
		if !declared[tok] {
			t.Errorf("closed read-refusal token %q is declared by no op (dangling)", tok)
		}
	}
}

func TestContextRestoreIsFullWitnessedTaintGated(t *testing.T) {
	spec, ok := Spec(OpContextRestore)
	if !ok {
		t.Fatal("context-restore not registered")
	}
	if spec.Disclosure != DisclosureFull {
		t.Errorf("context-restore disclosure = %q, want %q", spec.Disclosure, DisclosureFull)
	}
	if spec.Evidence != EvidenceWitnessed {
		t.Errorf("context-restore evidence = %q, want %q", spec.Evidence, EvidenceWitnessed)
	}
	if !slices.Contains(spec.RefusalReasons, ReasonReadTaintWithheld) {
		t.Errorf("context-restore must carry %q; has %v", ReasonReadTaintWithheld, spec.RefusalReasons)
	}

	var full []ReadOp
	for _, s := range Vocabulary() {
		if s.Disclosure == DisclosureFull {
			full = append(full, s.Op)
		}
	}
	if len(full) != 1 || full[0] != OpContextRestore {
		t.Errorf("expected context-restore to be the sole full-disclosure op; got %v", full)
	}
}

func TestReadSpecCopyIsMutationSafe(t *testing.T) {
	spec, ok := Spec(OpContextRestore)
	if !ok {
		t.Fatal("context-restore not registered")
	}
	if len(spec.RefusalReasons) == 0 {
		t.Fatal("context-restore has no refusal reasons to mutate")
	}
	spec.RefusalReasons[0] = "MUTATED"

	fresh, _ := Spec(OpContextRestore)
	if slices.Contains(fresh.RefusalReasons, "MUTATED") {
		t.Error("mutating a Spec() copy corrupted the registry")
	}

	vocab := Vocabulary()
	if len(vocab) == 0 || len(vocab[0].RefusalReasons) == 0 {
		t.Fatal("vocabulary returned empty specs")
	}
	vocab[0].RefusalReasons[0] = "MUTATED_VOCAB"
	vocab[0].Op = "MUTATED_OP"

	freshVocab := Vocabulary()
	if slices.Contains(freshVocab[0].RefusalReasons, "MUTATED_VOCAB") || freshVocab[0].Op == "MUTATED_OP" {
		t.Error("mutating a Vocabulary() copy corrupted the registry")
	}

	tokens := ReadRefusalTokens()
	if len(tokens) == 0 {
		t.Fatal("empty refusal tokens")
	}
	tokens[0] = "MUTATED_TOKEN"
	freshTokens := ReadRefusalTokens()
	if freshTokens[0] == "MUTATED_TOKEN" {
		t.Error("mutating a ReadRefusalTokens() copy corrupted the registry")
	}

	ops := Ops()
	if len(ops) == 0 {
		t.Fatal("empty ops")
	}
	ops[0] = "MUTATED_OP"
	freshOps := Ops()
	if freshOps[0] == "MUTATED_OP" {
		t.Error("mutating an Ops() copy corrupted the registry")
	}

	if _, ok := Spec(ReadOp("no-such-read")); ok {
		t.Error("Spec() reported an unknown op as known")
	}
}

func BenchmarkVocabulary(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		specs := Vocabulary()
		if len(specs) == 0 {
			b.Fatal("unexpected empty vocabulary")
		}
	}
}

func BenchmarkOps(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ops := Ops()
		if len(ops) == 0 {
			b.Fatal("unexpected empty ops")
		}
	}
}

func BenchmarkSpec_Hit(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spec, ok := Spec(OpContextRestore)
		if !ok || spec.Op != OpContextRestore {
			b.Fatal("lookup failed for OpContextRestore")
		}
	}
}

func BenchmarkSpec_Miss(b *testing.B) {
	unknown := ReadOp("unregistered-read-op")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, ok := Spec(unknown)
		if ok {
			b.Fatal("expected unregistered op to miss")
		}
	}
}

func BenchmarkSpec_CycleAll(b *testing.B) {
	ops := Ops()
	if len(ops) == 0 {
		b.Fatal("empty ops")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		op := ops[i%len(ops)]
		spec, ok := Spec(op)
		if !ok || spec.Op != op {
			b.Fatalf("lookup failed for op %q", op)
		}
	}
}

func BenchmarkReadRefusalTokens(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokens := ReadRefusalTokens()
		if len(tokens) == 0 {
			b.Fatal("unexpected empty refusal tokens")
		}
	}
}
