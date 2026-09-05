package sessionread

import (
	"regexp"
	"slices"
	"strings"
	"testing"
)

// vocab_test.go — the fidelity pin for the read-op vocabulary spine (#4191). It mirrors
// the four checks internal/sessionctl/vocab_test.go makes on the control plane, adapted
// to the read plane's four fixed properties (capability / disclosure / evidence /
// refusal): every op is complete over closed sets, the registry covers EXACTLY the
// shipped read seams (worst-first children deliberately absent), every refusal token is
// grounded in the closed read-refusal vocabulary and disjoint from the control-plane and
// tool-refusal vocabularies, and Spec/Vocabulary hand out mutation-safe deep copies.

// closedTokenShape is the SCREAMING_SNAKE grammar every refusal token must match — the
// same discipline the abi and session refusal vocabularies hold.
var closedTokenShape = regexp.MustCompile(`^[A-Z][A-Z0-9_]+$`)

// knownCapabilities / knownDisclosures / knownEvidence are the closed property sets. A
// spec property outside its set is a completeness failure — the enum grew without the
// vocabulary (or the test) keeping up.
var (
	knownCapabilities = map[Capability]bool{CapReadSelf: true, CapReadFleet: true}
	knownDisclosures  = map[Disclosure]bool{DisclosureMetadata: true, DisclosureRedacted: true, DisclosureFull: true}
	knownEvidence     = map[Evidence]bool{EvidenceObserved: true, EvidenceWitnessed: true}
)

// TestReadVocabularyCompleteness pins that every registered op is complete over the
// closed property sets: a capability, disclosure, and evidence each drawn from its set;
// at least one well-formed refusal token; a non-empty summary; and no duplicate op.
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
	}

	// Ops() and the registry must agree on the op set.
	if got, want := len(Ops()), len(Vocabulary()); got != want {
		t.Fatalf("Ops() len %d != Vocabulary() len %d", got, want)
	}
}

// TestReadVocabularyCoversShippedSeams pins the registered op set to EXACTLY the read
// seams that ship at HEAD. A not-yet-shipped read op (a durable content-addressed store,
// a live transcript query, a transcript-event subscribe, an MCP resource surface, the
// supervisor observe loop) is deliberately absent — it registers its row when its child
// lands. This test fails loudly if a seam is added to (or dropped from) the registry
// without updating the expected set, so the spine cannot silently drift from what ships.
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

// TestReadRefusalTokensGrounded pins three invariants on the read-refusal vocabulary:
//   - closure: every token any op declares is in ReadRefusalTokens(), and every token in
//     ReadRefusalTokens() is declared by at least one op (nothing invented, nothing dangling);
//   - category disjointness by namespace: every read-refusal token lives in the READ_
//     namespace. Unlike the control plane — whose vocabulary REUSES the abi/session tokens
//     and so grounds against them directly — this spine DEFINES its own tokens, so the
//     honest, self-contained guarantee that a read refusal never masquerades as a
//     tool-refusal (abi: TRUST_VIOLATION, POLICY_BLOCK, …) or a control-write refusal
//     (session: CONTROL_*) is that read tokens occupy a disjoint prefix. This holds against
//     every current AND future foreign vocabulary, not just today's two, and keeps this
//     tier-1 pure primitive's test hermetic (it imports no churny sibling to enumerate).
func TestReadRefusalTokensGrounded(t *testing.T) {
	const readNamespace = "READ_"

	closed := ReadRefusalTokens()
	closedSet := make(map[string]bool, len(closed))
	for _, tok := range closed {
		if !closedTokenShape.MatchString(tok) {
			t.Errorf("closed read-refusal token %q is not SCREAMING_SNAKE", tok)
		}
		if !strings.HasPrefix(tok, readNamespace) {
			t.Errorf("read-refusal token %q is outside the %q namespace that keeps it disjoint from the abi and control-write vocabularies", tok, readNamespace)
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

// TestContextRestoreIsFullWitnessedTaintGated binds the spine to the one seam where the
// taint-safe-outbound invariant is load-bearing today: context-restore is the sole
// full-disclosure op, it is WITNESSED (verbatim bytes fak authored), and it carries the
// READ_TAINT_WITHHELD token — the real ctxplan seal/tombstone gate. If a future edit
// relaxes any of these, the read plane's outbound trust floor has moved and this fails.
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
		t.Errorf("context-restore must carry %q (the seal/tombstone gate); has %v", ReasonReadTaintWithheld, spec.RefusalReasons)
	}

	// It must be the ONLY full-disclosure op — full bytes crossing the boundary is the
	// exception the plane isolates, not a default.
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

// TestReadSpecCopyIsMutationSafe pins that Vocabulary() and Spec() hand out deep copies:
// mutating a returned spec's RefusalReasons slice must not corrupt the registry.
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

	// Unknown op is rejected.
	if _, ok := Spec(ReadOp("no-such-read")); ok {
		t.Error("Spec() reported an unknown op as known")
	}
}

// BenchmarkVocabulary measures the retrieval and deep-copying (slice cloning) of the
// complete registered read-op vocabulary.
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

// BenchmarkOps measures the extraction and allocation of the registered ReadOp
// identifier slice.
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

// BenchmarkSpec_Hit measures lookup and deep-copying of a single known ReadOp spec.
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

// BenchmarkSpec_Miss measures lookup behavior when probing an unregistered ReadOp token.
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

// BenchmarkSpec_CycleAll measures cycling lookup across all registered ReadOp tokens,
// simulating varied read operation dispatches.
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

// BenchmarkReadRefusalTokens measures the retrieval and cloning of the closed
// read-refusal token set.
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
