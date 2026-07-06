package sessionsearch

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// --- Acceptance item 1: FTS-backed recall over the guard session journal --------

// TestSearchRanksRelevantAboveNoise proves the inverted index is a real relevance
// ranker: a rare, discriminating hit outranks a store full of common noise, and a
// query with no overlap recalls nothing (no false positives).
func TestSearchRanksRelevantAboveNoise(t *testing.T) {
	ix := NewIndex()
	for i := 0; i < 20; i++ {
		ix.Add(Doc{ID: "noise", Ordinal: i, Source: SourceInteractive, Text: "bash status check ok routine log line"})
	}
	ix.Add(Doc{ID: "gold", Ordinal: 20, Source: SourceInteractive, Text: "websearch kerberos ticket renewal runbook"})

	hits := ix.Search("kerberos renewal", 3, 0)
	if len(hits) == 0 {
		t.Fatal("expected a hit for a query that matches an indexed doc")
	}
	if hits[0].Doc.ID != "gold" {
		t.Fatalf("expected the discriminating doc to rank first, got %q (score %.3f)", hits[0].Doc.ID, hits[0].Score)
	}
	if got := ix.Search("thisquerymatchesnothingatall", 3, 0); got != nil {
		t.Fatalf("a non-matching query must recall nothing, got %d hits", len(got))
	}
}

// TestCronDemotedBelowInteractive proves _order_for_recall's demotion: two docs
// with byte-identical text but different sources rank interactive-above-cron.
func TestCronDemotedBelowInteractive(t *testing.T) {
	ix := NewIndex()
	ix.Add(Doc{ID: "cron", Ordinal: 0, Source: SourceCron, Text: "auth token rotation runbook"})
	ix.Add(Doc{ID: "human", Ordinal: 10, Source: SourceInteractive, Text: "auth token rotation runbook"})

	hits := ix.Search("auth token rotation", 5, 0) // window 0 so both distinct regions return
	if len(hits) != 2 {
		t.Fatalf("expected both docs recalled, got %d", len(hits))
	}
	if hits[0].Doc.ID != "human" || hits[1].Doc.ID != "cron" {
		t.Fatalf("cron must be demoted below interactive; got order %q then %q", hits[0].Doc.ID, hits[1].Doc.ID)
	}
	if !(hits[0].Score > hits[1].Score) {
		t.Fatalf("interactive score %.3f must exceed demoted cron score %.3f", hits[0].Score, hits[1].Score)
	}
}

// TestRecallOverGuardJournal proves recall runs over the REAL guard tool-process
// journal contract: a valid .fak/toolproc/journal.jsonl parses through
// toolproc.ParseEvents, a query for "deadline" recalls the kill row, and the
// ±window links it to the neighbouring spawn's tool name. A corrupt journal is
// refused at the boundary rather than half-recalled.
func TestRecallOverGuardJournal(t *testing.T) {
	journal := strings.Join([]string{
		`{"kind":"spawn","call_id":"c1","session":"interactive-42","tool":"slow_fetch","at_unix_ms":1700000000000,"deadline_ms":30000}`,
		`{"kind":"kill","call_id":"c1","session":"interactive-42","reason":"TOOL_DEADLINE_EXCEEDED","at_unix_ms":1700000032000}`,
		`{"kind":"spawn","call_id":"c2","session":"cron-nightly-1","tool":"bg_tail","at_unix_ms":1700000001000}`,
		`{"kind":"exit","call_id":"c2","status":"ok","at_unix_ms":1700000002000}`,
	}, "\n")

	docs, err := DocsFromJournal(strings.NewReader(journal))
	if err != nil {
		t.Fatalf("DocsFromJournal over a valid journal: %v", err)
	}
	if len(docs) != 4 {
		t.Fatalf("expected 4 docs from 4 events, got %d", len(docs))
	}
	// The cron-session spawn must classify as cron.
	if docs[2].Source != SourceCron {
		t.Fatalf("a cron-* session must classify SourceCron, got %q", docs[2].Source)
	}

	ix := NewIndex()
	for _, d := range docs {
		ix.Add(d)
	}
	hits := ix.Search("deadline", 3, 1)
	if len(hits) == 0 {
		t.Fatal("expected the deadline kill to be recalled")
	}
	if !strings.Contains(strings.ToLower(hits[0].Doc.Text), "deadline") {
		t.Fatalf("top hit should be the deadline row, got %q", hits[0].Doc.Text)
	}
	linked := false
	for _, w := range hits[0].Window {
		if strings.Contains(w.Text, "slow_fetch") {
			linked = true
		}
	}
	if !linked {
		t.Fatal("the ±window should link the kill row to the neighbouring spawn's tool")
	}

	if _, err := DocsFromJournal(strings.NewReader(`{"kind":"bogus_kind","at_unix_ms":1}`)); err == nil {
		t.Fatal("a corrupt journal (unknown kind) must be refused, not half-recalled")
	}
}

// --- Acceptance item 2: cache-safe injection proven byte-stable on the prefix ---

// TestInjectionByteStableOnPrefix proves the injection appends the recalled span
// as a fresh suffix that leaves the ENTIRE prior prefix cacheable, witnessed by
// cachemeta.Diverge — and that the witness actually distinguishes a prefix-mutating
// injection (the contrast case) from a safe one.
func TestInjectionByteStableOnPrefix(t *testing.T) {
	prefix := []cachemeta.PromptSegment{
		{Kind: cachemeta.SegStable, Tokens: 100, Content: []byte("SYSTEM PROMPT: you are a careful kernel agent.")},
		{Kind: cachemeta.SegToolSchema, Tokens: 50, Content: []byte("TOOLS: search_repo, read_file, run_tests")},
		{Kind: cachemeta.SegMessage, Tokens: 20, Content: []byte("user: what killed the fetch tool last night?")},
	}
	const wantPrefixTokens = int64(170)

	recalled := RecalledSpan([]Hit{{Doc: Doc{Source: SourceInteractive, Text: "kill TOOL_DEADLINE_EXCEEDED interactive-42"}}})
	injected, proof := Inject(prefix, recalled)

	if !proof.PrefixStable {
		t.Fatalf("a pure suffix append must be prefix-stable; divergence=%+v", proof.Divergence)
	}
	if proof.PrefixTokens != wantPrefixTokens || proof.Divergence.StableTokens != wantPrefixTokens {
		t.Fatalf("the whole prefix must stay cacheable: prefixTokens=%d stableTokens=%d want %d",
			proof.PrefixTokens, proof.Divergence.StableTokens, wantPrefixTokens)
	}
	if proof.Divergence.FirstDivergeSeg != len(prefix) {
		t.Fatalf("divergence must begin exactly at the appended tail (seg %d), got %d", len(prefix), proof.Divergence.FirstDivergeSeg)
	}
	if proof.Divergence.SealedStop {
		t.Fatal("no prefix segment is sealed; SealedStop must be false")
	}
	if want := EstimateTokens(recalled); proof.Divergence.LostTokens != want {
		t.Fatalf("only the recalled suffix is re-billed: LostTokens=%d want %d", proof.Divergence.LostTokens, want)
	}
	if len(injected) != len(prefix)+1 {
		t.Fatalf("injection appends exactly one segment, got %d from %d", len(injected), len(prefix))
	}
	// Inject must not mutate the caller's prefix, and the prefix segments must be
	// byte-identical in the injected turn.
	if len(prefix) != 3 {
		t.Fatal("Inject mutated the caller's prefix slice length")
	}
	for i := range prefix {
		if string(injected[i].Content) != string(prefix[i].Content) {
			t.Fatalf("prefix segment %d changed bytes under injection", i)
		}
	}

	// Contrast: injecting the recalled span at the FRONT (mutating the prefix) must
	// NOT witness as stable — the witness has to be able to say no.
	bad := append([]cachemeta.PromptSegment{{Kind: cachemeta.SegMessage, Tokens: EstimateTokens(recalled), Content: []byte(recalled)}}, prefix...)
	if d := cachemeta.Diverge(prefix, bad); d.FirstDivergeSeg == len(prefix) {
		t.Fatal("a front-injection must diverge before the tail; the byte-stability witness would be meaningless otherwise")
	}
}

// --- Acceptance item 3: a recall-usefulness witness metric ----------------------

// TestUsefulnessWitnessReferenceAndBlindness proves the witness credits a recall
// only for DISTINCTIVE tokens it added that survived into the outcome, and flags
// recall blindness when an injected hit changed nothing.
func TestUsefulnessWitnessReferenceAndBlindness(t *testing.T) {
	hits := []Hit{{Doc: Doc{Text: "kerberos ticket renewal runbook"}}}
	const prior = "user asked about the login failure this morning"

	used := WitnessUsefulness(hits, prior, "I followed the kerberos renewal steps and it worked")
	if used.Referenced != 1 || used.Blindness || used.ReferencedRatio != 1 {
		t.Fatalf("a referenced recall must witness Referenced=1, no blindness: %+v", used)
	}

	blind := WitnessUsefulness(hits, prior, "I could not find anything relevant and gave up")
	if blind.Referenced != 0 || blind.Unreferenced != 1 || !blind.Blindness {
		t.Fatalf("an unused recall must witness blindness: %+v", blind)
	}

	// If the prior context already carried every distinctive token, the recall added
	// nothing creditable even when the outcome mentions those words.
	priorHasAll := "notes: kerberos ticket renewal runbook already on file"
	tautology := WitnessUsefulness(hits, priorHasAll, "the kerberos renewal runbook says step one")
	if tautology.Referenced != 0 || !tautology.Blindness {
		t.Fatalf("a recall that only echoes prior context must not be credited: %+v", tautology)
	}
}

// TestUsefulnessLedgerRow proves the witness lowers into a schema-stamped JSONL
// row a higher-tier consumer can fold like the memory-value ledger.
func TestUsefulnessLedgerRow(t *testing.T) {
	u := Usefulness{Hits: 3, Referenced: 2, Unreferenced: 1}
	raw, err := u.MarshalRow()
	if err != nil {
		t.Fatalf("MarshalRow: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("row is not valid JSONL: %v", err)
	}
	if got["schema"] != UsefulnessLedgerSchema {
		t.Fatalf("row schema = %v, want %s", got["schema"], UsefulnessLedgerSchema)
	}
	if got["hits"] != float64(3) || got["referenced"] != float64(2) || got["unreferenced"] != float64(1) {
		t.Fatalf("row counts wrong: %v", got)
	}
}
