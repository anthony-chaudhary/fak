package bench

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// rowsByName indexes a receipt's rows so a case can be asserted by name rather
// than by position (a corpus edit must not silently retarget an assertion).
func rowsByName(t *testing.T, rec CtxAdmissionReceipt) map[string]CtxAdmissionRow {
	t.Helper()
	out := make(map[string]CtxAdmissionRow, len(rec.Rows))
	for _, r := range rec.Rows {
		if _, dup := out[r.Name]; dup {
			t.Fatalf("duplicate case name %q in the corpus", r.Name)
		}
		out[r.Name] = r
	}
	return out
}

func mustRow(t *testing.T, rows map[string]CtxAdmissionRow, name string) CtxAdmissionRow {
	t.Helper()
	r, ok := rows[name]
	if !ok {
		t.Fatalf("case %q missing from the receipt", name)
	}
	return r
}

// TestCtxAdmission is the re-runnable witness for issue #5694. One deterministic
// run demonstrates every half the proof artifacts name:
//
//   - the ACCEPTED case — all-vocabulary prose, rendered through the resolved
//     template and counted by the resolved tokenizer, fits and is admitted;
//   - the REFUSED cases — six evidence gaps that fail closed with no measurement,
//     plus the one measured negative (over the window); and
//   - the ANTI-VACUITY witness — a request of IDENTICAL character length to the
//     accepted one that the contract refuses, so the run proves the tokenizer did
//     work character length could not.
//
// The run is pinned to a committed golden (the scrubbed, re-derivable receipt).
func TestCtxAdmission(t *testing.T) {
	rec := DefaultCtxAdmissionReceipt()

	if rec.Schema != CtxAdmissionSchema {
		t.Errorf("schema = %q, want %q", rec.Schema, CtxAdmissionSchema)
	}
	if rec.Provenance.Kind != ProvenanceSimulated {
		t.Errorf("provenance = %q, want %q", rec.Provenance.Kind, ProvenanceSimulated)
	}
	if rec.Cases != len(rec.Rows) || rec.Cases == 0 {
		t.Fatalf("cases = %d, rows = %d", rec.Cases, len(rec.Rows))
	}
	if rec.Admitted+rec.Refused != rec.Cases {
		t.Errorf("admitted %d + refused %d != cases %d", rec.Admitted, rec.Refused, rec.Cases)
	}
	if len(rec.UnexercisedRefusals) != 0 {
		t.Errorf("corpus leaves typed refusals unexercised: %v — the fixture set does not cover the contract",
			rec.UnexercisedRefusals)
	}
	if rec.CharHeuristicDisagreements == 0 {
		t.Errorf("no measured row disagreed with the chars-per-token heuristic; the run is vacuous")
	}
	if rec.Verdict != CtxVerdictDiscriminating {
		t.Errorf("verdict = %q, want %q", rec.Verdict, CtxVerdictDiscriminating)
	}

	rows := rowsByName(t, rec)

	// --- the known-positive: the signal can succeed ---
	pos := mustRow(t, rows, "fits-common-prose").Admission
	if !pos.Admitted() || pos.Refusal != "" {
		t.Fatalf("known positive = %q refusal=%q, want ADMITTED", pos.Decision, pos.Refusal)
	}
	if !pos.Measured || pos.Measurement == nil {
		t.Fatalf("known positive carries no measurement")
	}
	if pos.Tokenizer == nil || pos.Template == nil {
		t.Fatalf("known positive did not preserve the resolved identities it counted with")
	}
	if pos.Measurement.HeadroomTokens < 0 {
		t.Errorf("known positive headroom = %d, want >= 0", pos.Measurement.HeadroomTokens)
	}

	// --- the adversarial case: IDENTICAL characters, opposite decisions ---
	adv := mustRow(t, rows, "same-chars-rare-tokens").Admission
	if adv.Admitted() || adv.Refusal != CtxRefuseOverLimit {
		t.Fatalf("adversarial case = %q refusal=%q, want REFUSED %s", adv.Decision, adv.Refusal, CtxRefuseOverLimit)
	}
	if adv.EvidenceGap {
		t.Errorf("over-limit is a MEASURED conclusion, not an evidence gap")
	}
	if adv.Measurement.RenderedChars != pos.Measurement.RenderedChars {
		t.Fatalf("adversarial pair rendered_chars = %d vs %d; the whole case rests on them being EQUAL",
			adv.Measurement.RenderedChars, pos.Measurement.RenderedChars)
	}
	if !adv.Measurement.CharHeuristicWouldAdmit {
		t.Errorf("the chars-per-token heuristic would have REFUSED the adversarial request too; "+
			"the fixture no longer proves character length is insufficient (heuristic tokens=%d, limit=%d)",
			adv.Measurement.CharHeuristicTokens, adv.Measurement.ContextLimitTokens)
	}
	if adv.Measurement.RenderedPromptTokens <= pos.Measurement.RenderedPromptTokens {
		t.Errorf("rare-word request counted %d tokens, common-word request %d; want strictly more at equal length",
			adv.Measurement.RenderedPromptTokens, pos.Measurement.RenderedPromptTokens)
	}
	t.Logf("equal-length pair: %d chars -> common %d tokens (admitted), rare %d tokens (refused), heuristic %d, limit %d",
		pos.Measurement.RenderedChars, pos.Measurement.RenderedPromptTokens,
		adv.Measurement.RenderedPromptTokens, adv.Measurement.CharHeuristicTokens,
		adv.Measurement.ContextLimitTokens)

	// --- the boundary probes: the comparison is strictly-greater ---
	exact := mustRow(t, rows, "exact-fit-boundary").Admission
	if !exact.Admitted() {
		t.Errorf("exact-fit boundary = %q, want ADMITTED — a request that fills the window EXACTLY fits",
			exact.Decision)
	}
	if exact.Measurement.HeadroomTokens != 0 {
		t.Errorf("exact-fit headroom = %d, want 0", exact.Measurement.HeadroomTokens)
	}
	over := mustRow(t, rows, "one-token-over-boundary").Admission
	if over.Admitted() || over.Refusal != CtxRefuseOverLimit {
		t.Errorf("one-token-over boundary = %q refusal=%q, want REFUSED %s",
			over.Decision, over.Refusal, CtxRefuseOverLimit)
	}
	if over.Measurement.HeadroomTokens != -1 {
		t.Errorf("one-token-over headroom = %d, want -1", over.Measurement.HeadroomTokens)
	}

	// --- every evidence gap fails CLOSED with no measurement attached ---
	for _, row := range rec.Rows {
		a := row.Admission
		if a.Refusal == "" {
			continue
		}
		if !CtxEvidenceGap(a.Refusal) {
			continue
		}
		if a.Admitted() {
			t.Errorf("case %q refused %s but decision = %q", row.Name, a.Refusal, a.Decision)
		}
		if a.Measured || a.Measurement != nil {
			t.Errorf("case %q refused for missing evidence but still shipped a measurement; "+
				"a consumer could read the zero as a token count", row.Name)
		}
		if !a.EvidenceGap {
			t.Errorf("case %q refusal %s not flagged as an evidence gap", row.Name, a.Refusal)
		}
	}

	// --- determinism: the run re-derives byte-for-byte ---
	gotJSON, err := rec.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	againJSON, err := DefaultCtxAdmissionReceipt().JSON()
	if err != nil {
		t.Fatalf("JSON (second run): %v", err)
	}
	if !bytes.Equal(gotJSON, againJSON) {
		t.Errorf("DefaultCtxAdmissionReceipt() drifted between runs — the receipt is not deterministic")
	}

	golden := filepath.Join("testdata", "context_admission_receipt.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(gotJSON, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(gotJSON, "\n")) {
		t.Errorf("receipt drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// TestCtxAdmissionTypedRefusals drives EVERY refusal in the closed vocabulary
// directly against the public entry point, so each typed refusal is exercised on
// its own rather than only inside the corpus fold — and asserts the evidence-gap
// classification that separates "cannot tell" from "does not fit".
func TestCtxAdmissionTypedRefusals(t *testing.T) {
	body := ctxBodyOfChars(ctxCommonPhrase, 400)
	base := ctxRefRequest("refusal-probe", body)

	noReserve := base
	noReserve.ReservedCompletionTokens = 0

	huge := ctxRefRequest("refusal-probe-huge", ctxBodyOfChars(ctxRarePhrase, 8000))

	wrongTok := ctxRefResolution(ctxRefLimit)
	wrongTok.Tokenizer = NewCtxRefTokenizer(ctxOtherModel)

	wrongTmpl := ctxRefResolution(ctxRefLimit)
	wrongTmpl.Template = NewCtxRefTemplate(ctxOtherModel)

	noTok := ctxRefResolution(ctxRefLimit)
	noTok.Tokenizer = nil

	noTmpl := ctxRefResolution(ctxRefLimit)
	noTmpl.Template = nil

	cases := []struct {
		name        string
		req         CtxRequest
		res         CtxResolution
		wantRefusal string
		wantGap     bool
	}{
		{"tokenizer unresolved", base, noTok, CtxRefuseTokenizerUnresolved, true},
		{"tokenizer bound to another model", base, wrongTok, CtxRefuseTokenizerUnbound, true},
		{"template unresolved", base, noTmpl, CtxRefuseTemplateUnresolved, true},
		{"template bound to another model", base, wrongTmpl, CtxRefuseTemplateUnbound, true},
		{"context limit undeclared", base, ctxRefResolution(0), CtxRefuseLimitUndeclared, true},
		{"negative context limit is undeclared", base, ctxRefResolution(-1), CtxRefuseLimitUndeclared, true},
		{"completion reserve undeclared", noReserve, ctxRefResolution(ctxRefLimit), CtxRefuseReserveUndeclared, true},
		{"over the context limit", huge, ctxRefResolution(ctxRefLimit), CtxRefuseOverLimit, false},
	}

	seen := map[string]bool{}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AdmitCtxRequest(tc.req, tc.res)
			if got.Admitted() {
				t.Fatalf("decision = %q, want REFUSED", got.Decision)
			}
			if got.Refusal != tc.wantRefusal {
				t.Fatalf("refusal = %q, want %q", got.Refusal, tc.wantRefusal)
			}
			if got.EvidenceGap != tc.wantGap {
				t.Errorf("evidence_gap = %v, want %v", got.EvidenceGap, tc.wantGap)
			}
			// An evidence gap must never carry a measurement; the measured
			// negative must always carry one.
			if tc.wantGap && (got.Measured || got.Measurement != nil) {
				t.Errorf("evidence-gap refusal shipped a measurement")
			}
			if !tc.wantGap && (!got.Measured || got.Measurement == nil) {
				t.Errorf("measured refusal shipped no measurement — the overflow size is the useful part")
			}
			seen[got.Refusal] = true
		})
	}

	for _, code := range CtxRefusalVocabulary() {
		if !seen[code] {
			t.Errorf("typed refusal %s is in the vocabulary but no focused case exercises it", code)
		}
	}
}

// TestCtxAdmissionCharLengthCannotDecide is the direct repro of the defect #5694
// names. Two requests of byte-identical rendered length get opposite decisions.
// An implementation that admitted on character length — the "about four characters
// per token" rule — could not distinguish them and would fail this test.
func TestCtxAdmissionCharLengthCannotDecide(t *testing.T) {
	res := ctxRefResolution(ctxRefLimit)
	common := AdmitCtxRequest(ctxRefRequest("cmp-common", ctxBodyOfChars(ctxCommonPhrase, ctxFixtureChars)), res)
	rare := AdmitCtxRequest(ctxRefRequest("cmp-rare", ctxBodyOfChars(ctxRarePhrase, ctxFixtureChars)), res)

	if !common.Measured || !rare.Measured {
		t.Fatalf("both probes must measure; common=%v rare=%v", common.Measured, rare.Measured)
	}
	if common.Measurement.RenderedChars != rare.Measurement.RenderedChars {
		t.Fatalf("probe lengths differ (%d vs %d); the comparison is not controlled for characters",
			common.Measurement.RenderedChars, rare.Measurement.RenderedChars)
	}
	if common.Measurement.CharHeuristicTokens != rare.Measurement.CharHeuristicTokens {
		t.Fatalf("the char heuristic separated the probes (%d vs %d) — it must be blind to the difference",
			common.Measurement.CharHeuristicTokens, rare.Measurement.CharHeuristicTokens)
	}
	if !common.Admitted() {
		t.Errorf("common-word probe = %q (%s), want ADMITTED", common.Decision, common.Refusal)
	}
	if rare.Admitted() {
		t.Errorf("rare-word probe was ADMITTED at %d tokens against a %d limit",
			rare.Measurement.TotalTokens, rare.Measurement.ContextLimitTokens)
	}
	if !rare.Measurement.CharHeuristicWouldAdmit {
		t.Errorf("the char heuristic would also have refused the rare probe; the case proves nothing")
	}
}

// TestCtxAdmissionTemplateIsCounted proves the prompt template is part of the
// measurement and not an afterthought: the same body costs strictly more once
// rendered, so a body that fits raw can still be refused.
func TestCtxAdmissionTemplateIsCounted(t *testing.T) {
	body := ctxBodyOfChars(ctxCommonPhrase, 600)
	req := ctxRefRequest("template-probe", body)
	tok := NewCtxRefTokenizer(ctxRefModel)
	tmpl := NewCtxRefTemplate(ctxRefModel)

	bare := tok.CountTokens(body)
	rendered := tok.CountTokens(tmpl.Render(req))
	if rendered <= bare {
		t.Fatalf("rendered tokens %d <= bare body tokens %d; the template framing is not being counted",
			rendered, bare)
	}

	// Pin the window to exactly the BARE body plus its reserve: counting only the
	// body would admit, counting what is actually sent must refuse.
	res := ctxRefResolution(bare + ctxRefReserve)
	got := AdmitCtxRequest(req, res)
	if got.Admitted() {
		t.Errorf("request admitted at a window sized to the UNRENDERED body; the template overhead was ignored")
	}
	if got.Refusal != CtxRefuseOverLimit {
		t.Errorf("refusal = %q, want %q", got.Refusal, CtxRefuseOverLimit)
	}
}

// TestCtxAdmissionVacuousRunIsNamed proves the anti-vacuity guard is itself real:
// a corpus that is all-green but demonstrates nothing must reach the VACUOUS
// verdict, not the discriminating one.
func TestCtxAdmissionVacuousRunIsNamed(t *testing.T) {
	prov := simulatedProvenance("test", "TestCtxAdmissionVacuousRunIsNamed", "negative control")

	// All-admitted corpus: nothing refused, no refusal exercised, no disagreement.
	allPass := []CtxAdmissionCase{{
		Name:       "trivially-fits",
		Request:    ctxRefRequest("vac-1", ctxBodyOfChars(ctxCommonPhrase, 100)),
		Resolution: ctxRefResolution(ctxRefLimit),
	}}
	if got := RunCtxAdmission(allPass, prov); got.Verdict != CtxVerdictVacuous {
		t.Errorf("all-admitted corpus verdict = %q, want %q", got.Verdict, CtxVerdictVacuous)
	}

	// Agreeing corpus: the contract refuses, but so would the char heuristic, so
	// the run has not shown the tokenizer mattered.
	agreeing := []CtxAdmissionCase{
		{
			Name:       "fits",
			Request:    ctxRefRequest("vac-2", ctxBodyOfChars(ctxCommonPhrase, 100)),
			Resolution: ctxRefResolution(ctxRefLimit),
		},
		{
			Name:       "over-by-a-mile",
			Request:    ctxRefRequest("vac-3", ctxBodyOfChars(ctxRarePhrase, 40000)),
			Resolution: ctxRefResolution(ctxRefLimit),
		},
	}
	got := RunCtxAdmission(agreeing, prov)
	if got.CharHeuristicDisagreements != 0 {
		t.Fatalf("control corpus produced %d disagreement(s); it is no longer a control",
			got.CharHeuristicDisagreements)
	}
	if got.Verdict != CtxVerdictVacuous {
		t.Errorf("agreeing corpus verdict = %q, want %q — a refusal both methods reach proves nothing",
			got.Verdict, CtxVerdictVacuous)
	}
	// It must also name what it did not cover.
	if len(got.UnexercisedRefusals) == 0 {
		t.Errorf("control corpus exercised one refusal but reported none unexercised")
	}
	if len(got.RefusalCounts) != len(CtxRefusalVocabulary()) {
		t.Errorf("refusal_counts covers %d of %d vocabulary entries; the zeros must be visible in the artifact",
			len(got.RefusalCounts), len(CtxRefusalVocabulary()))
	}
}

// TestCtxAdmissionReceiptScrubbed proves the committed artifact is publishable:
// prompt text never appears, only request-salted digests that still bind the
// decision to the exact bytes counted.
func TestCtxAdmissionReceiptScrubbed(t *testing.T) {
	raw, err := DefaultCtxAdmissionReceipt().JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	blob := string(raw)
	for _, secret := range []string{ctxCommonPhrase, ctxRarePhrase, "you are a summarizer"} {
		if strings.Contains(blob, secret) {
			t.Errorf("receipt leaks prompt text %q", secret)
		}
	}

	// The digest binds bytes AND request identity: the same text under a different
	// request id must not correlate, and changed text must move the digest.
	a := ctxDigest("req-a", "hello world")
	b := ctxDigest("req-b", "hello world")
	c := ctxDigest("req-a", "hello worlds")
	if a == b {
		t.Errorf("digest is not request-salted: identical text correlates across requests")
	}
	if a == c {
		t.Errorf("digest did not move when the counted text changed")
	}
	if len(a) != 16 {
		t.Errorf("digest length = %d, want 16", len(a))
	}
	for _, row := range DefaultCtxAdmissionReceipt().Rows {
		if row.Admission.Measurement == nil {
			continue
		}
		if len(row.Admission.Measurement.RenderedPromptDigest) != 16 {
			t.Errorf("case %q digest = %q, want 16 hex chars",
				row.Name, row.Admission.Measurement.RenderedPromptDigest)
		}
	}
}

// TestCtxPieceTokenizerCountsVocabularyNotBytes pins the property the whole
// contract rests on: a vocabulary word costs one token however long it is, and an
// unknown word costs by pieces — so token cost is not a function of length.
func TestCtxPieceTokenizerCountsVocabularyNotBytes(t *testing.T) {
	tok := NewCtxRefTokenizer(ctxRefModel)

	if got := tok.CountTokens("summarize"); got != 1 {
		t.Errorf("in-vocabulary word cost %d tokens, want 1", got)
	}
	if got := tok.CountTokens("SUMMARIZE"); got != 1 {
		t.Errorf("vocabulary lookup is case sensitive: %d tokens, want 1", got)
	}
	// "abcdefghi" is 9 runes out of vocabulary -> 3 pieces at 3 runes each.
	if got := tok.CountTokens("abcdefghi"); got != 3 {
		t.Errorf("out-of-vocabulary word cost %d tokens, want 3", got)
	}
	// Punctuation is a token of its own; it is not free.
	if got := tok.CountTokens("the , the"); got != 3 {
		t.Errorf("punctuation cost: %d tokens, want 3", got)
	}
	if got := tok.CountTokens(""); got != 0 {
		t.Errorf("empty text cost %d tokens, want 0", got)
	}

	// The identity is what admission binds against.
	if id := tok.CtxTokenizerID(); id.Model != ctxRefModel || id.Name == "" || id.Revision == "" {
		t.Errorf("tokenizer identity = %+v, want a fully named identity bound to %q", id, ctxRefModel)
	}
}
