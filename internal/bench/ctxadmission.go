// ctxadmission.go decides, BEFORE dispatch, whether an inference request fits the
// target's context window — and it decides it with the RESOLVED tokenizer and the
// RESOLVED prompt template, never with character length (issue #5694, a leaf of the
// #35 admission spine).
//
// The defect it closes is a measurement substitution. "About four characters per
// token" is a rule of thumb about English prose, not a property of a request: the
// same 2000 characters are ~320 tokens of common words and ~690 tokens of rare
// ones, because a tokenizer charges per vocabulary hit, not per byte. Admission
// that divides by four therefore does not know whether the request fits. When it
// guesses low, the request is dispatched, truncated or rejected at the provider,
// and the arm still emits plausible surrounding telemetry — a vacuous treatment
// that reads as a measured one. That is the failure this file makes impossible to
// reach silently.
//
// So the contract counts the string that will ACTUALLY be sent. The prompt template
// is applied first (role markers, system framing, and the generation trailer are
// real tokens the request must pay for), and the rendered result is counted by the
// tokenizer that is resolved FOR THIS MODEL. Both carry an identity, and an
// identity that does not bind to the dispatch target is refused rather than used:
// a tokenizer resolved for a different model produces a confident number about the
// wrong vocabulary, which is worse than no number at all.
//
// Seven typed refusals are managed distinctly, because "we cannot tell" and "it
// does not fit" are different facts and only one of them is a measurement:
//
//	TOKENIZER_UNRESOLVED          — no tokenizer; nothing can be counted
//	TOKENIZER_UNBOUND             — a tokenizer, but resolved for another model
//	TEMPLATE_UNRESOLVED           — no template; the string to count is unknown
//	TEMPLATE_UNBOUND              — a template, but resolved for another model
//	CONTEXT_LIMIT_UNDECLARED      — no window; nothing to compare against
//	COMPLETION_RESERVE_UNDECLARED — no headroom declared for the generation
//	OVER_CONTEXT_LIMIT            — MEASURED: rendered + reserved exceeds the window
//
// The first six are evidence gaps and fail closed with no measurement attached —
// the result carries Measured=false, so a consumer can never read a refusal-shaped
// zero as a token count. Only OVER_CONTEXT_LIMIT is a conclusion, and it ships the
// full accounting that produced it.
//
// The receipt is anti-vacuous by construction. Every measured row also records what
// the four-characters-per-token heuristic WOULD have decided, and the run counts the
// rows where the two disagree. A run in which the tokenizer never contradicts
// character length has not demonstrated that it is doing anything, and the verdict
// says so (CONTEXT_ADMISSION_VACUOUS) rather than passing quietly; the same verdict
// fires if the fixture set leaves any typed refusal unexercised. The committed
// corpus therefore includes two requests of IDENTICAL character length, one that
// fits and one that does not.
//
// The receipt is scrubbed (prompts appear only as request-salted digests, never as
// text) and deterministic — no clock, no randomness, no map iteration in output:
//
//	go test ./internal/bench -run TestCtxAdmission -count=1
//
// (the golden artifact regenerates into testdata/context_admission_receipt.json
// with UPDATE_GOLDEN=1).
package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"
	"unicode"
)

// CtxAdmissionSchema versions the machine-readable receipt.
const CtxAdmissionSchema = "fak-context-admission/1"

// The two decisions admission can reach. There is no third: a request the contract
// cannot establish as fitting is REFUSED, never admitted on the benefit of a doubt.
const (
	CtxAdmitted = "ADMITTED"
	CtxRefused  = "REFUSED"
)

// The typed refusal vocabulary. The first six are EVIDENCE GAPS — the contract
// cannot conclude, so it fails closed and attaches no measurement. The last is the
// measured negative: the accounting succeeded and the request does not fit.
const (
	CtxRefuseTokenizerUnresolved = "TOKENIZER_UNRESOLVED"
	CtxRefuseTokenizerUnbound    = "TOKENIZER_UNBOUND"
	CtxRefuseTemplateUnresolved  = "TEMPLATE_UNRESOLVED"
	CtxRefuseTemplateUnbound     = "TEMPLATE_UNBOUND"
	CtxRefuseLimitUndeclared     = "CONTEXT_LIMIT_UNDECLARED"
	CtxRefuseReserveUndeclared   = "COMPLETION_RESERVE_UNDECLARED"
	CtxRefuseOverLimit           = "OVER_CONTEXT_LIMIT"
)

// CtxRefusalVocabulary is the closed set, in the order admission evaluates it.
// Evidence gaps are checked before the measurement so a request missing its
// tokenizer is never reported as "over limit" on a number nobody could produce.
func CtxRefusalVocabulary() []string {
	return []string{
		CtxRefuseTokenizerUnresolved,
		CtxRefuseTokenizerUnbound,
		CtxRefuseTemplateUnresolved,
		CtxRefuseTemplateUnbound,
		CtxRefuseLimitUndeclared,
		CtxRefuseReserveUndeclared,
		CtxRefuseOverLimit,
	}
}

// CtxEvidenceGap reports whether code is a refusal for MISSING EVIDENCE (the
// contract could not conclude) rather than the measured negative conclusion.
func CtxEvidenceGap(code string) bool {
	return code != "" && code != CtxRefuseOverLimit
}

// ctxCharsPerTokenHeuristic is the "about four characters per token" rule this
// contract exists to replace. It is kept ONLY so every measured row can publish
// what the naive estimate would have decided — the anti-vacuity witness. It is
// never consulted for an admission decision.
const ctxCharsPerTokenHeuristic = 4

// --- resolved tokenizer ---------------------------------------------------

// CtxTokenizerID is a tokenizer's auditable identity. Model is the model the
// tokenizer was resolved FOR; admission refuses to count with a tokenizer whose
// Model does not match the dispatch target.
type CtxTokenizerID struct {
	Name     string `json:"name"`
	Revision string `json:"revision"`
	Model    string `json:"model"`
}

// CtxTokenizer counts tokens for exactly one declared identity. The interface is
// the seam a real GGUF/BPE vocabulary plugs into; the package ships one hermetic
// reference implementation so the contract is testable without a model file.
type CtxTokenizer interface {
	CtxTokenizerID() CtxTokenizerID
	CountTokens(text string) int
}

// CtxPieceTokenizer is the reference resolved tokenizer: a word in the declared
// vocabulary costs exactly one token, and a word outside it is split into
// fixed-size pieces. That is the property character length cannot model — cost
// tracks vocabulary coverage, not byte count — so two texts of identical length
// can differ in tokens by a factor of two or more, exactly as a real BPE
// vocabulary behaves on common prose versus rare technical terms.
type CtxPieceTokenizer struct {
	id         CtxTokenizerID
	vocab      map[string]bool
	pieceRunes int
}

// NewCtxPieceTokenizer builds a tokenizer resolved for model, whose vocabulary is
// the given words (matched case-insensitively) and whose out-of-vocabulary words
// cost one token per pieceRunes runes (rounded up).
func NewCtxPieceTokenizer(name, revision, model string, pieceRunes int, vocab []string) *CtxPieceTokenizer {
	if pieceRunes < 1 {
		pieceRunes = 1
	}
	set := make(map[string]bool, len(vocab))
	for _, w := range vocab {
		set[strings.ToLower(w)] = true
	}
	return &CtxPieceTokenizer{
		id:         CtxTokenizerID{Name: name, Revision: revision, Model: model},
		vocab:      set,
		pieceRunes: pieceRunes,
	}
}

// CtxTokenizerID reports the identity this tokenizer was resolved under.
func (t *CtxPieceTokenizer) CtxTokenizerID() CtxTokenizerID { return t.id }

// CountTokens counts the tokens in text. Deterministic and independent of map
// iteration order (the vocabulary is only probed, never ranged over).
func (t *CtxPieceTokenizer) CountTokens(text string) int {
	n := 0
	for _, w := range ctxSplitWords(text) {
		if t.vocab[strings.ToLower(w)] {
			n++
			continue
		}
		runes := len([]rune(w))
		n += (runes + t.pieceRunes - 1) / t.pieceRunes
	}
	return n
}

// ctxSplitWords splits text into word-ish units: runs of letters/digits/underscore
// are one unit, every other non-space rune is a unit of its own (punctuation and
// template markers are real tokens and must be paid for).
func ctxSplitWords(text string) []string {
	var out []string
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			out = append(out, string(cur))
			cur = cur[:0]
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_':
			cur = append(cur, r)
		case unicode.IsSpace(r):
			flush()
		default:
			flush()
			out = append(out, string(r))
		}
	}
	flush()
	return out
}

// --- resolved prompt template ---------------------------------------------

// CtxTemplateID is a prompt template's auditable identity, bound to the model whose
// serving stack renders it.
type CtxTemplateID struct {
	Name     string `json:"name"`
	Revision string `json:"revision"`
	Model    string `json:"model"`
}

// CtxPromptTemplate renders a request into the exact string that will be sent. The
// framing it adds is not decoration: role markers and the generation trailer are
// tokens the request pays for, and a body that fits before templating can fail
// after it.
type CtxPromptTemplate struct {
	ID            CtxTemplateID
	SystemOpen    string
	SystemClose   string
	TurnOpen      string // rendered as TurnOpen + role + TurnOpenClose
	TurnOpenEnd   string
	TurnClose     string
	GenerationCue string
}

// Render produces the literal prompt string for req.
func (t *CtxPromptTemplate) Render(req CtxRequest) string {
	var b strings.Builder
	if req.System != "" {
		b.WriteString(t.SystemOpen)
		b.WriteString(req.System)
		b.WriteString(t.SystemClose)
	}
	for _, turn := range req.Turns {
		b.WriteString(t.TurnOpen)
		b.WriteString(turn.Role)
		b.WriteString(t.TurnOpenEnd)
		b.WriteString(turn.Content)
		b.WriteString(t.TurnClose)
	}
	b.WriteString(t.GenerationCue)
	return b.String()
}

// --- the request and what must be resolved about it -----------------------

// CtxTurn is one conversational turn of the request.
type CtxTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CtxRequest is the inference request admission is asked about.
type CtxRequest struct {
	// RequestID identifies the request and salts its prompt digest.
	RequestID string `json:"request_id"`
	// Model is the DISPATCH TARGET — the model the request will actually run on.
	Model string `json:"model"`
	// System and Turns are the unrendered content.
	System string    `json:"-"`
	Turns  []CtxTurn `json:"-"`
	// ReservedCompletionTokens is the headroom the generation needs. The window
	// holds prompt AND completion, so a request that reserves nothing has not
	// established that it fits (COMPLETION_RESERVE_UNDECLARED).
	ReservedCompletionTokens int `json:"reserved_completion_tokens"`
}

// CtxTarget is the dispatch target's declared capacity.
type CtxTarget struct {
	Model string `json:"model"`
	// ContextLimitTokens is the hard window. Zero or negative means UNDECLARED —
	// admission refuses rather than assuming a default.
	ContextLimitTokens int `json:"context_limit_tokens"`
}

// CtxResolution is everything admission must have in hand before it may conclude.
// A nil Tokenizer or Template is an unresolved one, not an empty one.
type CtxResolution struct {
	Target    CtxTarget
	Tokenizer CtxTokenizer
	Template  *CtxPromptTemplate
}

// --- the typed result -----------------------------------------------------

// CtxMeasurement is the accounting that produced a decision. It is attached only
// when the contract could actually count (Measured=true on the admission); on an
// evidence-gap refusal it is absent entirely, so a zero can never be misread as a
// measured token count.
type CtxMeasurement struct {
	// RenderedPromptDigest binds the decision to the exact string counted, without
	// publishing it (request-salted, so the same text under two request IDs does
	// not correlate).
	RenderedPromptDigest string `json:"rendered_prompt_digest"`
	RenderedChars        int    `json:"rendered_chars"`
	RenderedPromptTokens int    `json:"rendered_prompt_tokens"`
	ReservedCompletion   int    `json:"reserved_completion_tokens"`
	TotalTokens          int    `json:"total_tokens"`
	ContextLimitTokens   int    `json:"context_limit_tokens"`
	// HeadroomTokens is limit - total; negative on an over-limit refusal, and the
	// size of the overflow is the useful part of the refusal.
	HeadroomTokens int `json:"headroom_tokens"`
	// CharHeuristicTokens and CharHeuristicWouldAdmit publish what dividing the
	// rendered length by four would have concluded. They are the anti-vacuity
	// witness and are NEVER an input to the decision.
	CharHeuristicTokens     int  `json:"char_heuristic_tokens"`
	CharHeuristicWouldAdmit bool `json:"char_heuristic_would_admit"`
}

// CtxAdmission is the typed, machine-readable result of one preflight.
type CtxAdmission struct {
	Schema    string `json:"schema"`
	RequestID string `json:"request_id"`
	Model     string `json:"model"`
	Decision  string `json:"decision"`
	// Refusal is one of CtxRefusalVocabulary, empty exactly when ADMITTED.
	Refusal string `json:"refusal,omitempty"`
	// EvidenceGap distinguishes "could not establish" from "established, does not
	// fit" — both refuse, but only the second is a measurement.
	EvidenceGap bool `json:"evidence_gap"`
	// Measured is true exactly when Measurement is present.
	Measured bool `json:"measured"`
	// Tokenizer / Template are the resolved identities used, preserved so the
	// decision can be re-derived. Absent when that half was never resolved.
	Tokenizer   *CtxTokenizerID `json:"tokenizer,omitempty"`
	Template    *CtxTemplateID  `json:"template,omitempty"`
	Measurement *CtxMeasurement `json:"measurement,omitempty"`
}

// Admitted reports whether the request may be dispatched. It is true for exactly
// one decision, so no refusal can be read as a soft pass.
func (a CtxAdmission) Admitted() bool { return a.Decision == CtxAdmitted }

// AdmitCtxRequest is the contract: decide whether req fits res.Target's window,
// using the resolved tokenizer and prompt template, and fail closed when the
// evidence needed to decide is absent.
func AdmitCtxRequest(req CtxRequest, res CtxResolution) CtxAdmission {
	out := CtxAdmission{
		Schema:    CtxAdmissionSchema,
		RequestID: req.RequestID,
		Model:     req.Model,
	}
	// Identities are preserved even on the refusal that rejects them — an auditor
	// needs to see WHICH tokenizer failed to bind, not merely that one did.
	if res.Tokenizer != nil {
		id := res.Tokenizer.CtxTokenizerID()
		out.Tokenizer = &id
	}
	if res.Template != nil {
		id := res.Template.ID
		out.Template = &id
	}

	refuse := func(code string) CtxAdmission {
		out.Decision = CtxRefused
		out.Refusal = code
		out.EvidenceGap = CtxEvidenceGap(code)
		return out
	}

	// --- evidence gaps, checked before any measurement is attempted ---
	if res.Tokenizer == nil {
		return refuse(CtxRefuseTokenizerUnresolved)
	}
	if out.Tokenizer.Model != req.Model {
		return refuse(CtxRefuseTokenizerUnbound)
	}
	if res.Template == nil {
		return refuse(CtxRefuseTemplateUnresolved)
	}
	if out.Template.Model != req.Model {
		return refuse(CtxRefuseTemplateUnbound)
	}
	if res.Target.ContextLimitTokens <= 0 {
		return refuse(CtxRefuseLimitUndeclared)
	}
	if req.ReservedCompletionTokens <= 0 {
		return refuse(CtxRefuseReserveUndeclared)
	}

	// --- the measurement: count the string that will actually be sent ---
	rendered := res.Template.Render(req)
	promptTokens := res.Tokenizer.CountTokens(rendered)
	total := promptTokens + req.ReservedCompletionTokens
	limit := res.Target.ContextLimitTokens
	heuristic := len(rendered) / ctxCharsPerTokenHeuristic

	out.Measured = true
	out.Measurement = &CtxMeasurement{
		RenderedPromptDigest:    ctxDigest(req.RequestID, rendered),
		RenderedChars:           len(rendered),
		RenderedPromptTokens:    promptTokens,
		ReservedCompletion:      req.ReservedCompletionTokens,
		TotalTokens:             total,
		ContextLimitTokens:      limit,
		HeadroomTokens:          limit - total,
		CharHeuristicTokens:     heuristic,
		CharHeuristicWouldAdmit: heuristic+req.ReservedCompletionTokens <= limit,
	}
	if total > limit {
		return refuse(CtxRefuseOverLimit)
	}
	out.Decision = CtxAdmitted
	return out
}

// ctxDigest is the scrubbing primitive: a prompt appears in the receipt only as a
// request-salted digest, so the artifact is publishable and still binds the
// decision to the exact bytes counted.
func ctxDigest(salt, s string) string {
	sum := sha256.Sum256([]byte(salt + "\x00" + s))
	return hex.EncodeToString(sum[:])[:16]
}

// --- the receipt ----------------------------------------------------------

// CtxAdmissionCase is one fixture: a request, what was resolved about it, and the
// reason the case is in the corpus.
type CtxAdmissionCase struct {
	Name       string
	Note       string
	Request    CtxRequest
	Resolution CtxResolution
}

// CtxAdmissionRow is one case's outcome in the receipt.
type CtxAdmissionRow struct {
	Name      string       `json:"name"`
	Note      string       `json:"note"`
	Admission CtxAdmission `json:"admission"`
}

// CtxRefusalCount is one typed refusal and how often the run reached it.
type CtxRefusalCount struct {
	Refusal string `json:"refusal"`
	Count   int    `json:"count"`
}

// Verdicts the receipt can reach.
const (
	// CtxVerdictDiscriminating: the run admitted at least one request, refused at
	// least one, exercised EVERY typed refusal, and — the part character length
	// cannot fake — reached at least one decision the four-chars-per-token
	// heuristic would have gotten wrong.
	CtxVerdictDiscriminating = "CONTEXT_ADMISSION_DISCRIMINATING"
	// CtxVerdictVacuous: the run did not demonstrate the contract. It may be all
	// green and still prove nothing, which is precisely the failure mode this
	// verdict exists to name.
	CtxVerdictVacuous = "CONTEXT_ADMISSION_VACUOUS"
)

// CtxAdmissionReceipt is the committed machine-readable artifact: every case's
// typed result plus the coverage and anti-vacuity accounting over them.
type CtxAdmissionReceipt struct {
	Schema     string            `json:"schema"`
	Provenance Provenance        `json:"provenance"`
	Cases      int               `json:"cases"`
	Admitted   int               `json:"admitted"`
	Refused    int               `json:"refused"`
	Rows       []CtxAdmissionRow `json:"rows"`
	// RefusalCounts covers the full vocabulary in evaluation order, including the
	// zeros — an unexercised refusal is visible in the artifact, not absent from it.
	RefusalCounts []CtxRefusalCount `json:"refusal_counts"`
	// UnexercisedRefusals names the typed refusals no case reached. Non-empty means
	// the corpus does not cover the contract.
	UnexercisedRefusals []string `json:"unexercised_refusals"`
	// CharHeuristicDisagreements counts measured rows where the naive
	// chars-per-token estimate would have decided differently. Zero means the
	// tokenizer never did work character length could not — a vacuous run.
	CharHeuristicDisagreements int    `json:"char_heuristic_disagreements"`
	Verdict                    string `json:"verdict"`
}

// JSON renders the receipt as the committed artifact.
func (r CtxAdmissionReceipt) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// RunCtxAdmission folds cases into the receipt. Deterministic: cases are evaluated
// in order and every aggregate is derived from that order, never from map ranging.
func RunCtxAdmission(cases []CtxAdmissionCase, provenance Provenance) CtxAdmissionReceipt {
	rec := CtxAdmissionReceipt{
		Schema:     CtxAdmissionSchema,
		Provenance: provenance,
		Cases:      len(cases),
		Rows:       make([]CtxAdmissionRow, 0, len(cases)),
	}
	counts := map[string]int{}
	for _, c := range cases {
		a := AdmitCtxRequest(c.Request, c.Resolution)
		rec.Rows = append(rec.Rows, CtxAdmissionRow{Name: c.Name, Note: c.Note, Admission: a})
		if a.Admitted() {
			rec.Admitted++
		} else {
			rec.Refused++
			counts[a.Refusal]++
		}
		if a.Measured && a.Measurement.CharHeuristicWouldAdmit != a.Admitted() {
			rec.CharHeuristicDisagreements++
		}
	}
	rec.UnexercisedRefusals = []string{}
	for _, code := range CtxRefusalVocabulary() {
		rec.RefusalCounts = append(rec.RefusalCounts, CtxRefusalCount{Refusal: code, Count: counts[code]})
		if counts[code] == 0 {
			rec.UnexercisedRefusals = append(rec.UnexercisedRefusals, code)
		}
	}
	sort.Strings(rec.UnexercisedRefusals)
	rec.Verdict = CtxVerdictVacuous
	if rec.Admitted > 0 && rec.Refused > 0 &&
		len(rec.UnexercisedRefusals) == 0 && rec.CharHeuristicDisagreements > 0 {
		rec.Verdict = CtxVerdictDiscriminating
	}
	return rec
}

// --- the committed corpus -------------------------------------------------

// ctxRefModel is the dispatch target the reference fixtures resolve against.
const ctxRefModel = "fak-ref-8b"

// ctxOtherModel is a DIFFERENT model, used only to build a tokenizer/template that
// resolves successfully but binds to the wrong target.
const ctxOtherModel = "fak-ref-70b"

// ctxRefVocab is the reference tokenizer's vocabulary — ordinary prose words that
// each cost exactly one token however long they are, which is what makes token cost
// diverge from character count.
var ctxRefVocab = []string{
	"a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
	"has", "have", "in", "is", "it", "of", "on", "or", "please", "that",
	"the", "this", "to", "was", "we", "were", "will", "with", "you",
	"assistant", "context", "limit", "model", "report", "request",
	"summarize", "system", "token", "user", "window",
}

// ctxRefPieceRunes: an out-of-vocabulary word costs one token per three runes. Real
// BPE vocabularies behave this way at the tail — a rare technical term is spelled
// out in pieces while a common word is a single id.
const ctxRefPieceRunes = 3

// NewCtxRefTokenizer builds the reference tokenizer resolved for model.
func NewCtxRefTokenizer(model string) *CtxPieceTokenizer {
	return NewCtxPieceTokenizer("fak-ref-piece", "v1", model, ctxRefPieceRunes, ctxRefVocab)
}

// NewCtxRefTemplate builds the reference prompt template for model. The markers are
// deliberately punctuation-heavy because that is the honest case: framing costs
// tokens, and a contract that counts only the body understates every request.
func NewCtxRefTemplate(model string) *CtxPromptTemplate {
	return &CtxPromptTemplate{
		ID:            CtxTemplateID{Name: "fak-ref-chat", Revision: "v1", Model: model},
		SystemOpen:    "<|start|>system\n",
		SystemClose:   "<|end|>\n",
		TurnOpen:      "<|start|>",
		TurnOpenEnd:   "\n",
		TurnClose:     "<|end|>\n",
		GenerationCue: "<|start|>assistant\n",
	}
}

// ctxCommonPhrase is all-vocabulary prose: cheap in tokens, ordinary in characters.
const ctxCommonPhrase = "the user will please summarize this report for the model with context"

// ctxRarePhrase is out-of-vocabulary technical prose: expensive in tokens, and by
// construction the SAME characters as the common one when both are cut to length.
const ctxRarePhrase = "antidisestablishmentarianism thermoluminescence electroencephalography"

// ctxFixtureChars is the identical character length both prose bodies are cut to.
// The whole adversarial case rests on this being equal for the two requests.
const ctxFixtureChars = 2000

// ctxRefLimit is the reference target's declared window. It is chosen so the
// four-chars-per-token heuristic admits BOTH prose bodies while the resolved
// tokenizer admits only the common one — without that gap the corpus could not
// prove character length is insufficient. At 2085 rendered characters the two
// bodies cost 382 and 714 real tokens against a 521-token heuristic estimate, so
// with a 256-token reserve the admissible band is [777, 969]; 870 sits mid-band,
// leaving ~90 tokens of margin on BOTH sides so the fixture cannot flip on a
// one-token change to the template or the vocabulary.
const ctxRefLimit = 870

// ctxRefReserve is the completion headroom the reference requests declare.
const ctxRefReserve = 256

// ctxBodyOfChars repeats phrase (space separated) and cuts the result to exactly n
// bytes, so two different phrases yield two bodies of identical character length.
func ctxBodyOfChars(phrase string, n int) string {
	var b strings.Builder
	for b.Len() < n {
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(phrase)
	}
	return b.String()[:n]
}

// ctxRefRequest builds a reference request with the given id and body.
func ctxRefRequest(id, body string) CtxRequest {
	return CtxRequest{
		RequestID: id,
		Model:     ctxRefModel,
		System:    "you are a summarizer",
		Turns: []CtxTurn{
			{Role: "user", Content: body},
		},
		ReservedCompletionTokens: ctxRefReserve,
	}
}

// ctxRefResolution is the fully resolved reference resolution at the given limit.
func ctxRefResolution(limit int) CtxResolution {
	return CtxResolution{
		Target:    CtxTarget{Model: ctxRefModel, ContextLimitTokens: limit},
		Tokenizer: NewCtxRefTokenizer(ctxRefModel),
		Template:  NewCtxRefTemplate(ctxRefModel),
	}
}

// DefaultCtxAdmissionCases is the committed corpus. It carries, on one run: the
// known-positive that proves the signal can succeed; the adversarial pair of
// IDENTICAL character length whose decisions differ; the two off-by-one boundary
// probes that pin the comparison; and one case per typed refusal, so no refusal in
// the vocabulary ships unexercised.
func DefaultCtxAdmissionCases() []CtxAdmissionCase {
	common := ctxBodyOfChars(ctxCommonPhrase, ctxFixtureChars)
	rare := ctxBodyOfChars(ctxRarePhrase, ctxFixtureChars)

	// The boundary probes derive their limit from the measured render, which is the
	// only way to sit exactly ON the edge: one at total == limit (must admit) and
	// one at total == limit+1 (must refuse). Together they pin the comparison to
	// strictly-greater and would catch an off-by-one in either direction.
	edge := ctxRefRequest("ctx-edge", ctxBodyOfChars(ctxCommonPhrase, 400))
	edgeTokens := NewCtxRefTokenizer(ctxRefModel).CountTokens(NewCtxRefTemplate(ctxRefModel).Render(edge))
	edgeExact := edgeTokens + ctxRefReserve

	unboundTok := ctxRefResolution(ctxRefLimit)
	unboundTok.Tokenizer = NewCtxRefTokenizer(ctxOtherModel)

	unboundTmpl := ctxRefResolution(ctxRefLimit)
	unboundTmpl.Template = NewCtxRefTemplate(ctxOtherModel)

	noTok := ctxRefResolution(ctxRefLimit)
	noTok.Tokenizer = nil

	noTmpl := ctxRefResolution(ctxRefLimit)
	noTmpl.Template = nil

	noReserve := ctxRefRequest("ctx-no-reserve", ctxBodyOfChars(ctxCommonPhrase, 400))
	noReserve.ReservedCompletionTokens = 0

	return []CtxAdmissionCase{
		{
			Name:       "fits-common-prose",
			Note:       "known positive: all-vocabulary prose, counted rendered, fits the window",
			Request:    ctxRefRequest("ctx-common", common),
			Resolution: ctxRefResolution(ctxRefLimit),
		},
		{
			Name: "same-chars-rare-tokens",
			Note: "adversarial: byte-for-byte the same LENGTH as fits-common-prose, " +
				"admitted by the chars-per-token heuristic, refused on real tokens",
			Request:    ctxRefRequest("ctx-rare", rare),
			Resolution: ctxRefResolution(ctxRefLimit),
		},
		{
			Name:       "exact-fit-boundary",
			Note:       "boundary: rendered + reserved is EXACTLY the window; must admit",
			Request:    edge,
			Resolution: ctxRefResolution(edgeExact),
		},
		{
			Name:       "one-token-over-boundary",
			Note:       "boundary: the same request one token over the window; must refuse",
			Request:    ctxRefRequest("ctx-edge-over", ctxBodyOfChars(ctxCommonPhrase, 400)),
			Resolution: ctxRefResolution(edgeExact - 1),
		},
		{
			Name:       "tokenizer-unresolved",
			Note:       "no tokenizer was resolved; nothing can be counted",
			Request:    ctxRefRequest("ctx-no-tokenizer", ctxBodyOfChars(ctxCommonPhrase, 400)),
			Resolution: noTok,
		},
		{
			Name:       "tokenizer-unbound",
			Note:       "a tokenizer resolved for a DIFFERENT model; a confident count of the wrong vocabulary",
			Request:    ctxRefRequest("ctx-wrong-tokenizer", ctxBodyOfChars(ctxCommonPhrase, 400)),
			Resolution: unboundTok,
		},
		{
			Name:       "template-unresolved",
			Note:       "no prompt template; the string that would be sent is unknown",
			Request:    ctxRefRequest("ctx-no-template", ctxBodyOfChars(ctxCommonPhrase, 400)),
			Resolution: noTmpl,
		},
		{
			Name:       "template-unbound",
			Note:       "a template resolved for a DIFFERENT model; the wrong framing would be counted",
			Request:    ctxRefRequest("ctx-wrong-template", ctxBodyOfChars(ctxCommonPhrase, 400)),
			Resolution: unboundTmpl,
		},
		{
			Name:       "limit-undeclared",
			Note:       "the target declares no context window; there is nothing to compare against",
			Request:    ctxRefRequest("ctx-no-limit", ctxBodyOfChars(ctxCommonPhrase, 400)),
			Resolution: ctxRefResolution(0),
		},
		{
			Name:       "completion-reserve-undeclared",
			Note:       "no completion headroom declared; the window holds prompt AND completion",
			Request:    noReserve,
			Resolution: ctxRefResolution(ctxRefLimit),
		},
	}
}

// DefaultCtxAdmissionReceipt is the re-derivable committed artifact for #5694.
func DefaultCtxAdmissionReceipt() CtxAdmissionReceipt {
	return RunCtxAdmission(DefaultCtxAdmissionCases(), simulatedProvenance(
		"go test ./internal/bench -run TestCtxAdmission -count=1",
		"bench.DefaultCtxAdmissionReceipt",
		"hermetic fixture corpus with a declared reference tokenizer and prompt template; "+
			"witnesses the admission CONTRACT and its typed refusals, not any provider's real vocabulary",
	))
}
