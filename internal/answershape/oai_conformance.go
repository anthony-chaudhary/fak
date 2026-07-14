package answershape

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// This file adds the OpenAI-compatible response-semantics conformance oracle: a
// deterministic, dependency-free judge over a captured chat-completion RESPONSE
// envelope. It is the structural dual of Measure — Measure judges only the
// message content's SHAPE; ConformOpenAI judges the envelope the content ships in
// (defaults, finish_reason, usage) and reuses Measure for the content-shape rung,
// so the two share one lane. It is the answershape-lane instance of the quality
// middle-contract (issue #4547, parent #4509): one independently verifiable layer
// between fak primitive correctness and coarse end-benchmark scores.
//
// The oracle walks a FIXED ladder of rungs and reports the FIRST actionable
// divergence (never a bag of every failure), fails CLOSED on missing or
// inconclusive evidence (absent provenance, usage, or closed-set is a divergence,
// never a pass), records the full provenance a case must carry (model, tokenizer,
// backend, seed/oracle, module revision, tolerance/baseline), and emits a
// scrubbed, portable replay bundle (secrets/PII redacted) so a failure replays in
// a clean environment. Every case is assigned an explicit CI Tier.
//
// Rung order (first divergence wins):
//
//	1. MISSING_EVIDENCE      — required provenance / usage / closed-set / choices absent (fail-closed).
//	2. STRUCTURE_DRIFT       — object, model, or message role diverges from the reference default.
//	3. FINISH_REASON_DRIFT   — finish_reason outside the closed set, or != the case's expected reason.
//	4. USAGE_DRIFT           — total != prompt+completion, or completion drifts from the token oracle beyond tolerance.
//	5. CONTENT_SHAPE_DEGENERATE — the message content trips the answershape degeneration guard (Measure).
//
// Cost: pure CPU, no I/O, O(response size); sub-millisecond per case. Default tier
// is PR (see Tier) — cheap enough to gate every pull request.
//
// ConformOpenAI is pure and deterministic: identical (case, Response, Reference)
// always yields an identical Conformance and an identical replay bundle.

// The closed reason vocabulary. A divergence always carries exactly one of these,
// mirroring the kernel's closed refusal vocabulary — a conformance failure is a
// typed, checkable value, never free-form prose.
const (
	ReasonMissingEvidence   = "MISSING_EVIDENCE"
	ReasonStructureDrift    = "STRUCTURE_DRIFT"
	ReasonFinishReasonDrift = "FINISH_REASON_DRIFT"
	ReasonUsageDrift        = "USAGE_DRIFT"
	ReasonContentShape      = "CONTENT_SHAPE_DEGENERATE"
)

// Tier names the CI cadence a conformance case runs at. A cheap, pure oracle like
// this defaults to PR (gates every pull request); a case wrapping a costly live
// serve would be assigned Nightly or Release instead.
type Tier string

const (
	TierPR      Tier = "pr"
	TierNightly Tier = "nightly"
	TierRelease Tier = "release"
)

// Provenance is the evidence a conformance case must record to be replayable and
// trustworthy: which model/tokenizer/backend produced the response, the seed or
// deterministic oracle that makes it reproducible, the code/module revision it was
// captured at, and the tolerance/baseline it is judged against. An incomplete
// Provenance fails the case CLOSED (missing evidence is never a pass).
type Provenance struct {
	Model     string `json:"model"`
	Tokenizer string `json:"tokenizer"`
	Backend   string `json:"backend"`  // engine / backend
	Oracle    string `json:"oracle"`   // seed or deterministic-oracle id
	Revision  string `json:"revision"` // code/module revision, e.g. "internal/answershape r12+gabcdef0"
	Baseline  string `json:"baseline"` // tolerance/baseline provenance
}

// missing returns the JSON paths of every required provenance field left empty, in
// a stable order, so the fail-closed rung can name the first one deterministically.
func (p Provenance) missing() []string {
	var m []string
	for _, f := range []struct{ path, val string }{
		{"provenance.model", p.Model},
		{"provenance.tokenizer", p.Tokenizer},
		{"provenance.backend", p.Backend},
		{"provenance.oracle", p.Oracle},
		{"provenance.revision", p.Revision},
		{"provenance.baseline", p.Baseline},
	} {
		if strings.TrimSpace(f.val) == "" {
			m = append(m, f.path)
		}
	}
	return m
}

// Usage mirrors the OpenAI chat-completion usage block. UsagePresent distinguishes
// an all-zero Usage that was genuinely reported from one that was OMITTED — an
// omitted usage is missing evidence, not a conforming zero.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Choice mirrors one element of an OpenAI chat-completion `choices` array (the
// fields the response-semantics contract checks).
type Choice struct {
	Index        int    `json:"index"`
	Role         string `json:"role"`
	Content      string `json:"content"`
	FinishReason string `json:"finish_reason"`
}

// Response is the captured OpenAI-compatible chat-completion envelope under test.
type Response struct {
	Object       string   `json:"object"`
	Model        string   `json:"model"`
	Choices      []Choice `json:"choices"`
	Usage        Usage    `json:"usage"`
	UsagePresent bool     `json:"usage_present"`
}

// Reference is the baseline a Response is judged against: the expected structural
// defaults, the closed finish_reason set and this case's expected reason, the
// usage tolerance, the content-shape Limits, the case Provenance, and the CI Tier.
type Reference struct {
	Object         string   // expected default object, e.g. "chat.completion"
	Role           string   // expected default message role, e.g. "assistant"
	Model          string   // expected model id ("" disables the model check)
	FinishReasons  []string // the closed set of allowed finish_reason values
	ExpectFinish   string   // the reason this deterministic case must finish with ("" disables)
	UsageTolerance int      // allowed |completion_tokens - oracle| drift, in tokens
	ShapeLimits    Limits   // the content-shape contract (judged by Measure)
	Provenance     Provenance
	Tier           Tier
}

// DeterministicTokens is the hermetic token oracle a conformance case scores usage
// against: a whitespace-delimited word count. It stands in for a real tokenizer so
// a case is reproducible with no model, weights, or seed — the "deterministic
// oracle" the acceptance contract allows in place of a seed.
func DeterministicTokens(s string) int {
	return len(strings.Fields(s))
}

// Divergence is the FIRST actionable difference the oracle found: a closed-vocab
// reason, the JSON path of the offending field, and the expected/actual values.
type Divergence struct {
	Reason   string `json:"reason"`
	Field    string `json:"field"`
	Detail   string `json:"detail"`
	Expected string `json:"expected,omitempty"`
	Actual   string `json:"actual,omitempty"`
}

// ReplayArtifact is the portable, scrubbed bundle a failure (or a pass) emits so it
// can be replayed in a clean environment. Its Response is a scrubbed copy, and
// JSON() runs a final redaction pass over the whole serialized bundle.
type ReplayArtifact struct {
	Case        string      `json:"case"`
	Tier        Tier        `json:"tier"`
	Provenance  Provenance  `json:"provenance"`
	Response    Response    `json:"response"`
	Divergence  *Divergence `json:"divergence,omitempty"`
	ShapeReport *Report     `json:"shape_report,omitempty"`
}

// JSON renders the replay bundle deterministically (struct field order, no maps)
// and runs a final scrub over the serialized bytes, so a secret cannot survive in
// a divergence detail or any nested string the field-level scrub did not reach.
func (a ReplayArtifact) JSON() []byte {
	b, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		// Marshal of these plain structs cannot fail; keep the surface total.
		return []byte(fmt.Sprintf("{%q:%q,%q:%q}", "case", a.Case, "marshal_error", err.Error()))
	}
	return []byte(scrub(string(b)))
}

// Conformance is the verdict: whether the Response conforms, the case identity and
// tier, the recorded provenance, the first divergence (nil iff conformant), and the
// scrubbed replay bundle.
type Conformance struct {
	Conformant bool
	Case       string
	Tier       Tier
	Provenance Provenance
	Divergence *Divergence
	Replay     ReplayArtifact
}

// ConformOpenAI judges resp against ref and returns the Conformance verdict. It is
// pure: no I/O, deterministic in (caseName, resp, ref).
func ConformOpenAI(caseName string, resp Response, ref Reference) Conformance {
	tier := ref.Tier
	if tier == "" {
		tier = TierPR
	}
	div, shape := firstDivergence(resp, ref)
	// The replay bundle is always built from a SCRUBBED copy of the response, so a
	// captured secret never reaches the artifact whether the case passes or fails.
	art := ReplayArtifact{
		Case:        caseName,
		Tier:        tier,
		Provenance:  ref.Provenance,
		Response:    scrubResponse(resp),
		Divergence:  div,
		ShapeReport: shape,
	}
	return Conformance{
		Conformant: div == nil,
		Case:       caseName,
		Tier:       tier,
		Provenance: ref.Provenance,
		Divergence: div,
		Replay:     art,
	}
}

// firstDivergence walks the fixed rung ladder and returns the first divergence (or
// nil) plus the content-shape Report when one was computed (attached to the replay
// even on a pass).
func firstDivergence(resp Response, ref Reference) (*Divergence, *Report) {
	// Rung 1: fail CLOSED on missing/inconclusive evidence.
	if d := missingEvidence(resp, ref); d != nil {
		return d, nil
	}
	primary := resp.Choices[0]

	// Rung 2: structural defaults.
	if resp.Object != ref.Object {
		return &Divergence{Reason: ReasonStructureDrift, Field: "object",
			Detail: "response object diverges from the reference default", Expected: ref.Object, Actual: resp.Object}, nil
	}
	if ref.Model != "" && resp.Model != ref.Model {
		return &Divergence{Reason: ReasonStructureDrift, Field: "model",
			Detail: "response model diverges from the reference", Expected: ref.Model, Actual: resp.Model}, nil
	}
	if primary.Role != ref.Role {
		return &Divergence{Reason: ReasonStructureDrift, Field: "choices[0].role",
			Detail: "message role diverges from the reference default", Expected: ref.Role, Actual: primary.Role}, nil
	}

	// Rung 3: finish reason — closed set first, then the case's expected value.
	if !contains(ref.FinishReasons, primary.FinishReason) {
		return &Divergence{Reason: ReasonFinishReasonDrift, Field: "choices[0].finish_reason",
			Detail:   "finish_reason is outside the closed set",
			Expected: strings.Join(ref.FinishReasons, "|"), Actual: primary.FinishReason}, nil
	}
	if ref.ExpectFinish != "" && primary.FinishReason != ref.ExpectFinish {
		return &Divergence{Reason: ReasonFinishReasonDrift, Field: "choices[0].finish_reason",
			Detail: "finish_reason diverges from the deterministic case oracle", Expected: ref.ExpectFinish, Actual: primary.FinishReason}, nil
	}

	// Rung 4: usage arithmetic, then token-oracle drift.
	if want := resp.Usage.PromptTokens + resp.Usage.CompletionTokens; resp.Usage.TotalTokens != want {
		return &Divergence{Reason: ReasonUsageDrift, Field: "usage.total_tokens",
			Detail:   "usage total does not equal prompt+completion",
			Expected: fmt.Sprintf("%d", want), Actual: fmt.Sprintf("%d", resp.Usage.TotalTokens)}, nil
	}
	if oracle := DeterministicTokens(primary.Content); abs(resp.Usage.CompletionTokens-oracle) > ref.UsageTolerance {
		return &Divergence{Reason: ReasonUsageDrift, Field: "usage.completion_tokens",
			Detail:   fmt.Sprintf("completion_tokens drifts from the token oracle by %d > tolerance %d", abs(resp.Usage.CompletionTokens-oracle), ref.UsageTolerance),
			Expected: fmt.Sprintf("%d", oracle), Actual: fmt.Sprintf("%d", resp.Usage.CompletionTokens)}, nil
	}

	// Rung 5: content shape (the answershape lane linkage). Always computed so the
	// replay carries the shape report even on a pass.
	rep := Measure([]byte(primary.Content), ref.ShapeLimits)
	if rep.Degenerate {
		return &Divergence{Reason: ReasonContentShape, Field: "choices[0].content",
			Detail:   "message content trips the answershape degeneration guard: " + strings.Join(rep.Reasons, "; "),
			Expected: "in-shape", Actual: rep.dominantSignal()}, &rep
	}
	return nil, &rep
}

// missingEvidence returns the fail-closed divergence when a case lacks the evidence
// needed to reach a conclusive verdict — required provenance, a closed finish set,
// at least one choice, or a usage block. Missing or inconclusive evidence is never
// a pass.
func missingEvidence(resp Response, ref Reference) *Divergence {
	if m := ref.Provenance.missing(); len(m) > 0 {
		return &Divergence{Reason: ReasonMissingEvidence, Field: m[0],
			Detail: "case is missing required provenance: " + strings.Join(m, ", "), Expected: "non-empty", Actual: "empty"}
	}
	if len(ref.FinishReasons) == 0 {
		return &Divergence{Reason: ReasonMissingEvidence, Field: "reference.finish_reasons",
			Detail: "no closed finish_reason set to check against — inconclusive", Expected: "non-empty set", Actual: "empty"}
	}
	if len(resp.Choices) == 0 {
		return &Divergence{Reason: ReasonMissingEvidence, Field: "choices",
			Detail: "response carries no choices", Expected: ">=1 choice", Actual: "0"}
	}
	if !resp.UsagePresent {
		return &Divergence{Reason: ReasonMissingEvidence, Field: "usage",
			Detail: "response reported no usage block — usage evidence is required, never assumed", Expected: "usage present", Actual: "absent"}
	}
	return nil
}

// Secret / PII needles redacted from a replay bundle so it is safe to attach to a
// public issue or PR — the "scrubbed replay artifact" the contract requires.
var (
	reBearer = regexp.MustCompile(`(?i)bearer\s+[A-Za-z0-9._\-]+`)
	reAPIKey = regexp.MustCompile(`sk-[A-Za-z0-9_\-]{8,}`)
	reEmail  = regexp.MustCompile(`[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`)
)

// scrub redacts secrets and PII deterministically. Bearer tokens are matched before
// bare keys so a "Bearer sk-…" collapses to one marker.
func scrub(s string) string {
	s = reBearer.ReplaceAllString(s, "Bearer ***REDACTED***")
	s = reAPIKey.ReplaceAllString(s, "sk-***REDACTED***")
	s = reEmail.ReplaceAllString(s, "***EMAIL-REDACTED***")
	return s
}

// scrubResponse returns a copy of resp with every string field redacted, so the
// structured replay Response (not just its JSON) is safe to hand a consumer.
func scrubResponse(resp Response) Response {
	out := resp
	out.Model = scrub(resp.Model)
	out.Choices = make([]Choice, len(resp.Choices))
	for i, c := range resp.Choices {
		c.Role = scrub(c.Role)
		c.Content = scrub(c.Content)
		out.Choices[i] = c
	}
	return out
}

func contains(set []string, v string) bool {
	for _, s := range set {
		if s == v {
			return true
		}
	}
	return false
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
