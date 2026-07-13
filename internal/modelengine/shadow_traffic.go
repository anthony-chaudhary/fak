package modelengine

// shadow_traffic.go is the scrubbed-shadow-request child of the quality spine
// (#4579, sibling of the load-parity child #4545): it evaluates a CANDIDATE
// engine against privacy-safe, production-like requests WITHOUT affecting the
// users those requests came from. A production request is mirrored into a
// shadow lane, scrubbed of every raw/identifying byte, replayed through the
// candidate, and its output compared against the live (reference) engine's
// output. A faithful candidate is invisible on the user path AND token-identical
// to the reference; an engine regression — a drifted sampler default, an early
// stop, a cache off-by-one — surfaces as the FIRST divergent token and is
// refused, while the user still receives exactly the reference response.
//
// Two invariants make this a shadow contract rather than just another diff:
//   - Isolation: the candidate (even a defective one) never reaches the user
//     path. shadowEvaluate returns the reference response the user would have
//     received regardless of the candidate, and never mutates the live request.
//   - Privacy: the candidate only ever sees a scrubbedRequest, a type that by
//     construction carries a content DIGEST and structural shape — never the raw
//     prompt or the end-user id. The replay artifact is built from the scrubbed
//     projection, so a failure bundle cannot leak what the request contained.
//
// The oracle is deterministic and self-contained (no real weights, no GPU, no
// network, no real users): a history-dependent decoder folds the scrubbed
// request's canonical bytes into a carried accumulator, so any engine defect
// that perturbs generation surfaces at the exact token where it first bites.
// Runtime/resource cost: pure in-process, microseconds per case, no external
// fixtures. Tier: PR (runs in the package unit gate).

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strings"
)

// shadowTrafficTier assigns every case in this child to the PR gate: it is a
// pure in-process unit test with no model download, accelerator, or live
// traffic, so it runs on every pull request rather than nightly or release.
const shadowTrafficTier = "PR"

// shadowRevision is the pinned code/module revision recorded in provenance.
const shadowRevision = "modelengine@shadow-traffic-1"

// shadowVocab is the small fixed vocabulary the deterministic decoder emits
// from. It is disjoint from the sibling oracles' vocabularies so a shadow trace
// is never confused with another child's trace in a failure bundle.
var shadowVocab = []string{"amber", "basalt", "cobalt", "dune", "ember", "flint", "garnet", "hazel"}

// liveRequest is a production-like request as it arrives on the user path. It
// carries raw, privacy-sensitive fields (the prompt text and the end-user id)
// that must NEVER cross into a shadow evaluation. It is the input to the live
// engine and the source that shadowScrub projects a privacy-safe copy from.
type liveRequest struct {
	id        string
	userID    string // PII — scrubbed away before any shadow evaluation.
	prompt    string // raw prompt text — scrubbed away before any shadow evaluation.
	model     string
	tokenizer string
	seed      uint64
	maxTokens int
}

// scrubbedRequest is the privacy-safe projection paired into shadow evaluation.
// It preserves only what a candidate engine needs to reproduce behavior — a
// content digest and structural shape — and by construction has NO field that
// can hold the raw prompt or the end-user id. Privacy is a type property here,
// not a runtime check that could be forgotten.
type scrubbedRequest struct {
	id           string
	model        string
	tokenizer    string
	seed         uint64
	maxTokens    int
	promptDigest string // sha256 of the raw prompt — stable, pairs requests, not reversible.
	promptLen    int    // structural feature only.
}

// shadowScrub projects a live request into its privacy-safe shadow form. The raw
// prompt is replaced by its digest and length; the user id is dropped entirely.
// The live request is read, never mutated.
func shadowScrub(live liveRequest) scrubbedRequest {
	sum := sha256.Sum256([]byte(live.prompt))
	return scrubbedRequest{
		id:           live.id,
		model:        live.model,
		tokenizer:    live.tokenizer,
		seed:         live.seed,
		maxTokens:    live.maxTokens,
		promptDigest: "sha256:" + hex.EncodeToString(sum[:]),
		promptLen:    len(live.prompt),
	}
}

// scrubbedCanonicalBytes serializes the scrubbed request in a stable form whose
// bytes change only when a scrubbed feature changes. It deliberately folds the
// DIGEST, never the raw prompt, so the decode stream — and therefore every
// derived artifact — is free of the original content.
func scrubbedCanonicalBytes(r scrubbedRequest) []byte {
	var b []byte
	var v [8]byte
	put := func(s string) { b = append(b, byte(len(s)>>8), byte(len(s))); b = append(b, s...) }
	put(r.model)
	put(r.tokenizer)
	put(r.promptDigest)
	binary.BigEndian.PutUint64(v[:], r.seed)
	b = append(b, v[:]...)
	binary.BigEndian.PutUint64(v[:], uint64(r.maxTokens))
	b = append(b, v[:]...)
	binary.BigEndian.PutUint64(v[:], uint64(r.promptLen))
	b = append(b, v[:]...)
	return b
}

// shadowStream is the history-dependent decode state: a splitmix64 stream folded
// with a carried accumulator so token i depends on the whole request prefix seen
// so far, not just (seed, i).
type shadowStream struct{ rng, acc uint64 }

func shadowNewStream(seed uint64) shadowStream {
	return shadowStream{rng: seed*0x9e3779b97f4a7c15 + 0x243f6a8885a308d3}
}

func (s *shadowStream) mix(b []byte) {
	for _, x := range b {
		s.acc = s.acc*0x100000001b3 + uint64(x) + 1
	}
}

func (s *shadowStream) draw() uint64 {
	s.rng += 0x9e3779b97f4a7c15
	z := s.rng ^ s.acc
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	s.acc = s.acc*0x100000001b3 + z
	return z
}

// shadowReferenceDecode is the deterministic generation shared by the live
// (reference) engine and any faithful candidate: fold the scrubbed request into
// the stream, then emit one token per step, folding the step index in so tokens
// vary along the sequence. temperature=0 semantics — same scrubbed request in,
// same tokens out.
func shadowReferenceDecode(r scrubbedRequest) []string {
	st := shadowNewStream(r.seed)
	st.mix(scrubbedCanonicalBytes(r))
	toks := make([]string, 0, r.maxTokens)
	var idx [8]byte
	for i := 0; i < r.maxTokens; i++ {
		binary.BigEndian.PutUint64(idx[:], uint64(i))
		st.mix(idx[:])
		toks = append(toks, shadowVocab[st.draw()%uint64(len(shadowVocab))])
	}
	return toks
}

// shadowEngine is a candidate (or the reference) engine: a name, the
// engine/backend it models, and a decode transform from a scrubbed request to
// output tokens. The reference and any faithful candidate share the reference
// decode; a defective candidate wraps it with a representative regression.
type shadowEngine struct {
	name    string
	backend string
	decode  func(r scrubbedRequest) []string
}

// shadowReferenceEngine is the live/production engine: the sole producer of the
// bytes a user sees, and the baseline every candidate is judged against.
func shadowReferenceEngine() shadowEngine {
	return shadowEngine{name: "reference", backend: "production-serve", decode: shadowReferenceDecode}
}

// shadowFaithfulCandidate models a lossless engine change — a kernel/scheduler
// refactor that is output-identical to the reference. It is the "after the fix"
// engine: token-identical to the reference and invisible on the user path.
func shadowFaithfulCandidate() shadowEngine {
	return shadowEngine{name: "candidate-faithful", backend: "native-refactor", decode: shadowReferenceDecode}
}

// --- planted representative defects (candidate-engine regressions) ----------

// shadowSamplerDriftDefect models a candidate that silently changed a sampler
// default so it no longer honors the request seed. It perturbs the fold at the
// very start, biting at (or near) token 0.
func shadowSamplerDriftDefect(r scrubbedRequest) []string {
	d := r
	d.seed = r.seed ^ 0xa5a5a5a5a5a5a5a5
	return shadowReferenceDecode(d)
}

// shadowEarlyStopDefect models a candidate whose stop handling regressed and
// truncates the response — a dropped-token/length regression near the tail.
func shadowEarlyStopDefect(r scrubbedRequest) []string {
	t := shadowReferenceDecode(r)
	if len(t) >= 2 {
		return t[:len(t)-2]
	}
	return t
}

// shadowCacheSwapDefect models a KV-cache off-by-one that substitutes one
// mid-stream token — the output stays the right length but diverges in the
// middle, the subtlest regression to localize.
func shadowCacheSwapDefect(r scrubbedRequest) []string {
	t := shadowReferenceDecode(r)
	if len(t) == 0 {
		return t
	}
	k := len(t) / 2
	for _, w := range shadowVocab {
		if w != t[k] {
			t[k] = w
			break
		}
	}
	return t
}

// shadowDefectiveCandidates are the in-scope planted regressions. Each is a
// genuinely different failure shape (start / tail / middle) so the oracle's
// first-divergence localization is exercised across the sequence.
func shadowDefectiveCandidates() []shadowEngine {
	return []shadowEngine{
		{name: "candidate-sampler-drift", backend: "native-sampler", decode: shadowSamplerDriftDefect},
		{name: "candidate-early-stop", backend: "native-stop", decode: shadowEarlyStopDefect},
		{name: "candidate-cache-swap", backend: "native-cache", decode: shadowCacheSwapDefect},
	}
}

// --- provenance, replay artifact, and the differential oracle ---------------

// shadowProvenance records everything the acceptance criteria require per case:
// model, tokenizer, engine/backend, seed/oracle, code revision, and
// tolerance/baseline. It is built from the SCRUBBED request, so it structurally
// cannot carry the raw prompt or the end-user id.
type shadowProvenance struct {
	CaseID       string
	Model        string
	Tokenizer    string
	Backend      string
	Seed         uint64
	Revision     string
	Baseline     string
	Tolerance    string
	Tier         string
	PromptDigest string
	PromptLen    int
}

func shadowBaseline(refTokens []string) [32]byte {
	h := sha256.New()
	for _, t := range refTokens {
		_, _ = h.Write([]byte(t))
		_, _ = h.Write([]byte{0})
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

func shadowProvenanceOf(caseID string, r scrubbedRequest, backend string, baseline [32]byte) shadowProvenance {
	return shadowProvenance{
		CaseID: caseID, Model: r.model, Tokenizer: r.tokenizer, Backend: backend,
		Seed: r.seed, Revision: shadowRevision, Baseline: fmt.Sprintf("%x", baseline[:6]),
		Tolerance: "exact (temperature=0, deterministic oracle)", Tier: shadowTrafficTier,
		PromptDigest: r.promptDigest, PromptLen: r.promptLen,
	}
}

// complete reports whether every required provenance field is populated — an
// unprovenanced case is inconclusive and must never be reported as pass.
func (p shadowProvenance) complete() bool {
	return p.Model != "" && p.Tokenizer != "" && p.Backend != "" && p.Revision != "" &&
		p.Baseline != "" && p.Tolerance != "" && p.Tier != "" && p.Seed != 0 &&
		p.PromptDigest != ""
}

// shadowDivergence is the first actionable divergence: the token index and the
// reference vs candidate tokens there.
type shadowDivergence struct {
	Index          int
	ReferenceToken string
	CandidateToken string
}

// shadowReplayArtifact is the scrubbed, independently-replayable failure bundle:
// full provenance plus the first divergence. It carries the prompt DIGEST and
// length, never the raw prompt or the user id — a failure bundle that is safe to
// store and replay in a clean environment.
type shadowReplayArtifact struct {
	Provenance shadowProvenance
	FailPath   string
	Reason     string
	Divergence *shadowDivergence
}

func (a shadowReplayArtifact) String() string {
	idx, ref, cand := -1, "<none>", "<none>"
	if a.Divergence != nil {
		idx, ref, cand = a.Divergence.Index, a.Divergence.ReferenceToken, a.Divergence.CandidateToken
	}
	p := a.Provenance
	return fmt.Sprintf("replay{case=%s model=%s tok=%s backend=%s seed=%#x rev=%s baseline=%s tol=%q tier=%s digest=%s promptlen=%d fail=%s reason=%s divergence=@%d ref=%q cand=%q}",
		p.CaseID, p.Model, p.Tokenizer, p.Backend, p.Seed, p.Revision, p.Baseline, p.Tolerance, p.Tier,
		p.PromptDigest, p.PromptLen, a.FailPath, a.Reason, idx, ref, cand)
}

type shadowVerdict struct {
	Pass     bool
	Detail   string
	Artifact *shadowReplayArtifact
}

func shadowTokenAt(t []string, i int) string {
	if i >= 0 && i < len(t) {
		return t[i]
	}
	return "<none>"
}

// shadowJudge is the differential oracle: a candidate engine must reproduce the
// reference (live) token sequence exactly. Empty/short evidence is never a pass;
// any divergence is reported as the first index with a scrubbed replay artifact.
func shadowJudge(ref, cand []string, prov shadowProvenance) shadowVerdict {
	mk := func(reason string, d *shadowDivergence) *shadowReplayArtifact {
		return &shadowReplayArtifact{Provenance: prov, FailPath: prov.Backend, Reason: reason, Divergence: d}
	}
	if len(cand) == 0 {
		return shadowVerdict{Pass: false, Detail: "candidate produced no tokens — inconclusive evidence is never pass",
			Artifact: mk("no-evidence", &shadowDivergence{Index: 0, ReferenceToken: shadowTokenAt(ref, 0), CandidateToken: "<none>"})}
	}
	n := len(ref)
	if len(cand) < n {
		n = len(cand)
	}
	for i := 0; i < n; i++ {
		if ref[i] != cand[i] {
			d := &shadowDivergence{Index: i, ReferenceToken: ref[i], CandidateToken: cand[i]}
			return shadowVerdict{Pass: false,
				Detail:   fmt.Sprintf("candidate diverged at token %d: reference %q, candidate %q — engine regression", i, ref[i], cand[i]),
				Artifact: mk("divergence", d)}
		}
	}
	if len(ref) != len(cand) {
		d := &shadowDivergence{Index: n, ReferenceToken: shadowTokenAt(ref, n), CandidateToken: shadowTokenAt(cand, n)}
		return shadowVerdict{Pass: false,
			Detail:   fmt.Sprintf("token count diverged at %d: reference has %d, candidate has %d — truncated or over-run", n, len(ref), len(cand)),
			Artifact: mk("length-divergence", d)}
	}
	return shadowVerdict{Pass: true, Detail: fmt.Sprintf("candidate reproduced the reference: %d tokens identical", len(ref))}
}

// shadowFirstDiff returns the first index at which two token streams differ, the
// min length if one is a prefix of the other, or -1 if identical. It lets the
// witness assert the oracle's localization without hard-coding an index.
func shadowFirstDiff(a, b []string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// shadowResult is the outcome of one shadow evaluation: the response the USER
// receives (from the reference engine, always), the regression verdict for the
// candidate, and the scrubbed request the candidate actually saw.
type shadowResult struct {
	Live     []string
	Verdict  shadowVerdict
	Scrubbed scrubbedRequest
}

// shadowServeLive is the user-facing path. It is the ONLY producer of the bytes
// a user sees; it depends solely on the reference engine and the live request,
// never on any candidate.
func shadowServeLive(live liveRequest, ref shadowEngine) []string {
	return ref.decode(shadowScrub(live))
}

// shadowEvaluate mirrors a live request into a shadow evaluation of the
// candidate WITHOUT affecting the user. It computes the user response from the
// reference engine, scrubs a privacy-safe copy for the candidate, and judges the
// candidate against the live response. The returned Live tokens are exactly what
// the user receives regardless of the candidate's behavior; the live request is
// never mutated.
func shadowEvaluate(live liveRequest, ref, cand shadowEngine) shadowResult {
	liveTokens := shadowServeLive(live, ref)
	scrubbed := shadowScrub(live)
	baseline := shadowBaseline(liveTokens)
	prov := shadowProvenanceOf("shadow/"+cand.name, scrubbed, cand.backend, baseline)
	candTokens := cand.decode(scrubbed)
	verdict := shadowJudge(liveTokens, candTokens, prov)
	return shadowResult{Live: liveTokens, Verdict: verdict, Scrubbed: scrubbed}
}

// shadowFixtureRequest is the pinned production-like request. Its raw prompt and
// user id embed recognizable secrets so the witness can prove they never appear
// in the scrubbed request or the replay artifact.
func shadowFixtureRequest() liveRequest {
	return liveRequest{
		id:        "req-7f3a",
		userID:    "enduser-42-PII-SECRET",
		prompt:    "SSN 123-45-6789: summarize the patient chart in one line",
		model:     "fak-fixture-7b",
		tokenizer: "fak-bpe-v1",
		seed:      0x5eed1234,
		maxTokens: 6,
	}
}

// shadowSecretsOf returns the raw, must-never-leak substrings of a live request,
// so the witness can scan any shadow artifact for accidental PII.
func shadowSecretsOf(live liveRequest) []string {
	return []string{live.prompt, live.userID}
}

var _ = strings.TrimSpace
