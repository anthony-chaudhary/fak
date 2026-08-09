package quality

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
)

// This file is the first-failure LOCALIZER behind `fak quality explain` (#4520).
// RunCase already answers "did this case fail, and at which token" — this answers
// the question an operator actually has next: WHICH LAYER of the serving path is
// the first actionable suspect, so a failing report is handed to the tokenizer
// owner or the sampler owner rather than to whoever noticed it.
//
// Two properties make the answer trustworthy rather than merely plausible:
//
//   - Every stage is reached only through a POSITIVE evidence signature read off
//     the failure bundle itself — never off the failing oracle's NAME. A bundle
//     assembled by hand therefore classifies exactly as one produced by a live
//     run, which is what makes the synthetic fixtures in the test file a real
//     check and not a tautology, and it means an oracle shipped later by the
//     #4509 cohort localizes without editing this file.
//   - A bundle whose evidence fits no signature ABSTAINS. Abstention is the
//     honest answer to thin evidence, and it never softens the verdict: a run
//     that cannot be localized is still a failing run. Classify reports WHERE,
//     never WHETHER.
//
// The one assumption worth stating: a short engine stream whose every shared
// token agrees is read as a stop-criteria divergence, and a stream cut off
// mid-flight presents identically. The two are separated by the transport
// signature below (a stream that arrived structurally empty), so a partial cut
// that carries a well-formed prefix is attributed to stops.

// Stages is the closed localization vocabulary, in the order the classifier
// probes it. Classify emits a member of this set or StageAbstain, never anything
// else, so a consumer may route on the value without a default arm.
var Stages = []string{
	"rubric",
	"transport",
	"tokenization",
	"normalization",
	"logits",
	"stops",
	"sampling",
	"cache",
}

// StageAbstain is not a stage: it is the refusal to name one when the bundle's
// evidence supports no signature. It is deliberately spelled in the same field
// so a consumer cannot read a missing stage as an absent problem.
const StageAbstain = "unclassified"

// Classification is one first-failure localization: the stage the evidence points
// at and the reason naming the evidence it was read from. It is a pure function
// of the bundle, so it replays identically and is safe to embed in the emitted
// artifact.
type Classification struct {
	Stage  string `json:"stage"`
	Reason string `json:"reason"`
}

// Abstained reports whether the classifier declined to name a stage.
func (c Classification) Abstained() bool { return c.Stage == StageAbstain || c.Stage == "" }

// Classify localizes the first failure of a result. A passing result and a
// failing result with no bundle both abstain — the first because there is nothing
// to localize, the second because there is no evidence to localize from.
func Classify(r Result) Classification {
	if r.Pass {
		return abstain("the run passed: there is no first divergence to localize")
	}
	if r.FailureBundle == nil {
		return abstain("the run failed but carries no failure bundle: there is no evidence to localize from")
	}
	return classifyFailure(*r.FailureBundle)
}

func abstain(reason string) Classification {
	return Classification{Stage: StageAbstain, Reason: reason}
}

// classifyFailure walks the signature ladder over one bundle. Order is evidence
// depth, not convenience: a signature that reads the numbers is consulted before
// one that reads the text, and the text before the sampling configuration, so the
// deepest thing the bundle can actually prove wins.
func classifyFailure(fb FailureBundle) Classification {
	ref, eng := fb.Reference, fb.Engine

	// A scorer failed, not a comparator: the stream was never contradicted, its
	// scored content was. This is the one signature read from the oracle, and it
	// is read from its KIND — a contract every oracle already declares — rather
	// than from its name.
	if fb.FailingKind == "rubric" {
		return Classification{Stage: "rubric", Reason: fmt.Sprintf(
			"the first failing oracle %q scores content instead of comparing streams: the defect is in what the output claims, not in how it was decoded",
			fb.FailingOracle)}
	}

	// Nothing usable arrived. Anything deeper would be localized from absence.
	if len(ref.Tokens) > 0 && len(eng.Tokens) == 0 {
		return Classification{Stage: "transport", Reason: fmt.Sprintf(
			"the engine trace carries no tokens while the reference carries %d: the payload did not survive the hop, so no deeper layer is observable",
			len(ref.Tokens))}
	}
	if len(eng.Tokens) > 0 && eng.Text == "" && ref.Text != "" {
		return Classification{Stage: "transport", Reason: fmt.Sprintf(
			"the engine emitted %d tokens but arrived with empty assembled text where the reference carries %d bytes: the body was lost in flight",
			len(eng.Tokens), len(ref.Text))}
	}

	tokensSame := equalTokens(ref.Tokens, eng.Tokens)
	textSame := ref.Text == eng.Text

	// The two streams disagree in exactly one of their two surfaces. Either way
	// the boundary between tokens and text is where they parted, which is the
	// tokenizer's own seam — unless the texts are the same string under case and
	// spacing, which is the narrower normalization claim.
	if tokensSame && !textSame {
		if normalizedSame(ref.Text, eng.Text) {
			return Classification{Stage: "normalization", Reason: fmt.Sprintf(
				"identical tokens assembled to text that differs only in case or whitespace (reference %q, engine %q): the divergence is in output normalization, not in the decode",
				trimForReason(ref.Text), trimForReason(eng.Text))}
		}
		return Classification{Stage: "tokenization", Reason: fmt.Sprintf(
			"the token streams are identical yet the assembled text differs (reference %q, engine %q): the divergence is in detokenization, downstream of every decode step",
			trimForReason(ref.Text), trimForReason(eng.Text))}
	}
	if !tokensSame && textSame {
		return Classification{Stage: "tokenization", Reason: fmt.Sprintf(
			"both paths assembled byte-identical text (%d bytes) from different token streams (%d vs %d tokens): the same string was segmented two ways",
			len(ref.Text), len(ref.Tokens), len(eng.Tokens))}
	}

	d := fb.FirstDivergence
	if d == nil {
		return abstain(fmt.Sprintf(
			"oracle %q (%s) failed without pinning a first divergence: a verdict with no step to point at cannot be attributed to a layer — re-run the case under a differential oracle to localize it",
			fb.FailingOracle, fb.FailingKind))
	}

	// Both paths captured the numbers at the divergent step and the numbers
	// themselves disagree. A whole row displaced by one constant is the missing
	// log-softmax — a normalization defect that leaves the argmax, and therefore
	// the tokens, untouched; anything else is the distribution itself.
	if a, b, ok := logitRowsAt(ref, eng, d.Index); ok && rowsDiffer(a, b) {
		if delta, flat := rowOffset(a, b); flat {
			return Classification{Stage: "normalization", Reason: fmt.Sprintf(
				"every logit at step %d is displaced by the same constant %.6f: the row was reported without its log-softmax, which is a normalization defect rather than a different distribution",
				d.Index, delta)}
		}
		return Classification{Stage: "logits", Reason: fmt.Sprintf(
			"the captured logit rows at step %d disagree by more than a constant offset: the distribution the token was drawn from had already diverged before any selection ran",
			d.Index)}
	}

	// Every position both streams share agrees, and one simply kept going or quit
	// early. Nothing about the token selection differed — only the decision to end.
	if sharedTokensAgree(ref.Tokens, eng.Tokens) && len(ref.Tokens) != len(eng.Tokens) {
		verb, longer := "stopped early at", "reference"
		if len(eng.Tokens) > len(ref.Tokens) {
			verb, longer = "kept decoding past", "engine"
		}
		return Classification{Stage: "stops", Reason: fmt.Sprintf(
			"all %d shared tokens agree and the %s stream is longer (reference %d, engine %d): the engine %s token %d, so the divergence is in the stop decision and not in the decode",
			minLen(ref.Tokens, eng.Tokens), longer, len(ref.Tokens), len(eng.Tokens), verb, minLen(ref.Tokens, eng.Tokens))}
	}

	// The divergent tokens are the same string once case and spacing are set
	// aside, so the selection agreed and the rendering did not.
	if normalizedSame(d.Reference, d.Engine) && d.Reference != d.Engine {
		return Classification{Stage: "normalization", Reason: fmt.Sprintf(
			"the tokens at step %d differ only in case or whitespace (reference %q, engine %q): the selection agreed and the rendering did not",
			d.Index, d.Reference, d.Engine)}
	}

	// A case that declares a stochastic decode is permitted, by its own
	// configuration, to draw a different token from an identical distribution.
	// That is the sampler's surface, and no deeper evidence contradicted it above.
	if p := fb.Case.Params; p.Temperature > 0 || p.TopK > 0 || (p.TopP > 0 && p.TopP < 1) {
		return Classification{Stage: "sampling", Reason: fmt.Sprintf(
			"the case decodes stochastically (temperature %.3f, top_k %d, top_p %.3f, seed %d) and no captured logit evidence contradicted the distribution: the first divergence at step %d is a draw that differed, not a distribution that did",
			p.Temperature, p.TopK, p.TopP, p.Seed, d.Index)}
	}

	// A greedy case cannot legally differ from its own reference by chance. When
	// the run also declares an enabled reuse flag and the streams part MID-stream
	// rather than at the very first token, reuse of a stored prefix is the first
	// actionable suspect: step 0 has nothing stored yet, which is exactly why the
	// index has to be positive for this arm to fire.
	if k, v, ok := firstFlagMatching(fb.Case, "cache"); ok && d.Index > 0 {
		return Classification{Stage: "cache", Reason: fmt.Sprintf(
			"a greedy case (temperature 0) diverged mid-stream at step %d with reuse enabled by engine flag %s=%q: the shared prefix was served from storage, so the stored-prefix path is the first actionable suspect",
			d.Index, k, v)}
	}

	return abstain(fmt.Sprintf(
		"the first divergence at step %d (reference %q, engine %q) is real but unattributed: the bundle carries no logit rows at that step, no length or segmentation difference, no case-or-spacing explanation, and a deterministic case with no reuse flag declared — capture per-step logits or declare the engine flags to localize it",
		d.Index, d.Reference, d.Engine))
}

// equalTokens reports whether two token streams are element-wise identical.
func equalTokens(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// sharedTokensAgree reports whether the two streams agree at every position they
// both have. It is the prefix half of the stop signature: the length difference
// is checked by the caller, because agreement alone is what a pass looks like.
func sharedTokensAgree(a, b []string) bool {
	for i := 0; i < minLen(a, b); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func minLen(a, b []string) int {
	if len(b) < len(a) {
		return len(b)
	}
	return len(a)
}

// normalizedSame reports whether two strings are equal once case is folded and
// every run of whitespace is collapsed to one space. The probe is deliberately
// narrow: this package is stdlib-only, so Unicode composition (NFC vs NFD) is out
// of reach and a composition-only divergence abstains rather than being labelled
// from a check that never ran.
func normalizedSame(a, b string) bool {
	return a != b && strings.EqualFold(collapseSpaces(a), collapseSpaces(b))
}

func collapseSpaces(s string) string {
	return strings.Join(strings.FieldsFunc(s, unicode.IsSpace), " ")
}

// trimForReason bounds a quoted trace excerpt so one long report cannot turn a
// one-line localization into a wall of text.
func trimForReason(s string) string {
	const max = 72
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// logitRowsAt returns both paths' captured logit rows for step i when both
// captured one. A step only one path recorded is not evidence of disagreement.
func logitRowsAt(ref, eng Trace, i int) (a, b []float64, ok bool) {
	if i < 0 || i >= len(ref.Logits) || i >= len(eng.Logits) {
		return nil, nil, false
	}
	return ref.Logits[i], eng.Logits[i], true
}

// rowsDiffer reports whether two logit rows disagree at all, by length or by any
// element beyond floating-point identity.
func rowsDiffer(a, b []float64) bool {
	if len(a) != len(b) {
		return true
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-12 {
			return true
		}
	}
	return false
}

// rowOffset reports the constant by which row b is displaced from row a, and
// whether the displacement is in fact constant across every element. A single
// element proves nothing — any two numbers differ by some constant — so a row
// shorter than two entries is never flat.
func rowOffset(a, b []float64) (delta float64, flat bool) {
	if len(a) != len(b) || len(a) < 2 {
		return 0, false
	}
	delta = b[0] - a[0]
	if math.Abs(delta) <= 1e-9 {
		return 0, false
	}
	for i := range a {
		if math.Abs((b[i]-a[i])-delta) > 1e-9 {
			return 0, false
		}
	}
	return delta, true
}

// firstFlagMatching returns the case's lowest-named engine flag whose key
// contains needle and whose value does not read as switched off, so a declared
// but disabled surface never implicates itself. Keys are sorted before the scan
// because Go map order is random and a reason string that names a different flag
// on every run is not replayable evidence.
func firstFlagMatching(c QualityCase, needle string) (key, value string, ok bool) {
	flags := c.Metadata.Engine.Flags
	keys := make([]string, 0, len(flags))
	for k := range flags {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !strings.Contains(strings.ToLower(k), needle) {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(flags[k])) {
		case "", "0", "off", "no", "false", "none", "disabled":
			continue
		}
		return k, flags[k], true
	}
	return "", "", false
}
