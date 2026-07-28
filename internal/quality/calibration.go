package quality

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ConfidenceCalibration is the confidence/uncertainty calibration oracle for
// executive reports (#4555): every claim a report asserts states a confidence,
// and the case's reference carries the GROUND-TRUTH support flag for each
// claim. Calibration means the stated confidence never exceeds what the
// evidence licenses — a high-confidence claim must not be one that is actually
// unsupported, and a weak-evidence claim must be hedged. This is orthogonal to
// grounding (claim-grounding checks a claim HAS evidence; this oracle checks
// the report's certainty LANGUAGE matches the evidence it has).
//
// Payload contract (both hermetic, JSON-in-Trace per the additive-seam rule):
//
//	eng.Text: [{"claim":"...","confidence":"high|medium|low"}, ...]
//	ref.Text: [{"claim":"...","support":"strong|weak|none"}, ...]
//
// Calibration rule (deterministic, documented): each support level fixes a
// CEILING on the confidence a calibrated report may state for that claim:
//
//	strong -> high   (any confidence is calibrated; hedging harder is safe)
//	weak   -> low    (a weak-evidence claim must be hedged)
//	none   -> low    (an unsupported claim may at most be floated as hedged
//	                  speculation; asserting it at medium/high confidence is
//	                  the overconfident-fabrication class this oracle exists
//	                  to catch)
//
// A claim with no ground-truth record is treated as support "none" — fail
// closed: unverifiable support never licenses confidence. A claim stating an
// unknown confidence token is itself miscalibrated (uncheckable certainty is
// not calibrated certainty).
//
// Score = calibrated claims / total claims; Pass iff Score >= Rubric.MinScore
// (default 1: every claim must be calibrated). On failure Detail names the
// FIRST miscalibrated claim and its failure class — localizing the
// overconfidence, per the spine contract.
//
// Edge behavior (defined and tested): a report asserting no claims has
// nothing miscalibrated and passes at score 1; an unparseable claim or
// support payload fails closed at score 0 (a payload that cannot be checked
// is not a green payload).
type ConfidenceCalibration struct{}

func (ConfidenceCalibration) Name() string { return "confidence-calibration" }
func (ConfidenceCalibration) Kind() string { return "rubric" }

func init() { Register(ConfidenceCalibration{}) }

// Stated confidence levels (engine side) and ground-truth support flags
// (reference side). Matching is case-insensitive.
const (
	calConfHigh   = "high"
	calConfMedium = "medium"
	calConfLow    = "low"

	calSupportStrong = "strong"
	calSupportWeak   = "weak"
	calSupportNone   = "none"
)

func (ConfidenceCalibration) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "confidence-calibration", Kind: "rubric", Pass: true, Score: 1}
	claims, err := calParseClaims(eng.Text)
	if err != nil {
		return rubricFail(v, fmt.Sprintf("engine claim payload unparseable: %v", err))
	}
	if len(claims) == 0 {
		v.Detail = "report asserts no claims; nothing to calibrate"
		return v
	}
	support, err := calParseSupport(ref.Text)
	if err != nil {
		return rubricFail(v, fmt.Sprintf("reference support payload unparseable: %v", err))
	}
	calibrated := 0
	firstBad := ""
	for i, cl := range claims {
		bad := calJudgeClaim(i, cl, support)
		if bad == "" {
			calibrated++
		} else if firstBad == "" {
			firstBad = bad
		}
	}
	min, short := rubricScore(&v, c, calibrated, len(claims))
	if short {
		v.Detail = fmt.Sprintf("calibration %.2f < %.2f (%d/%d claims calibrated); first miscalibrated: %s",
			v.Score, min, calibrated, len(claims), firstBad)
		return v
	}
	if firstBad != "" {
		v.Detail = fmt.Sprintf("calibration %.2f >= %.2f (%d/%d calibrated; tolerated miscalibration: %s)",
			v.Score, min, calibrated, len(claims), firstBad)
		return v
	}
	v.Detail = fmt.Sprintf("all %d claim(s) state calibrated confidence", len(claims))
	return v
}

// calJudgeClaim judges one stated claim against the ground-truth support map.
// It returns "" for a calibrated claim, else the localized miscalibration
// message naming the claim and its failure class.
func calJudgeClaim(i int, cl calClaim, support map[string]string) string {
	rank, ok := calConfidenceRank(cl.Confidence)
	if !ok {
		return fmt.Sprintf("claim %d %q states unknown confidence %q", i, cl.Claim, cl.Confidence)
	}
	sup, has := support[calKey(cl.Claim)]
	if !has {
		sup = calSupportNone // fail closed: no ground-truth record never licenses confidence
	}
	if rank <= calSupportCeiling(sup) {
		return ""
	}
	switch sup {
	case calSupportNone:
		return fmt.Sprintf("claim %d %q: stated %q confidence but the claim is unsupported", i, cl.Claim, cl.Confidence)
	case calSupportWeak:
		return fmt.Sprintf("claim %d %q: weak-evidence claim is unhedged (stated %q, must be %q)",
			i, cl.Claim, cl.Confidence, calConfLow)
	default:
		return fmt.Sprintf("claim %d %q: stated %q confidence exceeds what %q support licenses",
			i, cl.Claim, cl.Confidence, sup)
	}
}

// calClaim is one stated claim in the engine payload: the claim text and the
// confidence the report attached to it.
type calClaim struct {
	Claim      string `json:"claim"`
	Confidence string `json:"confidence"`
}

// calSupportRecord is one ground-truth row in the reference payload: the claim
// text and the evidential support it actually has.
type calSupportRecord struct {
	Claim   string `json:"claim"`
	Support string `json:"support"`
}

// calParseClaims parses the engine payload. Blank text is a report asserting
// no claims (nil, no error); entries with empty claim text assert nothing
// checkable and are dropped.
func calParseClaims(text string) ([]calClaim, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}
	var raw []calClaim
	if err := json.Unmarshal([]byte(text), &raw); err != nil {
		return nil, err
	}
	out := make([]calClaim, 0, len(raw))
	for _, cl := range raw {
		if strings.TrimSpace(cl.Claim) == "" {
			continue
		}
		out = append(out, cl)
	}
	return out, nil
}

// calParseSupport parses the reference payload into a normalized-claim ->
// support map, validating every support token so a malformed ground truth is
// refused loudly rather than silently defaulting. A later duplicate record
// overrides an earlier one.
func calParseSupport(text string) (map[string]string, error) {
	out := map[string]string{}
	if strings.TrimSpace(text) == "" {
		return out, nil
	}
	var raw []calSupportRecord
	if err := json.Unmarshal([]byte(text), &raw); err != nil {
		return nil, err
	}
	for _, r := range raw {
		if strings.TrimSpace(r.Claim) == "" {
			continue
		}
		sup := strings.ToLower(strings.TrimSpace(r.Support))
		switch sup {
		case calSupportStrong, calSupportWeak, calSupportNone:
			out[calKey(r.Claim)] = sup
		default:
			return nil, fmt.Errorf("claim %q carries unknown support flag %q", r.Claim, r.Support)
		}
	}
	return out, nil
}

// calConfidenceRank orders the stated confidence vocabulary: low(0) <
// medium(1) < high(2). Unknown tokens report ok=false.
func calConfidenceRank(s string) (rank int, ok bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case calConfLow:
		return 0, true
	case calConfMedium:
		return 1, true
	case calConfHigh:
		return 2, true
	default:
		return 0, false
	}
}

// calSupportCeiling is the highest calibrated confidence rank each support
// level licenses: strong -> high, weak/none -> low (hedged only).
func calSupportCeiling(sup string) int {
	if sup == calSupportStrong {
		return 2
	}
	return 0
}

// calKey normalizes claim text for matching stated claims to ground-truth
// records: lowercased, whitespace-trimmed.
func calKey(claim string) string {
	return strings.ToLower(strings.TrimSpace(claim))
}
