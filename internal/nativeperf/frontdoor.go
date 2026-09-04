package nativeperf

import (
	"fmt"
	"strings"
	"time"
)

const (
	FrontDoorBegin = "<!-- qwen38-frontdoor:begin -->"
	FrontDoorEnd   = "<!-- qwen38-frontdoor:end -->"

	FrontDoorSurfaceREADME = "readme"
	FrontDoorSurfaceIndex  = "index"
	FrontDoorSurfaceLatest = "latest"
)

// FrontDoorResult is one publication-classified result. ReviewBy is inclusive:
// the result is reaped from active presentation on the following day.
type FrontDoorResult struct {
	Class       string  `json:"class"`
	EnvelopeID  string  `json:"envelope_id"`
	NativeTPS   float64 `json:"native_tokens_per_second,omitempty"`
	BaselineTPS float64 `json:"baseline_tokens_per_second,omitempty"`
	Ratio       float64 `json:"ratio,omitempty"`
	Quality     string  `json:"quality"`
	ObservedOn  string  `json:"observed_on"`
	ReviewBy    string  `json:"review_by"`
	Provenance  string  `json:"provenance"`
	Caveat      string  `json:"caveat"`
}

type ReapedFrontDoorResult struct {
	Class      string `json:"class"`
	EnvelopeID string `json:"envelope_id"`
	ReviewBy   string `json:"review_by"`
	Reason     string `json:"reason"`
}

// FrontDoorSnapshot is the deterministic publication view derived from the
// typed graph plus committed accepted/diagnostic metadata. Reaping affects only
// these active pointers; immutable witness files remain untouched.
type FrontDoorSnapshot struct {
	Schema      string                  `json:"schema"`
	AsOf        string                  `json:"as_of"`
	Accepted    *FrontDoorResult        `json:"accepted,omitempty"`
	Approximate *FrontDoorResult        `json:"approximate,omitempty"`
	Diagnostic  *FrontDoorResult        `json:"diagnostic,omitempty"`
	Reaped      []ReapedFrontDoorResult `json:"reaped,omitempty"`
}

const FrontDoorSchema = "fak.qwen38-frontdoor/1"

// BuildFrontDoorSnapshot binds the public Qwen readout to the existing typed
// native-performance graph. The 47% observation is deliberately approximate:
// its native arm used P31/T64 while the comparison target was P32/T64, and no
// joint quality-complete receipt captured both engines.
func BuildFrontDoorSnapshot(graph Graph, asOf time.Time) (FrontDoorSnapshot, error) {
	if err := Validate(graph); err != nil {
		return FrontDoorSnapshot{}, err
	}
	asOf = asOf.UTC()
	s := FrontDoorSnapshot{Schema: FrontDoorSchema, AsOf: asOf.Format(time.DateOnly)}

	accepted := newFrontDoorEntry("accepted", "q38-q4km-native-metal-m3pro-fullrun", 2.9, "PASS (three functional probes; accepted range 2.3-2.9 tok/s)", "2026-08-20", "2026-08-27", "docs/_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json", "accepted native Metal full-run range; not a matched llama.cpp parity receipt")
	addFrontDoorResult(&s, &accepted, asOf)

	var witnessed *Throughput
	for i := range graph.Rungs {
		if graph.Rungs[i].ID == "resident-q4k-baseline" {
			witnessed = graph.Rungs[i].Witnessed
			break
		}
	}
	if witnessed == nil || witnessed.Classification != ClassWitnessed || graph.Comparison.Classification != ClassComparison {
		return FrontDoorSnapshot{}, fmt.Errorf("native performance graph lacks the witnessed Metal baseline/comparison pair; restore the resident-q4k-baseline witnessed row and comparison classification")
	}
	if witnessed.Provenance != graph.Comparison.Provenance || witnessed.ObservedOn != graph.Comparison.ObservedOn {
		return FrontDoorSnapshot{}, fmt.Errorf("Metal baseline/comparison provenance or observation date differs; use one shared provenance and observed_on value for both rows")
	}
	approx := FrontDoorResult{
		Class: "approximate", EnvelopeID: graph.Envelope.ID,
		NativeTPS: witnessed.TokensPerSecond, BaselineTPS: graph.Comparison.TokensPerSecond,
		Ratio:   witnessed.TokensPerSecond / graph.Comparison.TokensPerSecond,
		Quality: "not jointly quality-complete", ObservedOn: witnessed.ObservedOn, ReviewBy: "2026-08-27",
		Provenance: witnessed.Provenance,
		Caveat:     "P31/T64 native versus P32/T64 llama.cpp; no joint quality-complete receipt",
	}
	addFrontDoorResult(&s, &approx, asOf)

	diagnostic := newFrontDoorEntry("diagnostic", "q38-q4km-cuda-a100-cache-restore", 0.2, "0/5 exact", "2026-08-25", "2026-09-01", "docs/_witnesses/issue-8819-qwen38-cache-attribution/summary.json", "failed-quality A100 cache-restore arm; diagnostic only and never a parity headline")
	addFrontDoorResult(&s, &diagnostic, asOf)
	return s, nil
}

func addFrontDoorResult(s *FrontDoorSnapshot, result *FrontDoorResult, asOf time.Time) {
	review, err := time.Parse(time.DateOnly, result.ReviewBy)
	if err != nil || asOf.After(review.Add(24*time.Hour-time.Nanosecond)) {
		s.Reaped = append(s.Reaped, ReapedFrontDoorResult{
			Class: result.Class, EnvelopeID: result.EnvelopeID, ReviewBy: result.ReviewBy,
			Reason: "review date passed without renewed comparable evidence",
		})
		return
	}
	switch result.Class {
	case "accepted":
		s.Accepted = result
	case "approximate":
		s.Approximate = result
	case "diagnostic":
		s.Diagnostic = result
	}
}

func newFrontDoorEntry(class, envID string, tps float64, quality, observed, review, prov, caveat string) FrontDoorResult {
	return FrontDoorResult{
		Class:      class,
		EnvelopeID: envID,
		NativeTPS:  tps,
		Quality:    quality,
		ObservedOn: observed,
		ReviewBy:   review,
		Provenance: prov,
		Caveat:     caveat,
	}
}

// FrontDoorBlock renders the generated block for one public surface.
func FrontDoorBlock(snapshot FrontDoorSnapshot, surface string) (string, error) {
	var body string
	switch surface {
	case FrontDoorSurfaceREADME:
		body = renderREADMEFrontDoor(snapshot)
	case FrontDoorSurfaceIndex:
		body = renderIndexFrontDoor(snapshot)
	case FrontDoorSurfaceLatest:
		body = renderLatestFrontDoor(snapshot)
	default:
		return "", fmt.Errorf("unknown Qwen front-door surface %q; choose %q, %q, or %q", surface, FrontDoorSurfaceREADME, FrontDoorSurfaceIndex, FrontDoorSurfaceLatest)
	}
	return FrontDoorBegin + "\n" + body + "\n" + FrontDoorEnd, nil
}

func renderREADMEFrontDoor(s FrontDoorSnapshot) string {
	accepted := "No accepted Metal result remains inside its review window."
	if s.Accepted != nil {
		accepted = "Accepted fak-native Metal Q4_K_M: **2.3-2.9 decode tok/s**, with functional `PASS` in the frozen M3 Pro full-run envelope. ([accepted receipt](docs/_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json))"
	}
	comparison := "The closest comparison has passed review and is omitted pending remeasurement."
	if s.Approximate != nil {
		comparison = fmt.Sprintf("Closest near-matched observation: **%.1f vs %.6f tok/s (~%.0f%%)** on the same M3 Pro and artifact. This is approximate, not accepted parity: native used P31/T64 versus P32/T64 and no joint quality-complete receipt exists. ([#8697](%s))", s.Approximate.NativeTPS, s.Approximate.BaselineTPS, s.Approximate.Ratio*100, s.Approximate.Provenance)
	}
	diagnostic := "The cache diagnostic has passed review and is omitted pending remeasurement."
	if s.Diagnostic != nil {
		diagnostic = "Separate A100 cache-restore diagnostic: **~0.2 tok/s with 0/5 exact**. Failed quality makes it diagnostic only, never the parity headline. ([cache attribution](docs/_witnesses/issue-8819-qwen38-cache-attribution/README.md))"
	}
	return "| [Qwen3.8-27B](docs/benchmarks/QWEN-PERFORMANCE-INDEX.md) | " + accepted + " " + comparison + " | " + diagnostic + " |"
}

func renderIndexFrontDoor(s FrontDoorSnapshot) string {
	lines := []string{"## Generated front-door readout", "", "This block is derived by `fak native-performance --frontdoor-md`; classifications cannot be spliced across envelopes."}
	if s.Accepted != nil {
		lines = append(lines, "", "- **ACCEPTED:** fak-native Metal Q4_K_M delivered **2.3-2.9 decode tok/s** with functional `PASS` in the frozen M3 Pro full-run envelope. [Receipt](../_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json).")
	}
	if s.Approximate != nil {
		lines = append(lines, fmt.Sprintf("- **APPROXIMATE:** the closest near-matched observation is **%.1f vs %.6f tok/s (~%.0f%%)**. It is not accepted parity: P31/T64 native versus P32/T64 llama.cpp, with no joint quality-complete receipt. [Issue #8697](%s).", s.Approximate.NativeTPS, s.Approximate.BaselineTPS, s.Approximate.Ratio*100, s.Approximate.Provenance))
	}
	if s.Diagnostic != nil {
		lines = append(lines, "- **DIAGNOSTIC:** the separate A100 cache-restore arm measured **~0.2 tok/s with 0/5 exact**. Failed quality keeps it out of accepted and approximate comparison headlines. [Attribution](../_witnesses/issue-8819-qwen38-cache-attribution/README.md).")
	}
	if len(s.Reaped) > 0 {
		lines = append(lines, "", fmt.Sprintf("_%d reviewed row(s) are reaped from active presentation; immutable witnesses remain._", len(s.Reaped)))
	}
	return strings.Join(lines, "\n")
}

func renderLatestFrontDoor(s FrontDoorSnapshot) string {
	lines := []string{"## Generated current readout", ""}
	if s.Accepted != nil {
		lines = append(lines, "- **Accepted:** native fak Metal Q4_K_M, **2.3-2.9 decode tok/s**, functional `PASS` in the frozen M3 Pro full-run envelope.")
	}
	if s.Approximate != nil {
		lines = append(lines, fmt.Sprintf("- **Closest near-matched observation:** **%.1f vs %.6f tok/s (~%.0f%%)**; approximate only because native was P31/T64 versus P32/T64 and no joint quality-complete receipt exists.", s.Approximate.NativeTPS, s.Approximate.BaselineTPS, s.Approximate.Ratio*100))
	}
	if s.Diagnostic != nil {
		lines = append(lines, "- **Separate diagnostic:** A100 cache restore was **~0.2 tok/s with 0/5 exact**; failed quality excludes it from parity presentation.")
	}
	if len(s.Reaped) > 0 {
		lines = append(lines, fmt.Sprintf("- **Reaped:** %d row(s) passed review without renewal and are omitted here; their witnesses remain.", len(s.Reaped)))
	}
	return strings.Join(lines, "\n")
}

func ExtractFrontDoorBlock(doc string) (string, bool) {
	// Generated surfaces own exactly one block. Reject ambiguity instead of
	// updating one marker pair while leaving a second active-looking block.
	if strings.Count(doc, FrontDoorBegin) != 1 || strings.Count(doc, FrontDoorEnd) != 1 {
		return "", false
	}
	start := strings.Index(doc, FrontDoorBegin)
	end := strings.Index(doc, FrontDoorEnd)
	if start < 0 || end < start {
		return "", false
	}
	end += len(FrontDoorEnd)
	return doc[start:end], true
}

func SpliceFrontDoorBlock(doc, block string) (string, error) {
	current, ok := ExtractFrontDoorBlock(doc)
	if !ok {
		return "", fmt.Errorf("Qwen front-door markers not found or ambiguous; restore exactly one Qwen front-door marker pair before running fak native-performance --write-doc")
	}
	return strings.Replace(doc, current, block, 1), nil
}
