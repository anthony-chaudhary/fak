//go:build darwin && arm64 && cgo

// q4k_m5_crossover_receipt.go — the machine-admissible receipt for the P-band-keyed Q4_K
// wide-tile crossover (fak#13133, parent fak#13124/#13088/#9937).
//
// fak#9937 shipped the wide-tile cooperative-SMEM selector, and fak#13124 pinned one
// device/version row from a physical candidate-vs-scalar measurement. That row was keyed on
// {Family, OSVersion} alone and pinned its MinRatio at the 1.10x gate FLOOR, so the measured
// prompt-length-dependent lever (1.49x @ P=64, 1.41x @ P=128 in the fak#13124 capture) could not
// bound the routing decision: the row admitted the candidate at EVERY P>=64, and could not express
// a P where the candidate regresses.
//
// fak#13133 fixes that on two axes. First, the row now carries a measured P band [MinP,MaxP], and
// the admission predicate requires the band to cover THIS P — so a prompt length with no measured
// row executes scalar (fail-closed). Second, the row's MinRatio becomes the MEASURED floor over
// that band (still gated at >= 1.10), so the receipt's magnitude is what the table records rather
// than a constant. The band — not the ratio — is what changes routing; MinRatio is the recorded
// margin the band must keep clearing, and the witness fails if a regression drops it below the gate.
//
// This receipt closes that gap in the same discipline the fak#13089 paired harness enforces:
// a fail-closed, paired candidate-vs-scalar record with provenance, an evidence kind that
// distinguishes a physical run from a software one, and a validator that refuses to admit an
// unmeasured, unbalanced, non-finite, or below-gate band. Here the arms are KERNELS (mode 0 vs
// mode 2) rather than whole-model engines, so the schema is metalgemm-local; the paired-receipt
// DISCIPLINE is the #13089 contract.

package metalgemm

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Q4KM5CrossoverReceiptSchema is the schema string every banded crossover receipt carries. A
// reader that does not know this exact string must refuse the artifact, so the schema is a
// closed contract rather than a hint.
const Q4KM5CrossoverReceiptSchema = "fak.metalgemm.q4k_m5_crossover_banded/v1"

// Q4KM5CrossoverReceiptHarness names the paired-receipt harness whose discipline this artifact
// follows (the matched baseline/candidate prefill receipt in internal/macbench, fak#13089).
const Q4KM5CrossoverReceiptHarness = "fak#13089 paired baseline/candidate receipt discipline"

// Q4KReceiptEvidenceHW is the evidence kind for a receipt captured by executing both kernels on
// physical silicon. A Go-only (mock or schema-only) run must record SW_VERIFIED and can never
// justify a pinned band.
const (
	Q4KReceiptEvidenceHW = "HW_WITNESSED"
	Q4KReceiptEvidenceSW = "SW_VERIFIED"
)

// Q4KReceiptArmScalar / Q4KReceiptArmCandidate name the two arms of the pair. The ratio is always
// scalar-over-candidate (a speedup, so >1 means the candidate is faster).
const (
	Q4KReceiptArmScalar    = "scalar"
	Q4KReceiptArmCandidate = "candidate"
)

// Q4KM5CrossoverBand is one measured prompt-length band: the inclusive P interval whose endpoints
// were physically measured, the median candidate/scalar ratio at each measured shape, and the
// conservative floor (the smallest measured ratio) that becomes the row's MinRatio.
type Q4KM5CrossoverBand struct {
	MinP           int                `json:"min_p"`             // inclusive lower bound of the measured band
	MaxP           int                `json:"max_p"`             // inclusive upper bound (0 = unbounded above)
	Measured       map[int]float64    `json:"measured_ratios"`   // prompt shape -> median candidate/scalar on-GPU ratio
	Floor          float64            `json:"measured_floor"`    // smallest measured ratio over the band
	ScalarGPUMS    map[int]float64    `json:"scalar_gpu_ms"`     // per-shape median scalar on-GPU window
	CandidateGPUMS map[int]float64    `json:"candidate_gpu_ms"`  // per-shape median candidate on-GPU window
	Samples        map[int]ArmSamples `json:"samples,omitempty"` // per-shape raw balanced samples (fresh captures)
}

// ArmSamples holds the balanced raw samples for both arms at one prompt shape, so a reader can
// re-derive the medians and the spread instead of trusting the summary. Balanced means equal
// counts; the validator refuses an unbalanced shape.
type ArmSamples struct {
	ScalarMS    []float64 `json:"scalar_gpu_ms"`
	CandidateMS []float64 `json:"candidate_gpu_ms"`
}

// Measurement provenance. A "fresh" receipt was captured by executing both kernels on this host
// and carries the raw balanced samples the medians were derived from, so the validator re-derives
// every recorded ratio. A "recorded" receipt transcribes the medians of a PRIOR on-silicon
// capture named by Provenance (a commit or artifact path) when the GPU cannot be re-taken; it
// carries no raw samples and is admissible only because it cites the capture it transcribes and
// still clears the gate. Only "fresh" may additionally carry samples.
const (
	Q4KMeasurementFresh    = "fresh"
	Q4KMeasurementRecorded = "recorded"
)

// Q4KM5CrossoverReceipt is the non-fabricable artifact behind the pinned q4kM5CrossoverTable
// rows. It binds the host identity, the source commit and emitting test, the measured geometry,
// the metric, the repeat count, the measurement provenance, and one band per measured prompt
// interval.
type Q4KM5CrossoverReceipt struct {
	Schema        string               `json:"schema"`
	ContractIssue string               `json:"contract_issue"`
	PairedHarness string               `json:"paired_harness"`
	GeneratedAt   string               `json:"generated_at"`
	EvidenceKind  string               `json:"evidence_kind"`
	Measurement   string               `json:"measurement"` // fresh | recorded
	Provenance    string               `json:"provenance"`  // required when measurement=recorded: the prior capture cited
	DeviceName    string               `json:"device_name"`
	OSVersion     string               `json:"os_version"`
	SourceCommit  string               `json:"source_commit"`
	SourceTest    string               `json:"source_test"`
	GeometryOut   int                  `json:"geometry_out"`
	GeometryIn    int                  `json:"geometry_in"`
	Metric        string               `json:"metric"`
	Repeats       int                  `json:"repeats"`
	GateMargin    float64              `json:"gate_margin"`
	Bands         []Q4KM5CrossoverBand `json:"bands"`
	Notes         []string             `json:"notes,omitempty"`
}

// q4kReceiptWithinRelative reports whether a and b agree to within tolerance (relative to b).
func q4kReceiptWithinRelative(a, b, tolerance float64) bool {
	if b == 0 {
		return a == 0
	}
	return math.Abs(a-b)/math.Abs(b) <= tolerance
}

// medianFloat returns the median of a sample; it copies so the caller's slice is untouched. It is
// shared by the receipt validator (to re-derive a recorded ratio from its raw samples) and by the
// on-silicon witness, so the recorded median and the validated median are the same function.
func medianFloat(xs []float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[len(s)/2]
}

// ValidateQ4KM5CrossoverReceipt is the fail-closed publication gate for a banded crossover
// receipt. It accepts only a physical, paired, above-gate artifact with a real band and rejects
// anything one-sided, below the fak#9937 gate, non-finite, missing a measured band, or (for a
// fresh capture) missing or unbalanced raw samples. It never inspects the live device — a receipt
// is admissible on its own contents.
func ValidateQ4KM5CrossoverReceipt(r Q4KM5CrossoverReceipt) error {
	var problems []string
	require := func(ok bool, field, detail string) {
		if !ok {
			problems = append(problems, field+": "+detail)
		}
	}

	require(r.Schema == Q4KM5CrossoverReceiptSchema, "schema", "must be "+Q4KM5CrossoverReceiptSchema)
	require(strings.TrimSpace(r.ContractIssue) != "", "contract_issue", "is required")
	require(strings.TrimSpace(r.PairedHarness) != "", "paired_harness", "is required")
	require(strings.TrimSpace(r.DeviceName) != "", "device_name", "is required")
	require(strings.TrimSpace(r.OSVersion) != "", "os_version", "is required")
	require(strings.TrimSpace(r.SourceCommit) != "", "source_commit", "is required")
	require(strings.TrimSpace(r.SourceTest) != "", "source_test", "is required")
	if _, err := time.Parse(time.RFC3339, r.GeneratedAt); err != nil {
		require(false, "generated_at", "must be RFC3339")
	}
	// Only a physical run can pin a band; a SW_VERIFIED artifact is structurally refused.
	require(strings.EqualFold(strings.TrimSpace(r.EvidenceKind), Q4KReceiptEvidenceHW),
		"evidence_kind", "must be "+Q4KReceiptEvidenceHW+" to justify a pinned band")
	// Measurement provenance is a closed set, and a recorded capture must name what it transcribes.
	fresh := r.Measurement == Q4KMeasurementFresh
	require(fresh || r.Measurement == Q4KMeasurementRecorded,
		"measurement", "must be "+Q4KMeasurementFresh+" or "+Q4KMeasurementRecorded)
	if r.Measurement == Q4KMeasurementRecorded {
		require(strings.TrimSpace(r.Provenance) != "", "provenance",
			"a recorded receipt must cite the prior on-silicon capture it transcribes")
	}
	require(r.GeometryOut > 0 && r.GeometryIn > 0, "geometry", "out and in must be positive")
	require(strings.TrimSpace(r.Metric) != "", "metric", "is required")
	require(r.Repeats >= 3, "repeats", "must be at least 3 balanced repeats")
	require(r.GateMargin >= q4kM5CrossoverMargin,
		"gate_margin", fmt.Sprintf("must be >= the fak#9937 gate %.2f", q4kM5CrossoverMargin))
	require(len(r.Bands) > 0, "bands", "must contain at least one measured band")

	for i, band := range r.Bands {
		field := fmt.Sprintf("bands[%d]", i)
		require(band.MinP > 0, field+".min_p", "must be positive")
		require(band.MaxP == 0 || band.MaxP >= band.MinP, field+".max_p", "must be 0 (unbounded) or >= min_p")
		require(len(band.Measured) > 0, field+".measured_ratios", "must record at least one measured shape")
		// A fresh capture must carry the raw balanced samples the medians came from; a recorded
		// capture cites its provenance instead and carries none.
		if fresh {
			require(len(band.Samples) > 0, field+".samples", "a fresh capture must record the raw balanced samples")
		}
		for P, ratio := range band.Measured {
			sampleField := fmt.Sprintf("%s.measured_ratios[%d]", field, P)
			require(P >= band.MinP && (band.MaxP == 0 || P <= band.MaxP),
				sampleField, "measured shape must lie inside the declared band")
			require(q4kFinite(ratio), sampleField, "must be a finite ratio")
			require(ratio >= q4kM5CrossoverMargin,
				sampleField, fmt.Sprintf("measured ratio %.4f is below the fak#9937 gate %.2f", ratio, q4kM5CrossoverMargin))
			if !fresh {
				continue
			}
			samples, ok := band.Samples[P]
			require(ok, sampleField, "must have a matching raw sample entry")
			if !ok {
				continue
			}
			require(len(samples.ScalarMS) >= 3 && len(samples.ScalarMS) == len(samples.CandidateMS),
				fmt.Sprintf("%s.samples[%d]", field, P),
				"both arms must carry equal, >=3 balanced repeats")
			if len(samples.ScalarMS) == 0 || len(samples.CandidateMS) == 0 {
				continue
			}
			// Re-derive the medians from the raw samples so a receipt cannot carry a summary that
			// its own samples do not support.
			wantRatio := medianFloat(samples.ScalarMS) / medianFloat(samples.CandidateMS)
			require(q4kFinite(wantRatio) && q4kReceiptWithinRelative(ratio, wantRatio, 1e-9),
				sampleField, fmt.Sprintf("recorded ratio %.6f must equal median(scalar)/median(candidate) %.6f", ratio, wantRatio))
		}
		if len(band.Measured) > 0 {
			floor := math.Inf(1)
			for _, ratio := range band.Measured {
				if ratio < floor {
					floor = ratio
				}
			}
			require(q4kFinite(band.Floor) && q4kReceiptWithinRelative(band.Floor, floor, 1e-9),
				field+".measured_floor", fmt.Sprintf("recorded floor %.6f must equal the smallest measured ratio %.6f", band.Floor, floor))
			require(band.Floor >= q4kM5CrossoverMargin,
				field+".measured_floor", fmt.Sprintf("floor %.4f is below the fak#9937 gate %.2f", band.Floor, q4kM5CrossoverMargin))
		}
	}

	if len(problems) > 0 {
		return fmt.Errorf("q4k m5 crossover receipt invalid: %s", strings.Join(problems, "; "))
	}
	return nil
}

// q4kFinite reports whether f is neither NaN nor an infinity. Every threshold comparison below is
// a silent lie on a NaN (all comparisons are false) and an Inf floor would clear any gate, so the
// validator rejects non-finite values before trusting a comparison.
func q4kFinite(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) }

// WriteQ4KM5CrossoverReceipt validates a receipt fail-closed and writes it as indented JSON to
// path (creating the parent directory). It returns an error rather than writing an artifact the
// gate would refuse, so an invalid run can never leave an admissible-looking file behind.
func WriteQ4KM5CrossoverReceipt(path string, r Q4KM5CrossoverReceipt) error {
	if err := ValidateQ4KM5CrossoverReceipt(r); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	payload, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	payload = append(payload, '\n')
	return os.WriteFile(path, payload, 0o644)
}
