package model

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// This file is the V4.1 Flash DSpark RESIDENCY port STUDY (DS4.1 P8, ticket 3344): a typed,
// fail-closed record of (1) the revision-pinned upstream sources it adapts from, (2) the
// draft-resident / target-streamed residency byte plan and the host budget it is checked against,
// and (3) the code-vs-prose acceptance figures, each labelled by evidence class.
//
// It is a STUDY, not a production envelope: no decoder, no draft model, no verify loop, no KV
// plan. Its only executable behavior is admission/refusal — it must REFUSE a single-node
// resident-DSpark claim the host budget cannot hold, because the census shape (~152 GiB target +
// ~7.4 GiB draft resident) is a two-node story on one Strix Halo. Every acceptance figure is
// UPSTREAM-REPORTED, not fak-measured; nothing here is a fak performance claim.

// ErrV41DSparkResidencyOverclaim is the typed fail-closed refusal for a resident-DSpark claim the
// study cannot support, as an errors.Is-comparable sentinel so callers and witness tests branch on
// the exact failure. A >=2-node claim, or a 1-node claim with a passing load, is admitted on its
// own terms.
var ErrV41DSparkResidencyOverclaim = errors.New("model: V4.1 Flash DSpark resident claim exceeds the host budget without a passing load")

// V41DSparkPortClass names how faithfully a pinned upstream anchor is being ported, explicitly
// rather than implied: collapsing the classes would let an INSPIRE-ONLY source masquerade as an
// executable port.
type V41DSparkPortClass string

const (
	// Adapt: the structure is adapted, not copied verbatim — the shape is taken, the implementation
	// is fak's own. The class of all four pinned anchors.
	V41DSparkPortAdapt V41DSparkPortClass = "ADAPT"
	// DirectPort: intended to be ported directly. Retained so a future direct port cannot be
	// silently reclassified as an adaptation.
	V41DSparkPortDirectPort V41DSparkPortClass = "DIRECT-PORT"
	// InspireOnly: only inspired the design, contributes no executable code; can never be counted
	// as an implemented port.
	V41DSparkPortInspireOnly V41DSparkPortClass = "INSPIRE-ONLY"
)

// v41DSparkPortClassValid reports whether c is one of the three closed-set port classes. An unset
// zero value is invalid on purpose: a pin with no declared class must fail validation.
func v41DSparkPortClassValid(c V41DSparkPortClass) bool {
	switch c {
	case V41DSparkPortAdapt, V41DSparkPortDirectPort, V41DSparkPortInspireOnly:
		return true
	default:
		return false
	}
}

// V41DSparkSourcePin anchors one upstream artifact this study adapts from: repository, exact
// 40-hex revision, path, line range, blob SHA when fetched, license, and port class. The revision
// is pinned by full commit hash, never a branch or tag — a moving ref would make the study
// unreproducible the moment upstream advanced. LineStart/LineEnd may both be 0 for a
// documentation-only pin.
type V41DSparkSourcePin struct {
	// Repo is the upstream repository in owner/name form.
	Repo string
	// Revision is the exact upstream commit (full 40-character lowercase hex SHA), not a branch or tag.
	Revision string
	// Path is the repository-relative path of the pinned artifact.
	Path string
	// LineStart / LineEnd bound the studied window (1-based); both 0 for a documentation-only pin.
	LineStart int
	LineEnd   int
	// BlobSHA is the 40-hex git blob hash when the artifact was fetched; empty for doc-only pins.
	BlobSHA string
	// License is the SPDX-style license id ("Apache-2.0" or "MIT"), validated against a closed set.
	License string
	// Class is the port class; an unset class fails validation.
	Class V41DSparkPortClass
}

// V41DSparkPins returns the four revision-pinned anchors of the DSpark port study, verified
// 2026-10-13, as a fresh copy each call. The SGLang pin is the composite DSpark decoder file and
// carries its real 1..1203 line range; the three ds4 pins are documentation/config anchors with
// no studied line window.
func V41DSparkPins() []V41DSparkSourcePin {
	return []V41DSparkSourcePin{
		{
			Repo:      "sgl-project/sglang",
			Revision:  "45418d7dc648d0d71501b8b905a2499a40e9b971",
			Path:      "python/sglang/srt/models/deepseek_v4_dspark.py",
			LineStart: 1,
			LineEnd:   1203,
			BlobSHA:   "62b57766582dcfbed071c8f4f3a5d347328264f3",
			License:   "Apache-2.0",
			Class:     V41DSparkPortAdapt,
		},
		{
			Repo:     "evolvemarketingitalia/ds4",
			Revision: "5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75",
			Path:     "docs/STRIX_HALO_DSPARK.md",
			License:  "MIT",
			Class:    V41DSparkPortAdapt,
		},
		{
			Repo:     "evolvemarketingitalia/ds4",
			Revision: "5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75",
			Path:     "docs/SPECULATIVE_DECODING.md",
			License:  "MIT",
			Class:    V41DSparkPortAdapt,
		},
		{
			Repo:     "evolvemarketingitalia/ds4",
			Revision: "5c3fde8e2cbcb7e1a4e81e1bca2db528e9b56c75",
			Path:     "tools/strix-halo/dspark41_config.json",
			License:  "MIT",
			Class:    V41DSparkPortAdapt,
		},
	}
}

// v41DSparkRevisionValid reports whether rev is exactly 40 lowercase hex characters; a truncated
// or case-variant revision is not a reproducible pin.
func v41DSparkRevisionValid(rev string) bool {
	if len(rev) != 40 {
		return false
	}
	return strings.Trim(rev, "0123456789abcdef") == ""
}

// v41DSparkLicenseValid reports whether license is one of the two licenses this study admits, so
// a pin cannot be admitted with an unknown or empty license.
func v41DSparkLicenseValid(license string) bool {
	switch license {
	case "Apache-2.0", "MIT":
		return true
	default:
		return false
	}
}

// ValidateV41DSparkPins is the fail-closed validator for a pin set: an empty set is refused
// (unverifiable), every repo/path/license must be present with an admitted license, every
// revision a full 40-hex lowercase commit, every class one of the three known values, and the
// line window must satisfy LineStart >= 0 and LineEnd >= LineStart. A code anchor (LineStart > 0)
// MUST carry a non-empty BlobSHA so its studied lines are content-addressed; any present BlobSHA
// must itself be a full 40-hex lowercase blob hash.
func ValidateV41DSparkPins(pins []V41DSparkSourcePin) error {
	if len(pins) == 0 {
		return fmt.Errorf("model: V4.1 Flash DSpark pin set is empty; a port study with no anchored source is unverifiable")
	}
	for i, p := range pins {
		if p.Repo == "" {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d has an empty repo", i)
		}
		if !v41DSparkRevisionValid(p.Revision) {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d (%s %s) revision %q is not a full 40-hex lowercase commit", i, p.Repo, p.Path, p.Revision)
		}
		if p.Path == "" {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d (%s) has an empty path", i, p.Repo)
		}
		if p.License == "" {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d (%s %s) has an empty license", i, p.Repo, p.Path)
		}
		if !v41DSparkLicenseValid(p.License) {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d (%s %s) has unadmitted license %q", i, p.Repo, p.Path, p.License)
		}
		if !v41DSparkPortClassValid(p.Class) {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d (%s %s) has unadmitted port class %q", i, p.Repo, p.Path, p.Class)
		}
		if p.LineStart < 0 || p.LineEnd < p.LineStart {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d (%s %s) has invalid line window [%d,%d]", i, p.Repo, p.Path, p.LineStart, p.LineEnd)
		}
		if p.LineStart > 0 && p.BlobSHA == "" {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d (%s %s) is a code anchor (LineStart %d) with an empty BlobSHA", i, p.Repo, p.Path, p.LineStart)
		}
		if p.BlobSHA != "" && !v41DSparkRevisionValid(p.BlobSHA) {
			return fmt.Errorf("model: V4.1 Flash DSpark pin %d (%s %s) blob %q is not a full 40-hex lowercase blob hash", i, p.Repo, p.Path, p.BlobSHA)
		}
	}
	return nil
}

// V41DSparkWorkload names the acceptance workload a row measures. Closed two-value set because
// the study's core finding is the code-vs-prose gap: a row that is neither is unclassifiable.
type V41DSparkWorkload string

const (
	// V41DSparkWorkloadCode is the code-editing acceptance workload.
	V41DSparkWorkloadCode V41DSparkWorkload = "code"
	// V41DSparkWorkloadProse is the natural-language/prose acceptance workload.
	V41DSparkWorkloadProse V41DSparkWorkload = "prose"
)

// V41DSparkAcceptance is one UPSTREAM-REPORTED acceptance row: the baseline and DSpark decode
// rates, the accepted tokens per speculative cycle, the acceptance rate, whether the output was
// byte-identical to the non-speculative baseline, and the evidence class. EvidenceClass is
// carried explicitly so no reader can mistake an upstream report for a fak measurement.
type V41DSparkAcceptance struct {
	// Workload is the measured workload ("code" or "prose").
	Workload V41DSparkWorkload
	// BaselineTokPerSec is the upstream no-speculation decode rate in tokens/s.
	BaselineTokPerSec float64
	// DSparkTokPerSec is the upstream DSpark decode rate in tokens/s; a study is only meaningful
	// when this exceeds BaselineTokPerSec.
	DSparkTokPerSec float64
	// AcceptedTokensPerCycle is the upstream accepted tokens per speculative cycle; positive for
	// real speculation.
	AcceptedTokensPerCycle float64
	// AcceptanceRate is the upstream fraction of drafted tokens accepted, in (0,1].
	AcceptanceRate float64
	// ByteIdentical records whether the upstream reported output was byte-identical to its
	// non-speculative baseline; this study only records the label, it does not verify it.
	ByteIdentical bool
	// EvidenceClass labels the provenance of the row; every row here is UPSTREAM-REPORTED.
	EvidenceClass string
}

// V41DSparkEvidenceUpstreamReported is the only evidence class carried by this study's acceptance
// rows: the figures are reproduced from the upstream report, not measured on fak hardware.
const V41DSparkEvidenceUpstreamReported = "UPSTREAM-REPORTED"

// V41DSparkGiB is one gibibyte (2^30 bytes), so the census shape is expressed in the same
// power-of-two units the hardware vendors quote.
const V41DSparkGiB int64 = 1 << 30

// V41DSparkDefaultTargetBytes is the census target-model resident footprint, ~152 GiB — a
// documented planning figure, not a measured fak allocation.
const V41DSparkDefaultTargetBytes int64 = 152 * V41DSparkGiB

// V41DSparkDefaultDraftBytes is the census DSpark draft-model resident footprint, ~7.4 GiB (3
// stages, block size 5, Markov rank 256) — a planning figure, not a measured allocation. Expressed
// as 74 GiB / 10 (integer division truncates to the same 7945689497 bytes a float conversion
// would) so it stays a compile-time constant.
const V41DSparkDefaultDraftBytes int64 = 74 * V41DSparkGiB / 10

// V41DSparkDefaultHostBudgetBytes is the residency budget of one Strix Halo (128 GiB). The census
// target+draft residency exceeds it, which is why a single-node resident claim is refused without
// a passing load.
const V41DSparkDefaultHostBudgetBytes int64 = 128 * V41DSparkGiB

// V41DSparkDefaultNodeCount is the number of nodes the census residency shape requires: two. A
// one-node resident claim is the overclaim this contract exists to refuse.
const V41DSparkDefaultNodeCount = 2

// V41DSparkResidency is the draft-resident / target-streamed residency plan: the draft bytes held
// resident, the target bytes streamed, the per-node host budget, the node count, and the two
// residency booleans (both false in the zero value, so a zero-value plan claims nothing).
type V41DSparkResidency struct {
	// DraftBytes is the draft-model bytes held resident in host memory.
	DraftBytes int64
	// TargetBytes is the target-model bytes streamed rather than held resident.
	TargetBytes int64
	// HostBudgetBytes is the per-node host memory budget the plan is checked against.
	HostBudgetBytes int64
	// NodeCount is how many hosts the plan spans; the census shape requires >= 2.
	NodeCount int
	// DraftResident records the invariant that the draft stages stay resident.
	DraftResident bool
	// TargetStreamed records the invariant that the target model streams rather than staying resident.
	TargetStreamed bool
}

// ResidentBytes is the total bytes the plan would have to hold resident: DraftBytes + TargetBytes,
// for a valid nonnegative, representable byte plan. Admission validates that
// precondition before using this arithmetic. This is the quantity compared against
// the host budget, computed in one place so FitsHost and
// AdmitResidentClaim cannot drift.
func (r V41DSparkResidency) ResidentBytes() int64 {
	return r.DraftBytes + r.TargetBytes
}

// FitsHost reports whether the resident bytes fit the per-node host budget: exactly
// valid byte counts and ResidentBytes() <= HostBudgetBytes. The zero value (0 <= 0) trivially fits; AdmitResidentClaim
// separately refuses a zero-value plan at one node so "fits" cannot be confused with "claims
// residency".
func (r V41DSparkResidency) FitsHost() bool {
	return r.validBytePlan() && r.ResidentBytes() <= r.HostBudgetBytes
}

// validBytePlan checks individual declarations before adding them, so a
// negative component cannot cancel a real requirement or hide overflow.
func (r V41DSparkResidency) validBytePlan() bool {
	return r.DraftBytes >= 0 && r.TargetBytes >= 0 && r.DraftBytes <= math.MaxInt64-r.TargetBytes
}

// V41DSparkStudy is the whole port study: the pinned sources, the residency plan, and the
// acceptance rows. The zero value is inert and claims nothing; nothing about a DSpark residency
// may be inferred from an unconfigured study.
type V41DSparkStudy struct {
	// Pins are the revision-anchored upstream sources; a study must have at least one.
	Pins []V41DSparkSourcePin
	// Residency is the draft-resident / target-streamed byte plan.
	Residency V41DSparkResidency
	// Acceptance holds the UPSTREAM-REPORTED acceptance rows; both the code and prose workloads
	// must be present for Validate to pass, because the code-vs-prose gap is the study's finding.
	Acceptance []V41DSparkAcceptance
}

// DefaultV41DSparkStudy returns the study the ticket defines: the four pinned anchors, the census
// ~152 GiB target + ~7.4 GiB draft residency against a 128 GiB one-Strix-Halo budget at two
// nodes, and the two UPSTREAM-REPORTED acceptance rows (code 13.8 -> 31.7 at 3.6 accepted/cycle
// and rate 0.975; prose 13.4 -> 16.6 at 1.5 accepted/cycle and rate 0.55). All figures are
// upstream-reported planning data, not fak measurements.
func DefaultV41DSparkStudy() V41DSparkStudy {
	return V41DSparkStudy{
		Pins: V41DSparkPins(),
		Residency: V41DSparkResidency{
			DraftBytes:      V41DSparkDefaultDraftBytes,
			TargetBytes:     V41DSparkDefaultTargetBytes,
			HostBudgetBytes: V41DSparkDefaultHostBudgetBytes,
			NodeCount:       V41DSparkDefaultNodeCount,
			DraftResident:   true,
			TargetStreamed:  true,
		},
		Acceptance: []V41DSparkAcceptance{
			{
				Workload:               V41DSparkWorkloadCode,
				BaselineTokPerSec:      13.8,
				DSparkTokPerSec:        31.7,
				AcceptedTokensPerCycle: 3.6,
				AcceptanceRate:         0.975,
				ByteIdentical:          true,
				EvidenceClass:          V41DSparkEvidenceUpstreamReported,
			},
			{
				Workload:               V41DSparkWorkloadProse,
				BaselineTokPerSec:      13.4,
				DSparkTokPerSec:        16.6,
				AcceptedTokensPerCycle: 1.5,
				AcceptanceRate:         0.55,
				ByteIdentical:          true,
				EvidenceClass:          V41DSparkEvidenceUpstreamReported,
			},
		},
	}
}

// AcceptanceFor returns the acceptance row for workload w and whether one was present. Duplicate
// rows resolve first-wins so the lookup is deterministic; ok=false is the fail-closed signal that
// the workload carries no evidence.
func (s V41DSparkStudy) AcceptanceFor(w V41DSparkWorkload) (V41DSparkAcceptance, bool) {
	for _, a := range s.Acceptance {
		if a.Workload == w {
			return a, true
		}
	}
	return V41DSparkAcceptance{}, false
}

// v41DSparkFinitePositive reports whether v is a finite, strictly positive number; a rate that is
// not a real number cannot describe a measurement.
func v41DSparkFinitePositive(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v > 0
}

// v41DSparkValidateAcceptance is the fail-closed check for a single acceptance row: the workload
// must be one of the two known values, the evidence class exactly UPSTREAM-REPORTED (validated
// rather than trusted — the study's central promise is that every figure is unmistakably
// upstream-reported), the baseline and DSpark rates finite and positive with DSpark strictly
// greater, the accepted tokens per cycle finite and positive, and the acceptance rate in (0,1].
// Bad values refuse rather than being silently clamped.
func v41DSparkValidateAcceptance(a V41DSparkAcceptance) error {
	if a.Workload != V41DSparkWorkloadCode && a.Workload != V41DSparkWorkloadProse {
		return fmt.Errorf("model: V4.1 Flash DSpark acceptance row has unknown workload %q", a.Workload)
	}
	if a.EvidenceClass != V41DSparkEvidenceUpstreamReported {
		return fmt.Errorf("model: V4.1 Flash DSpark %s acceptance row has evidence class %q, want %q", a.Workload, a.EvidenceClass, V41DSparkEvidenceUpstreamReported)
	}
	if !v41DSparkFinitePositive(a.BaselineTokPerSec) {
		return fmt.Errorf("model: V4.1 Flash DSpark %s acceptance baseline %v is not a finite positive rate", a.Workload, a.BaselineTokPerSec)
	}
	if !v41DSparkFinitePositive(a.DSparkTokPerSec) || a.DSparkTokPerSec <= a.BaselineTokPerSec {
		return fmt.Errorf("model: V4.1 Flash DSpark %s acceptance DSpark rate %v does not exceed baseline %v", a.Workload, a.DSparkTokPerSec, a.BaselineTokPerSec)
	}
	if !v41DSparkFinitePositive(a.AcceptedTokensPerCycle) {
		return fmt.Errorf("model: V4.1 Flash DSpark %s accepted tokens per cycle %v is not finite and positive", a.Workload, a.AcceptedTokensPerCycle)
	}
	if math.IsNaN(a.AcceptanceRate) || a.AcceptanceRate <= 0 || a.AcceptanceRate > 1 {
		return fmt.Errorf("model: V4.1 Flash DSpark %s acceptance rate %v is outside (0,1]", a.Workload, a.AcceptanceRate)
	}
	return nil
}

// Validate is the fail-closed consistency gate for a whole study: it refuses empty pins, a
// residency plan that does not claim the census shape (fewer than two nodes, either residency
// invariant false, no resident bytes, or no host budget), a missing code or prose acceptance row,
// and any acceptance row with non-finite, non-positive, or non-improving rates or a rate outside
// (0,1]. It proves consistency, not measurement: passing says the study is well-formed, never
// that its UPSTREAM-REPORTED figures were reproduced on fak.
func (s V41DSparkStudy) Validate() error {
	if err := ValidateV41DSparkPins(s.Pins); err != nil {
		return err
	}
	r := s.Residency
	if !r.validBytePlan() {
		return fmt.Errorf("model: V4.1 Flash DSpark residency has negative or overflowing byte counts")
	}
	if r.NodeCount < V41DSparkDefaultNodeCount {
		return fmt.Errorf("model: V4.1 Flash DSpark residency needs at least %d nodes, got %d", V41DSparkDefaultNodeCount, r.NodeCount)
	}
	if !r.DraftResident {
		return fmt.Errorf("model: V4.1 Flash DSpark residency must hold the draft stages resident")
	}
	if !r.TargetStreamed {
		return fmt.Errorf("model: V4.1 Flash DSpark residency must stream the target model")
	}
	if r.ResidentBytes() <= 0 {
		return fmt.Errorf("model: V4.1 Flash DSpark residency declares no resident bytes")
	}
	if r.HostBudgetBytes <= 0 {
		return fmt.Errorf("model: V4.1 Flash DSpark residency declares no host budget")
	}
	for _, w := range []V41DSparkWorkload{V41DSparkWorkloadCode, V41DSparkWorkloadProse} {
		if _, ok := s.AcceptanceFor(w); !ok {
			return fmt.Errorf("model: V4.1 Flash DSpark study is missing the %s acceptance row", w)
		}
	}
	for _, a := range s.Acceptance {
		if err := v41DSparkValidateAcceptance(a); err != nil {
			return err
		}
	}
	return nil
}

// AdmitResidentClaim decides whether a resident-DSpark claim may be made for nodeCount nodes. It
// is the study's one admission seam, and it is fail-closed:
//
//   - nodeCount < 1 is refused outright: a claim spanning no host is meaningless.
//   - nodeCount >= 2 is refused with ErrV41DSparkResidencyOverclaim unless the study declares the
//     draft-resident / target-streamed plan (resident bytes and both invariants); a study that
//     declares it at two nodes is admitted, because the census shape is the documented two-node
//     story. An unconfigured study therefore claims nothing at any node count.
//   - nodeCount == 1 is refused with ErrV41DSparkResidencyOverclaim when the resident bytes are
//     non-positive (no plan at all), even if loadPassed; otherwise it is admitted when a passing
//     load was supplied or the resident bytes fit the one-node host budget.
//
// A plan that declares no resident bytes is refused as an overclaim even though its trivial
// 0 <= 0 comparison would "fit". loadPassed overrides a declared-but-unfitting plan, never a
// study that declares no plan at all. Passing loadPassed asserts a real measured load, which this
// study cannot supply itself; the caller owns that evidence. This method never claims residency on
// the caller's behalf.
func (s V41DSparkStudy) AdmitResidentClaim(nodeCount int, loadPassed bool) error {
	if nodeCount < 1 {
		return fmt.Errorf("model: V4.1 Flash DSpark resident claim needs at least one node, got %d", nodeCount)
	}
	if !s.Residency.validBytePlan() {
		return fmt.Errorf("%w: negative or overflowing residency byte counts", ErrV41DSparkResidencyOverclaim)
	}
	if nodeCount >= V41DSparkDefaultNodeCount {
		if s.Residency.ResidentBytes() <= 0 || !s.Residency.DraftResident || !s.Residency.TargetStreamed {
			return fmt.Errorf("%w: study does not declare the draft-resident / target-streamed plan; an unconfigured study claims nothing at any node count", ErrV41DSparkResidencyOverclaim)
		}
		return nil
	}
	if s.Residency.ResidentBytes() <= 0 {
		return fmt.Errorf("%w: study declares no resident bytes; a zero-value plan cannot substantiate residency", ErrV41DSparkResidencyOverclaim)
	}
	if loadPassed {
		return nil
	}
	if !s.Residency.FitsHost() {
		return fmt.Errorf("%w: %d resident bytes exceed the %d-byte host budget at one node", ErrV41DSparkResidencyOverclaim, s.Residency.ResidentBytes(), s.Residency.HostBudgetBytes)
	}
	return nil
}
