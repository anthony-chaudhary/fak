package model

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// This file is the V4.1 Flash DSpark RESIDENCY port STUDY (DS4.1 P8, ticket
// 3344). It is a typed, fail-closed record of three things and nothing more:
//
//  1. the pinned upstream shape it adapts from (revision-anchored, license and
//     port class explicit), so a later implementer cannot quietly drift from the
//     studied source;
//  2. the draft-resident / target-streamed residency byte plan and the host
//     budget it is checked against; and
//  3. the code-vs-prose acceptance figures, each labelled by evidence class.
//
// It is a STUDY, not a production envelope. It implements no decoder, no draft
// model, no verify loop, and no KV plan. Its only executable behavior is
// admission/refusal: it must REFUSE a single-node resident-DSpark claim that the
// host budget cannot hold, because the census shape (~152 GiB target + ~7.4 GiB
// draft resident) is a two-node story on one Strix Halo and claiming otherwise
// would be an overclaim. Every acceptance figure below is UPSTREAM-REPORTED,
// not fak-measured; nothing here is a fak performance claim.

// ErrV41DSparkResidencyOverclaim is the typed fail-closed refusal for a
// single-node resident-DSpark claim the host budget cannot hold. It is a named
// errors.Is-comparable sentinel so an admission caller can branch on the exact
// failure instead of string-matching, and so a witness test can prove the
// refusal rather than trusting a message. It is returned only when the claim is
// genuinely unsupported: a >=2-node claim, or a 1-node claim with a passing
// load, is admitted on its own terms.
var ErrV41DSparkResidencyOverclaim = errors.New("model: V4.1 Flash DSpark resident claim exceeds the host budget without a passing load")

// V41DSparkPortClass names how faithfully a pinned upstream anchor is being
// ported. The class is explicit rather than implied because the port study's
// whole value is knowing which sources were adapted, which were ported
// directly, and which merely inspired the shape; collapsing them would let an
// INSPIRE-ONLY source masquerade as an executable port.
type V41DSparkPortClass string

const (
	// V41DSparkPortAdapt marks a source whose structure is adapted, not copied
	// verbatim: the residency/acceptance shape is taken, but the implementation
	// is fak's own. This is the class of all four pinned anchors.
	V41DSparkPortAdapt V41DSparkPortClass = "ADAPT"
	// V41DSparkPortDirectPort marks a source intended to be ported directly. No
	// pin below uses it yet; it is retained so a future direct port cannot be
	// silently reclassified as an adaptation.
	V41DSparkPortDirectPort V41DSparkPortClass = "DIRECT-PORT"
	// V41DSparkPortInspireOnly marks a source that only inspired the design and
	// contributes no executable code. It is retained so an inspiration-only
	// citation can never be counted as an implemented port.
	V41DSparkPortInspireOnly V41DSparkPortClass = "INSPIRE-ONLY"
)

// v41DSparkPortClassValid reports whether c is one of the three closed-set port
// classes. An unset zero value is invalid on purpose: a pin with no declared
// class must fail validation rather than pass as an unclassified adaptation.
func v41DSparkPortClassValid(c V41DSparkPortClass) bool {
	switch c {
	case V41DSparkPortAdapt, V41DSparkPortDirectPort, V41DSparkPortInspireOnly:
		return true
	default:
		return false
	}
}

// V41DSparkSourcePin anchors one upstream artifact this study adapts from: the
// repository, the exact 40-hex revision, the path, the file's line range, the
// blob SHA when one was fetched, the license, and the port class.
//
// The revision is pinned by full commit hash, never a branch or tag, because a
// moving ref would make the study unreproducible the moment upstream advanced.
// LineStart/LineEnd may both be 0 for a documentation-only pin that has no
// meaningful line window; the SGLang code pin carries the real range.
type V41DSparkSourcePin struct {
	// Repo is the upstream repository in owner/name form.
	Repo string
	// Revision is the exact upstream commit (a full 40-character lowercase hex
	// SHA), not a branch or tag.
	Revision string
	// Path is the repository-relative path of the pinned artifact.
	Path string
	// LineStart is the 1-based first line of the studied window, or 0 for a
	// documentation-only pin with no line anchor.
	LineStart int
	// LineEnd is the 1-based last line of the studied window, or 0 alongside a
	// zero LineStart for a documentation-only pin.
	LineEnd int
	// BlobSHA is the 40-hex git blob hash when the artifact was fetched, or
	// empty for documentation pins whose blob was not needed.
	BlobSHA string
	// License is the SPDX-style license id of the pinned artifact ("Apache-2.0"
	// or "MIT" for this study). It is validated against a closed set so a pin
	// can never be admitted with an unknown or missing license.
	License string
	// Class is the port class: ADAPT, DIRECT-PORT, or INSPIRE-ONLY. An unset
	// class fails validation rather than passing as an implied adaptation.
	Class V41DSparkPortClass
}

// V41DSparkPins returns the four revision-pinned anchors of the DSpark port
// study, verified 2026-10-13. It returns a fresh copy each call so a caller
// cannot mutate the study's canonical pins in place.
//
// The SGLang pin is the composite DSpark decoder file and therefore carries its
// real 1..1203 line range. The three ds4 pins are documentation/config anchors
// with no studied line window, so their LineStart/LineEnd are 0.
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

// v41DSparkRevisionValid reports whether rev is exactly 40 lowercase hex
// characters. The check is strict (no uppercase, no short SHA) because a
// truncated or case-variant revision is not a reproducible pin.
func v41DSparkRevisionValid(rev string) bool {
	if len(rev) != 40 {
		return false
	}
	return strings.Trim(rev, "0123456789abcdef") == ""
}

// v41DSparkLicenseValid reports whether license is one of the two licenses this
// study admits. A closed set is used so a pin cannot be admitted with an
// unknown or empty license, which would leave its reuse rights unclear.
func v41DSparkLicenseValid(license string) bool {
	switch license {
	case "Apache-2.0", "MIT":
		return true
	default:
		return false
	}
}

// ValidateV41DSparkPins is the fail-closed validator for a pin set. An empty or
// nil set is refused (a study with no anchored sources is unverifiable), every
// repo, path and license must be present and the license must be in the admitted
// set, every revision must be a full 40-hex lowercase commit, every class must
// be one of the three known values, and the line window must satisfy
// LineStart >= 0 and LineEnd >= LineStart (both 0 allowed for a doc-only pin).
// A code anchor (LineStart > 0) MUST carry a non-empty BlobSHA so its studied
// lines are content-addressed; a documentation-only pin (LineStart == 0) may
// leave BlobSHA empty. Any present BlobSHA (and any present code-anchor blob)
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

// V41DSparkWorkload names the acceptance workload a row measures. It is a
// closed two-value set because the ticket's core finding is the code-vs-prose
// gap: a row whose workload is neither is unclassifiable and must not be
// counted toward that finding.
type V41DSparkWorkload string

const (
	// V41DSparkWorkloadCode is the code-editing acceptance workload.
	V41DSparkWorkloadCode V41DSparkWorkload = "code"
	// V41DSparkWorkloadProse is the natural-language/prose acceptance workload.
	V41DSparkWorkloadProse V41DSparkWorkload = "prose"
)

// V41DSparkAcceptance is one UPSTREAM-REPORTED acceptance row: the baseline and
// DSpark decode rates, the accepted tokens per speculative cycle, the
// acceptance rate, whether the output was byte-identical to the non-speculative
// baseline, and the evidence class.
//
// EvidenceClass is carried explicitly so no reader can mistake an upstream
// report for a fak measurement. This study does not run these rates and does
// not claim them.
type V41DSparkAcceptance struct {
	// Workload is the measured workload ("code" or "prose").
	Workload V41DSparkWorkload
	// BaselineTokPerSec is the upstream no-speculation decode rate in tokens/s.
	BaselineTokPerSec float64
	// DSparkTokPerSec is the upstream DSpark decode rate in tokens/s. A study is
	// only meaningful when this exceeds BaselineTokPerSec.
	DSparkTokPerSec float64
	// AcceptedTokensPerCycle is the upstream accepted tokens per speculative
	// cycle. It must be positive for the row to describe real speculation.
	AcceptedTokensPerCycle float64
	// AcceptanceRate is the upstream fraction of drafted tokens accepted, in
	// (0,1]. A non-positive or out-of-range rate is refused.
	AcceptanceRate float64
	// ByteIdentical records whether the upstream reported output was
	// byte-identical to its non-speculative baseline. This study only records
	// the upstream label; it does not itself verify bit-exactness.
	ByteIdentical bool
	// EvidenceClass labels the provenance of the row. Every row in this study is
	// V41DSparkEvidenceUpstreamReported and therefore NOT fak-measured.
	EvidenceClass string
}

// V41DSparkEvidenceUpstreamReported is the only evidence class carried by this
// study's acceptance rows. It states plainly that the figures are reproduced
// from the upstream report and have not been measured on fak hardware.
const V41DSparkEvidenceUpstreamReported = "UPSTREAM-REPORTED"

// V41DSparkGiB is one gibibyte (2^30 bytes) as an int64. Byte plans below use
// it so the census shape is expressed in the same power-of-two units the
// hardware vendors quote.
const V41DSparkGiB int64 = 1 << 30

// V41DSparkDefaultTargetBytes is the census target-model resident footprint,
// ~152 GiB. It is a documented planning figure, not a measured fak allocation.
const V41DSparkDefaultTargetBytes int64 = 152 * V41DSparkGiB

// V41DSparkDefaultDraftBytes is the census DSpark draft-model resident
// footprint, ~7.4 GiB (3 stages, block size 5, Markov rank 256). It is a
// documented planning figure, not a measured fak allocation. It is expressed as
// 74 GiB / 10 (integer division truncates to the same 7945689497 bytes a float
// conversion would) so it stays a compile-time constant.
const V41DSparkDefaultDraftBytes int64 = 74 * V41DSparkGiB / 10

// V41DSparkDefaultHostBudgetBytes is the residency budget of one Strix Halo
// (128 GiB). The census target+draft residency exceeds it, which is exactly why
// a single-node resident claim is refused below without a passing load.
const V41DSparkDefaultHostBudgetBytes int64 = 128 * V41DSparkGiB

// V41DSparkDefaultNodeCount is the number of nodes the census residency shape
// requires: two. A one-node resident claim is the overclaim this contract
// exists to refuse.
const V41DSparkDefaultNodeCount = 2

// V41DSparkResidency is the draft-resident / target-streamed residency plan:
// the draft bytes held resident, the target bytes streamed, the per-node host
// budget, the node count, and the two residency booleans.
type V41DSparkResidency struct {
	// DraftBytes is the draft-model bytes held resident in host memory.
	DraftBytes int64
	// TargetBytes is the target-model bytes streamed rather than held resident.
	TargetBytes int64
	// HostBudgetBytes is the per-node host memory budget the plan is checked
	// against (one Strix Halo by default).
	HostBudgetBytes int64
	// NodeCount is how many hosts the plan spans. The census shape requires >= 2.
	NodeCount int
	// DraftResident records the invariant that the draft stages stay resident.
	DraftResident bool
	// TargetStreamed records the invariant that the target model streams rather
	// than staying resident. Both booleans default false in the zero value, so a
	// zero-value residency plan claims nothing.
	TargetStreamed bool
}

// ResidentBytes is the total bytes the plan would have to hold resident:
// DraftBytes + TargetBytes. It is the quantity compared against the host budget;
// computing it in one place keeps FitsHost and AdmitResidentClaim from drifting.
func (r V41DSparkResidency) ResidentBytes() int64 {
	return r.DraftBytes + r.TargetBytes
}

// FitsHost reports whether the resident bytes fit the per-node host budget. It
// is exactly ResidentBytes() <= HostBudgetBytes and nothing else, so a caller
// can predict the admission seam's memory test without re-deriving it. Note
// that the zero value (0 <= 0) trivially fits; AdmitResidentClaim separately
// refuses a zero-value plan at one node so "fits" cannot be confused with
// "claims residency".
func (r V41DSparkResidency) FitsHost() bool {
	return r.ResidentBytes() <= r.HostBudgetBytes
}

// V41DSparkStudy is the whole port study: the pinned sources, the residency
// plan, and the acceptance rows. The zero value is inert and claims nothing:
// its pins are empty (so Validate fails), its residency booleans are false, and
// its acceptance list is empty. Nothing about a DSpark residency may be
// inferred from an unconfigured study.
type V41DSparkStudy struct {
	// Pins are the revision-anchored upstream sources. A study must have at
	// least one; ValidateV41DSparkPins refuses an empty set.
	Pins []V41DSparkSourcePin
	// Residency is the draft-resident / target-streamed byte plan.
	Residency V41DSparkResidency
	// Acceptance holds the UPSTREAM-REPORTED acceptance rows. Both the code and
	// prose workloads must be present for Validate to pass, because the
	// code-vs-prose gap is the study's finding.
	Acceptance []V41DSparkAcceptance
}

// DefaultV41DSparkStudy returns the study the ticket defines: the four pinned
// anchors, the census ~152 GiB target + ~7.4 GiB draft residency against a
// 128 GiB one-Strix-Halo budget at two nodes, and the two UPSTREAM-REPORTED
// acceptance rows (code 13.8 -> 31.7 at 3.6 accepted/cycle and rate 0.975;
// prose 13.4 -> 16.6 at 1.5 accepted/cycle and rate 0.55). All figures are
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

// AcceptanceFor returns the acceptance row for workload w and whether one was
// present. Duplicate rows are resolved first-wins so the lookup is
// deterministic; a caller that needs to detect duplicates should validate the
// slice itself. ok=false is the fail-closed signal that the workload carries no
// evidence.
func (s V41DSparkStudy) AcceptanceFor(w V41DSparkWorkload) (V41DSparkAcceptance, bool) {
	for _, a := range s.Acceptance {
		if a.Workload == w {
			return a, true
		}
	}
	return V41DSparkAcceptance{}, false
}

// v41DSparkFinitePositive reports whether v is a finite, strictly positive
// number. NaN and infinities are treated as invalid because a rate that is not
// a real number cannot describe a measurement.
func v41DSparkFinitePositive(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v > 0
}

// v41DSparkValidateAcceptance is the fail-closed check for a single acceptance
// row: the workload must be one of the two known values, the evidence class
// must be exactly V41DSparkEvidenceUpstreamReported, the baseline and DSpark
// rates must be finite and positive with DSpark strictly greater, the accepted
// tokens per cycle must be finite and positive, and the acceptance rate must be
// finite and in (0,1]. A NaN rate, a non-positive rate, a rate above 1, or a
// non-improving DSpark rate all refuse rather than being silently clamped. The
// evidence class is validated rather than trusted because the study's central
// promise is that every figure is unmistakably UPSTREAM-REPORTED, not
// fak-measured; an empty or unadmitted class would undercut that promise.
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

// Validate is the fail-closed consistency gate for a whole study. It refuses
// a study that anchors nothing (empty pins), a residency plan that does not
// actually claim the census shape (fewer than two nodes, either residency
// invariant false, no resident bytes, or no host budget), a study missing
// either the code or the prose acceptance row, and any acceptance row whose
// rates are non-finite, non-positive, non-improving, or whose acceptance rate
// falls outside (0,1]. It proves consistency, not measurement: passing here
// says the study is well-formed, never that its UPSTREAM-REPORTED figures were
// reproduced on fak.
func (s V41DSparkStudy) Validate() error {
	if err := ValidateV41DSparkPins(s.Pins); err != nil {
		return err
	}
	r := s.Residency
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

// AdmitResidentClaim decides whether a resident-DSpark claim may be made for
// nodeCount nodes. It is the study's one admission seam, and it is fail-closed:
//
//   - nodeCount < 1 is refused outright: a claim spanning no host is
//     meaningless.
//   - nodeCount >= 2 is refused with ErrV41DSparkResidencyOverclaim when the
//     study does not actually declare the draft-resident / target-streamed
//     plan (no resident bytes, or either residency invariant false); a study
//     that declares that plan at two nodes is admitted, because the census
//     shape is the documented two-node story. An unconfigured study therefore
//     claims nothing at any node count.
//   - nodeCount == 1 is refused with ErrV41DSparkResidencyOverclaim when the
//     study's resident bytes are non-positive (it declares no plan at all),
//     even if loadPassed is true; otherwise it is admitted when either a
//     passing load was supplied (loadPassed) or the resident bytes fit the
//     one-node host budget.
//
// The zero-value study must not silently admit a claim at any node count, so a
// plan that declares no resident bytes (or does not declare the residency
// invariants) is refused as an overclaim even though its trivial 0 <= 0
// comparison would otherwise "fit". loadPassed overrides a
// declared-but-unfitting plan, never a study that declares no plan at all.
// Passing loadPassed asserts a real measured load, which this study cannot
// supply itself; the caller owns that evidence and any label attached to it.
// This method never claims residency on the caller's behalf.
func (s V41DSparkStudy) AdmitResidentClaim(nodeCount int, loadPassed bool) error {
	if nodeCount < 1 {
		return fmt.Errorf("model: V4.1 Flash DSpark resident claim needs at least one node, got %d", nodeCount)
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
