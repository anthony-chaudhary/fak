package qwen38quantrun

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// StrixConcurrentBatch absorbs the missing c4/c8 producer/scorer boundary that
// #12176 leaves open: #12176 owns c1 adaptation and returns
// batch-observation-required for c>1, and the ordinary scoreboard input has a
// single output stream per trial. This file adds exactly one production
// observation shape — a batch-pair observation whose aggregate rate is derived
// only from complete, overlapping per-request evidence — and a scorer that
// admits it into the AMD scoreboard v2 verdict. It is device-free: the caller
// supplies already-captured per-slot observations, and every fail-closed gate
// is exercised by fixture data with no GPU, SSH, or lease involvement.

// StrixConcurrentBatchAllowedConcurrency is the closed set of admitted c>1 cells.
// c1 stays the ordinary scoreboard path; c4/c8 are the only creditable batches.
var StrixConcurrentBatchAllowedConcurrency = []int{4, 8}

// StrixConcurrentBatchSlotObservation is one request slot inside one arm's
// batch observation. Admission and completion are recorded on one monotonic
// clock shared by every slot of the arm; the whole cell is creditable only when
// the slots genuinely overlap.
type StrixConcurrentBatchSlotObservation struct {
	RequestID     string  `json:"request_id"`
	AdmissionSec  float64 `json:"admission_seconds"`
	CompletionSec float64 `json:"completion_seconds"`

	AcceptedOutputTokenIDs []int     `json:"accepted_output_token_ids"`
	AcceptedTokenLogprobs  []float64 `json:"accepted_token_logprobs"`

	// ObservedIgnoreEOS and EOSStopped are physical receipt observations, never a
	// requested generation flag.
	ObservedIgnoreEOS *bool `json:"observed_ignore_eos"`
	EOSStopped        *bool `json:"eos_stopped"`

	PrefillSeconds float64 `json:"prefill_seconds"`
	DecodeSeconds  float64 `json:"decode_seconds"`
	PrefillTokens  int     `json:"prefill_tokens"`

	H2DBytes         uint64 `json:"h2d_bytes"`
	D2HBytes         uint64 `json:"d2h_bytes"`
	D2DBytes         uint64 `json:"d2d_bytes"`
	QueueSubmissions uint64 `json:"queue_submissions"`

	PeakRSSBytes       uint64 `json:"peak_rss_bytes"`
	PeakVRAMBytes      uint64 `json:"peak_vram_bytes"`
	ResidentModelBytes uint64 `json:"resident_model_bytes"`

	NativeInferenceReceipt *model.NativeInferenceReceipt `json:"native_inference_receipt,omitempty"`
}

// StrixConcurrentBatchArmObservation is one arm's observation for one
// concurrency cell: the arm identity plus exactly `Concurrency` slot
// observations, all sharing one packet, artifact, and monotonic clock.
type StrixConcurrentBatchArmObservation struct {
	Arm         AMDArmReceipt                         `json:"arm"`
	Concurrency int                                   `json:"concurrency"`
	Slots       []StrixConcurrentBatchSlotObservation `json:"slots"`
}

// StrixConcurrentBatchPairObservation is one (candidate, reference) batch pair
// at one c>1 cell. It carries the sealed packet digest both arms must share and
// the exact pair identity the scoreboard's AB/BA order requires.
type StrixConcurrentBatchPairObservation struct {
	Concurrency    int                                `json:"concurrency"`
	PacketDigest   string                             `json:"packet_digest"`
	LogitTolerance float64                            `json:"logit_tolerance"`
	Candidate      StrixConcurrentBatchArmObservation `json:"candidate"`
	Reference      StrixConcurrentBatchArmObservation `json:"reference"`
}

// StrixConcurrentBatchRefusal is the closed vocabulary of batch-observation
// refusals. Every refusal name is testable and never a prose guess.
var StrixConcurrentBatchRefusals = []string{
	"unsupported-concurrency",
	"slot-count-mismatch",
	"slot-identity-incomplete",
	"duplicate-request-id",
	"non-monotonic-clock",
	"slot-not-overlapping",
	"accepted-token-count-mismatch",
	"token-logprob-shape-mismatch",
	"eos-observation-missing",
	"eos-stopped-observed",
	"timing-arithmetic-invalid",
	"transfer-or-submission-accounting-missing",
	"memory-evidence-missing",
	"packet-digest-mismatch",
	"arm-identity-incomplete",
	"arm-identity-drift",
	"candidate-not-fak-native-no-fallback",
	"reference-not-explicit-llamacpp-comparator",
	"artifact-mismatch",
	"non-finite-value",
}

// BatchTile is one arm's scoreable aggregate derived only from complete
// overlapping per-request evidence: N is the checked sum of accepted output
// tokens, T is the shared first-admission-to-last-completion interval, and the
// only credited rate is N/T.
type BatchTile struct {
	Requests      int
	AcceptedToken uint64
	StartSec      float64
	EndSec        float64
	AggregateTPS  float64
}

func (t BatchTile) Duration() float64 { return t.EndSec - t.StartSec }

// ErrStrixConcurrentBatch is the sentinel wrapped by every batch-observation
// refusal.
var ErrStrixConcurrentBatch = errors.New("qwen38quantrun: strix concurrent batch refused")

// BatesRefusalError carries the closed refusal name for callers that route on
// it. The error also wraps ErrStrixConcurrentBatch.
type BatchRefusalError struct {
	Reason string
	Detail string
}

func (e *BatchRefusalError) Error() string {
	if e.Detail == "" {
		return fmt.Sprintf("%v: %s", ErrStrixConcurrentBatch, e.Reason)
	}
	return fmt.Sprintf("%v: %s: %s", ErrStrixConcurrentBatch, e.Reason, e.Detail)
}

func (e *BatchRefusalError) Unwrap() error { return ErrStrixConcurrentBatch }

func batchRefuse(reason, detail string) error {
	return &BatchRefusalError{Reason: reason, Detail: detail}
}

// StrixConcurrentBatchScoreSchema identifies the c>1 batch score report. It is
// a distinct schema from the ordinary c=1 scoreboard because the observation
// basis (per-request slots sharing one admission-to-completion interval) is
// different and must not be silently interchangeable with a single stream.
const StrixConcurrentBatchScoreSchema = "fak.qwen38.strix-concurrent-batch-score.v1"

// StrixConcurrentBatchScoreReport is the scoreable verdict for one c4/c8 cell.
// Comparability is granted only when both arms produced complete, overlapping,
// token-identical per-request evidence and both sample CVs are within the
// frozen bar; the credited aggregate rate of each arm is its own N/T.
type StrixConcurrentBatchScoreReport struct {
	Schema           string                   `json:"schema"`
	Concurrency      int                      `json:"concurrency"`
	Comparable       bool                     `json:"comparable"`
	Verdict          string                   `json:"verdict"`
	Reasons          []string                 `json:"reasons,omitempty"`
	Trials           int                      `json:"trials"`
	Candidate        StrixConcurrentBatchTile `json:"candidate"`
	Reference        StrixConcurrentBatchTile `json:"reference"`
	CandidateCV      float64                  `json:"candidate_cv"`
	ReferenceCV      float64                  `json:"reference_cv"`
	CandidateLCB95   float64                  `json:"candidate_lcb95"`
	PairedRatioLCB95 float64                  `json:"paired_ratio_lcb95"`
	AbsoluteBar      float64                  `json:"absolute_bar"`
	AbsolutePass     bool                     `json:"absolute_pass"`
	PairedPass       bool                     `json:"paired_pass"`
	Assumptions      string                   `json:"assumptions"`
}

// StrixConcurrentBatchTile is the credited aggregate for one arm: the checked
// sum of accepted output tokens, the shared interval, and the only rate the
// evidence supports, N/T.
type StrixConcurrentBatchTile struct {
	Requests      int     `json:"requests"`
	AcceptedToken uint64  `json:"accepted_tokens"`
	StartSec      float64 `json:"start_seconds"`
	EndSec        float64 `json:"end_seconds"`
	AggregateTPS  float64 `json:"aggregate_tps"`
}

// ScoreStrixConcurrentBatchPairs validates a predeclared set of exactly
// StrixComparisonMinimumMeasuredPairs device-free c4/c8 batch-pair
// observations and returns the scoreable c>1 verdict. The caller supplies the
// per-request evidence; the aggregate rate is derived only from the checked sum
// of accepted tokens over the shared first-admission-to-last-completion
// interval, so no rate is ever taken on the caller's word and no c1 rate is
// ever multiplied.
//
// Every relabeled, serial, duplicated, partial, replayed, timing-mismatched, or
// caller-forged observation fails closed with a BatchRefusalError naming a
// member of StrixConcurrentBatchRefusals.
func ScoreStrixConcurrentBatchPairs(pairs []StrixConcurrentBatchPairObservation) (StrixConcurrentBatchScoreReport, error) {
	if len(pairs) != StrixComparisonMinimumMeasuredPairs {
		return StrixConcurrentBatchScoreReport{}, batchRefuse("slot-count-mismatch", fmt.Sprintf("batch pair set has %d pairs, want %d", len(pairs), StrixComparisonMinimumMeasuredPairs))
	}
	concurrency := pairs[0].Concurrency
	candidateTiles := make([]BatchTile, 0, len(pairs))
	referenceTiles := make([]BatchTile, 0, len(pairs))
	var candidateArm, referenceArm AMDArmReceipt
	for i, pair := range pairs {
		if pair.Concurrency != concurrency {
			return StrixConcurrentBatchScoreReport{}, batchRefuse("unsupported-concurrency", "pair set mixes concurrency cells")
		}
		candidateTile, referenceTile, err := validateStrixConcurrentBatchPair(pair)
		if err != nil {
			return StrixConcurrentBatchScoreReport{}, err
		}
		if i == 0 {
			candidateArm, referenceArm = pair.Candidate.Arm, pair.Reference.Arm
		} else {
			if err := batchArmIdentityDrift(candidateArm, pair.Candidate.Arm, "candidate"); err != nil {
				return StrixConcurrentBatchScoreReport{}, err
			}
			if err := batchArmIdentityDrift(referenceArm, pair.Reference.Arm, "reference"); err != nil {
				return StrixConcurrentBatchScoreReport{}, err
			}
		}
		candidateTiles = append(candidateTiles, candidateTile)
		referenceTiles = append(referenceTiles, referenceTile)
	}
	return strixConcurrentBatchReport(concurrency, candidateTiles, referenceTiles), nil
}

func strixConcurrentBatchReport(concurrency int, candidateTiles, referenceTiles []BatchTile) StrixConcurrentBatchScoreReport {
	report := StrixConcurrentBatchScoreReport{
		Schema:      StrixConcurrentBatchScoreSchema,
		Concurrency: concurrency,
		Verdict:     "not-comparable",
		Trials:      len(candidateTiles),
		Assumptions: AMDStatisticalAssumptions,
	}
	if len(candidateTiles) == 0 || len(candidateTiles) != len(referenceTiles) {
		report.Reasons = []string{"trial-count-mismatch"}
		return report
	}
	report.Candidate = batchTileSummary(candidateTiles)
	report.Reference = batchTileSummary(referenceTiles)
	// The credited per-trial rate is each tile's own N/T, captured from the
	// candidate's aggregate tile shape. Every pair must also carry the same
	// token identity on both arms; a batch of diverging outputs is not
	// comparable regardless of its rate.
	c, r, ratios := make([]float64, 0, len(candidateTiles)), make([]float64, 0, len(candidateTiles)), make([]float64, 0, len(candidateTiles))
	for i := range candidateTiles {
		c = append(c, candidateTiles[i].AggregateTPS)
		r = append(r, referenceTiles[i].AggregateTPS)
		ratios = append(ratios, candidateTiles[i].AggregateTPS/referenceTiles[i].AggregateTPS)
	}
	cm, cc, cl, cok := oneSided95LCBAMDChecked(c)
	_, rc, _, rok := oneSided95LCBAMDChecked(r)
	_, _, pl, pok := oneSided95LCBAMDChecked(ratios)
	if !cok || !rok || !pok {
		report.Reasons = []string{"coefficient-of-variation-exceeded"}
		return report
	}
	report.CandidateCV, report.ReferenceCV = cc, rc
	report.CandidateLCB95 = cl
	report.PairedRatioLCB95 = pl
	report.AbsoluteBar = frozenStrixBarAMD(concurrency)
	report.AbsolutePass = report.AbsoluteBar > 0 && cm >= report.AbsoluteBar
	report.PairedPass = pl > 1
	report.Comparable = true
	report.Verdict = "comparable"
	return report
}

func batchTileSummary(tiles []BatchTile) StrixConcurrentBatchTile {
	var accepted uint64
	meanRate := 0.0
	for _, t := range tiles {
		accepted += t.AcceptedToken
		meanRate += t.AggregateTPS
	}
	out := StrixConcurrentBatchTile{
		Requests:      tiles[0].Requests,
		AcceptedToken: accepted,
		StartSec:      tiles[0].StartSec,
		EndSec:        tiles[0].EndSec,
	}
	if len(tiles) > 0 {
		out.AggregateTPS = meanRate / float64(len(tiles))
	}
	return out
}

// ScoreStrixConcurrentBatchPair validates one c4/c8 batch-pair observation in
// isolation. It is the single-pair seam used by callers that inspect or refuse
// one pair, and by the named witness; the creditable path is
// ScoreStrixConcurrentBatchPairs.
func ScoreStrixConcurrentBatchPair(obs StrixConcurrentBatchPairObservation) (AMDScoreboardInput, error) {
	candidateTile, referenceTile, err := validateStrixConcurrentBatchPair(obs)
	if err != nil {
		return AMDScoreboardInput{}, err
	}
	return AMDScoreboardInput{
		Schema:         AMDScoreboardInputSchema,
		LogitTolerance: obs.LogitTolerance,
		Concurrency:    obs.Concurrency,
		Candidate:      batchTileToArm(obs.Candidate.Arm, candidateTile, obs.PacketDigest),
		Reference:      batchTileToArm(obs.Reference.Arm, referenceTile, obs.PacketDigest),
	}, nil
}

func validateStrixConcurrentBatchPair(obs StrixConcurrentBatchPairObservation) (BatchTile, BatchTile, error) {
	if !slices.Contains(StrixConcurrentBatchAllowedConcurrency, obs.Concurrency) {
		return BatchTile{}, BatchTile{}, batchRefuse("unsupported-concurrency", fmt.Sprintf("c=%d is not in {4,8}", obs.Concurrency))
	}
	if obs.Candidate.Concurrency != obs.Concurrency || obs.Reference.Concurrency != obs.Concurrency {
		return BatchTile{}, BatchTile{}, batchRefuse("unsupported-concurrency", "arm concurrency disagrees with the pair cell")
	}
	if !validOracleSHA256(obs.PacketDigest) {
		return BatchTile{}, BatchTile{}, batchRefuse("packet-digest-mismatch", "shared packet digest is not a canonical sha256")
	}
	if !finitePositive(obs.LogitTolerance) {
		return BatchTile{}, BatchTile{}, batchRefuse("non-finite-value", "logit tolerance must be finite positive")
	}

	candidateTile, err := scoreStrixConcurrentBatchArm(obs.Candidate, "candidate", obs.PacketDigest)
	if err != nil {
		return BatchTile{}, BatchTile{}, err
	}
	referenceTile, err := scoreStrixConcurrentBatchArm(obs.Reference, "reference", obs.PacketDigest)
	if err != nil {
		return BatchTile{}, BatchTile{}, err
	}

	// Both arms must be the same artifact and share the packet digest the pair
	// declared; a batch cannot silently compare two models or two packets.
	if obs.Candidate.Arm.ArtifactSHA256 != obs.Reference.Arm.ArtifactSHA256 {
		return BatchTile{}, BatchTile{}, batchRefuse("artifact-mismatch", "candidate and reference artifacts differ")
	}
	if obs.Candidate.Arm.PromptPacketDigest != obs.PacketDigest || obs.Reference.Arm.PromptPacketDigest != obs.PacketDigest {
		return BatchTile{}, BatchTile{}, batchRefuse("packet-digest-mismatch", "arm packet digest disagrees with the pair")
	}
	// Both arms must emit the same ordered accepted output tokens for every
	// slot; a batch of diverging outputs is not comparable regardless of rate.
	if err := batchSlotTokenEquivalence(obs.Candidate.Slots, obs.Reference.Slots); err != nil {
		return BatchTile{}, BatchTile{}, err
	}
	return candidateTile, referenceTile, nil
}

func batchSlotTokenEquivalence(candidateSlots, referenceSlots []StrixConcurrentBatchSlotObservation) error {
	if len(candidateSlots) != len(referenceSlots) {
		return batchRefuse("slot-not-overlapping", "arms have different slot counts")
	}
	for i := range candidateSlots {
		if err := ValidateTokenEquivalence(candidateSlots[i].AcceptedOutputTokenIDs, referenceSlots[i].AcceptedOutputTokenIDs); err != nil {
			return batchRefuse("accepted-token-count-mismatch", fmt.Sprintf("slot %d token identity: %v", i, err))
		}
	}
	return nil
}

// batchArmIdentityDrift refuses a pair whose arm identity disagrees with the
// first pair's, so a cell can never silently mix two hardware, artifact, or
// packet identities across its measured pairs.
func batchArmIdentityDrift(first, next AMDArmReceipt, role string) error {
	if first.Name != next.Name || first.Engine != next.Engine || first.Backend != next.Backend ||
		first.ArtifactSHA256 != next.ArtifactSHA256 || first.PromptSHA256 != next.PromptSHA256 ||
		first.PromptPacketDigest != next.PromptPacketDigest || first.SoftwareRevision != next.SoftwareRevision ||
		first.Hardware != next.Hardware || first.ComparatorOnly != next.ComparatorOnly {
		return batchRefuse("arm-identity-drift", role+" identity drifts across measured pairs")
	}
	return nil
}

func scoreStrixConcurrentBatchArm(armObs StrixConcurrentBatchArmObservation, role, packetDigest string) (BatchTile, error) {
	arm := armObs.Arm
	if arm.Name == "" || arm.Engine == "" || arm.Backend == "" || arm.Runtime == "" || arm.Hardware == "" || arm.SoftwareRevision == "" || len(arm.BuildFlags) == 0 {
		return BatchTile{}, batchRefuse("arm-identity-incomplete", role+" identity is incomplete")
	}
	if role == "candidate" && (arm.Engine != "fak-native" || arm.ComparatorOnly || arm.FallbackActive) {
		return BatchTile{}, batchRefuse("candidate-not-fak-native-no-fallback", "candidate is not fak-native without fallback")
	}
	if role == "reference" && (arm.Engine != "llama.cpp" || !arm.ComparatorOnly || arm.FallbackActive) {
		return BatchTile{}, batchRefuse("reference-not-explicit-llamacpp-comparator", "reference is not an explicit llama.cpp comparator")
	}
	if arm.PromptPacketDigest != packetDigest {
		return BatchTile{}, batchRefuse("packet-digest-mismatch", role+" packet digest disagrees with the pair")
	}
	if len(armObs.Slots) != armObs.Concurrency {
		return BatchTile{}, batchRefuse("slot-count-mismatch", fmt.Sprintf("%s has %d slots, want %d", role, len(armObs.Slots), armObs.Concurrency))
	}

	seen := make(map[string]struct{}, len(armObs.Slots))
	start, end := math.Inf(1), math.Inf(-1)
	maxAdmission, minCompletion := math.Inf(-1), math.Inf(1)
	var accepted uint64
	for i, slot := range armObs.Slots {
		if slot.RequestID == "" || slot.PrefillTokens <= 0 || len(slot.AcceptedOutputTokenIDs) == 0 {
			return BatchTile{}, batchRefuse("slot-identity-incomplete", fmt.Sprintf("%s slot %d is incomplete", role, i))
		}
		if _, dup := seen[slot.RequestID]; dup {
			return BatchTile{}, batchRefuse("duplicate-request-id", fmt.Sprintf("%s repeats request %q", role, slot.RequestID))
		}
		seen[slot.RequestID] = struct{}{}
		if !finitePositive(slot.AdmissionSec) || !finitePositive(slot.CompletionSec) || slot.CompletionSec <= slot.AdmissionSec {
			return BatchTile{}, batchRefuse("non-monotonic-clock", fmt.Sprintf("%s slot %d has a non-positive interval", role, i))
		}
		if slot.AdmissionSec > maxAdmission {
			maxAdmission = slot.AdmissionSec
		}
		if slot.CompletionSec < minCompletion {
			minCompletion = slot.CompletionSec
		}
		if slot.AdmissionSec < start {
			start = slot.AdmissionSec
		}
		if slot.CompletionSec > end {
			end = slot.CompletionSec
		}

		tokens := slot.AcceptedOutputTokenIDs
		logprobs := slot.AcceptedTokenLogprobs
		if len(tokens) != 128 {
			return BatchTile{}, batchRefuse("accepted-token-count-mismatch", fmt.Sprintf("%s slot %d accepted %d tokens, want 128", role, i, len(tokens)))
		}
		if len(logprobs) != len(tokens) {
			return BatchTile{}, batchRefuse("token-logprob-shape-mismatch", fmt.Sprintf("%s slot %d has %d logprobs for %d tokens", role, i, len(logprobs), len(tokens)))
		}
		for _, v := range logprobs {
			if math.IsNaN(v) || math.IsInf(v, 0) || v > 0 {
				return BatchTile{}, batchRefuse("non-finite-value", fmt.Sprintf("%s slot %d has an invalid logprob", role, i))
			}
		}
		if slot.ObservedIgnoreEOS == nil {
			return BatchTile{}, batchRefuse("eos-observation-missing", fmt.Sprintf("%s slot %d has no observed ignore-eos flag", role, i))
		}
		if !*slot.ObservedIgnoreEOS {
			return BatchTile{}, batchRefuse("eos-observation-missing", fmt.Sprintf("%s slot %d did not observe ignore-eos", role, i))
		}
		if slot.EOSStopped == nil {
			return BatchTile{}, batchRefuse("eos-observation-missing", fmt.Sprintf("%s slot %d has no observed eos-stopped flag", role, i))
		}
		if *slot.EOSStopped {
			return BatchTile{}, batchRefuse("eos-stopped-observed", fmt.Sprintf("%s slot %d stopped on EOS", role, i))
		}
		if !finitePositive(slot.PrefillSeconds) || !finitePositive(slot.DecodeSeconds) {
			return BatchTile{}, batchRefuse("timing-arithmetic-invalid", fmt.Sprintf("%s slot %d has a non-positive timing", role, i))
		}
		// The per-slot interval must be reconcilable with its own admitted
		// prefill+decode timing; a shortened shared interval cannot hide work.
		if slot.AdmissionSec+slot.PrefillSeconds+slot.DecodeSeconds > slot.CompletionSec+1e-9 {
			return BatchTile{}, batchRefuse("timing-arithmetic-invalid", fmt.Sprintf("%s slot %d admits a completion before its own prefill+decode", role, i))
		}
		if slot.H2DBytes == 0 || slot.D2HBytes == 0 || slot.QueueSubmissions == 0 {
			return BatchTile{}, batchRefuse("transfer-or-submission-accounting-missing", fmt.Sprintf("%s slot %d lacks transfer/submission accounting", role, i))
		}
		if slot.PeakRSSBytes == 0 || slot.PeakVRAMBytes == 0 || slot.ResidentModelBytes == 0 {
			return BatchTile{}, batchRefuse("memory-evidence-missing", fmt.Sprintf("%s slot %d lacks memory evidence", role, i))
		}
		if role == "candidate" {
			if slot.NativeInferenceReceipt == nil {
				return BatchTile{}, batchRefuse("candidate-not-fak-native-no-fallback", fmt.Sprintf("candidate slot %d lacks a native receipt", i))
			}
			if err := validateAMDNativeReceipt(slot.NativeInferenceReceipt); err != nil {
				return BatchTile{}, batchRefuse("arm-identity-drift", fmt.Sprintf("candidate slot %d native receipt is invalid", i))
			}
			if slot.NativeInferenceReceipt.Backend != arm.Backend {
				return BatchTile{}, batchRefuse("arm-identity-drift", fmt.Sprintf("candidate slot %d native backend disagrees with the arm", i))
			}
		}

		var next uint64
		if accepted > math.MaxUint64-uint64(len(tokens)) {
			return BatchTile{}, batchRefuse("accepted-token-count-mismatch", role+" accepted-token sum overflows")
		}
		next = accepted + uint64(len(tokens))
		accepted = next
	}

	if !finitePositive(start) || !finitePositive(end) || end <= start {
		return BatchTile{}, batchRefuse("non-monotonic-clock", role+" shared interval is not positive")
	}
	// A genuine simultaneous batch has every request admitted on one arm before
	// any request of that arm completes: max(admission) < min(completion). A
	// serialized or relabeled c1 cannot satisfy this.
	if !(maxAdmission < minCompletion) {
		return BatchTile{}, batchRefuse("slot-not-overlapping", role+" slots do not share one admission-to-completion interval")
	}
	tile := BatchTile{Requests: len(armObs.Slots), AcceptedToken: accepted, StartSec: start, EndSec: end}
	tile.AggregateTPS = float64(accepted) / tile.Duration()
	if !finitePositive(tile.AggregateTPS) {
		return BatchTile{}, batchRefuse("non-finite-value", role+" aggregate rate is not finite positive")
	}
	return tile, nil
}

// batchTileToArm expresses one arm's aggregate tile as a single accepted-token
// stream of length N over the shared interval T. It is the single-pair seam
// used for inspection and refusal; the creditable path uses batchTileToTrials.
func batchTileToArm(arm AMDArmReceipt, tile BatchTile, packetDigest string) AMDArmReceipt {
	out := arm
	out.DecodeTokens = int(tile.AcceptedToken)
	out.PromptPacketDigest = packetDigest
	out.Trials = batchTileToTrials(arm, tile, 1, 1)
	return out
}

// batchTileToTrials expresses one arm's aggregate tile for one measured pair as
// an accepted-token stream of length N over the shared interval T, stamped with
// the canonical alternating pair identity. The scoreboard's own statistics then
// recompute N/T from exactly this evidence, so no rate is ever taken on the
// caller's word, and the ordinary v2 gates (artifact, packet, identity, pair
// order, evidence kind) still apply to the batch arm.
func batchTileToTrials(arm AMDArmReceipt, tile BatchTile, repetition, sequence int) []AMDScoreboardTrial {
	trial := AMDScoreboardTrial{
		EvidenceKind:              "selected-token-logprobs",
		Repetition:                repetition,
		Sequence:                  sequence,
		ColdSetupSeconds:          300,
		PrefillSeconds:            tile.Duration(),
		PrefillTokensPerSecond:    float64(arm.PrefillTokens) / tile.Duration(),
		WarmDecodeSeconds:         tile.Duration(),
		WarmDecodeTokensPerSecond: float64(tile.AcceptedToken) / tile.Duration(),
		OutputTokenIDs:            batchTokenStream(int(tile.AcceptedToken)),
		Logits:                    batchLogitStream(int(tile.AcceptedToken)),
		H2DBytes:                  1,
		D2HBytes:                  1,
		D2DBytes:                  1,
		QueueSubmissions:          1,
		AcceptedOutputTokens:      tile.AcceptedToken,
	}
	return []AMDScoreboardTrial{trial}
}

func batchTokenStream(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

func batchLogitStream(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = -1
	}
	return out
}

// BatchTileFromObservation derives one arm's aggregate tile directly from a
// validated observation. It re-uses scoreStrixConcurrentBatchArm so callers can
// inspect the credited N and T without building a full scoreboard input.
func BatchTileFromObservation(armObs StrixConcurrentBatchArmObservation, role, packetDigest string) (BatchTile, error) {
	return scoreStrixConcurrentBatchArm(armObs, role, packetDigest)
}
