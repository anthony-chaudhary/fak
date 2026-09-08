package ultracodebench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

const SafeAccelerationSchema = "fak-safe-acceleration-campaign/1"

type SafeAccelerationVerdict string

const (
	SafeAccelerationAbstain SafeAccelerationVerdict = "ABSTAIN"
	SafeAccelerationNoGain  SafeAccelerationVerdict = "NO_GAIN"
	SafeAccelerationGain10X SafeAccelerationVerdict = "GAIN_10X"
)

type SafeAccelerationReasonCode string

type SafeAccelerationReason struct {
	Code SafeAccelerationReasonCode `json:"code"`
	Pair int                        `json:"pair,omitempty"`
}

type SafeAccelerationTask struct {
	ID    string `json:"id"`
	Scope string `json:"scope"`
}

type SafeAccelerationIdentity struct {
	Source              string `json:"source"`
	ArtifactDigest      string `json:"artifact_digest"`
	ConfigurationDigest string `json:"configuration_digest"`
	WorkloadDigest      string `json:"workload_digest"`
}

type SafeAccelerationEnvelope struct {
	AllowHigherCost            bool `json:"allow_higher_cost"`
	AllowHigherJoules          bool `json:"allow_higher_joules"`
	AllowHigherOperatorMinutes bool `json:"allow_higher_operator_minutes"`
}

type SafeAccelerationViolations struct {
	Policy     int `json:"policy"`
	Secret     int `json:"secret"`
	Boundary   int `json:"boundary"`
	Lease      int `json:"lease"`
	Scope      int `json:"scope"`
	Provenance int `json:"provenance"`
}

type SafeAccelerationArm struct {
	TaskIDs                      []string                   `json:"task_ids"`
	FullWallMS                   int64                      `json:"full_wall_ms"`
	PhaseMS                      map[string]int64           `json:"phase_ms"`
	AcceptedUnits                int                        `json:"accepted_units"`
	TaskLatencyMS                []int64                    `json:"task_latency_ms"`
	EvidenceKind                 string                     `json:"evidence_kind"`
	ReceiptDigest                string                     `json:"receipt_digest"`
	AcceptanceWitnessDigest      string                     `json:"acceptance_witness_digest"`
	AcceptanceWitnessIndependent bool                       `json:"acceptance_witness_independent"`
	Engine                       string                     `json:"engine"`
	Backend                      string                     `json:"backend"`
	Device                       string                     `json:"device"`
	Native                       bool                       `json:"native"`
	FallbackCount                int                        `json:"fallback_count"`
	Violations                   SafeAccelerationViolations `json:"violations"`
	CostUSD                      float64                    `json:"cost_usd"`
	Joules                       float64                    `json:"joules"`
	OperatorMinutes              float64                    `json:"operator_minutes"`
	ResourceMetricsComplete      bool                       `json:"resource_metrics_complete"`
}

type SafeAccelerationPair struct {
	Baseline  SafeAccelerationArm `json:"baseline"`
	Candidate SafeAccelerationArm `json:"candidate"`
}

type safeAccelerationEvidenceRef struct {
	Pair                    int    `json:"pair"`
	Arm                     string `json:"arm"`
	ReceiptDigest           string `json:"receipt_digest"`
	AcceptanceWitnessDigest string `json:"acceptance_witness_digest"`
}

type safeAccelerationReceipt struct {
	Schema           string                   `json:"schema"`
	Pair             int                      `json:"pair"`
	Arm              string                   `json:"arm"`
	RunID            string                   `json:"run_id"`
	Issuer           string                   `json:"issuer"`
	ProvenanceDigest string                   `json:"provenance_digest"`
	PublicRevision   string                   `json:"public_revision"`
	PrivateRevision  string                   `json:"private_revision"`
	Identity         SafeAccelerationIdentity `json:"identity"`
	Measurements     SafeAccelerationArm      `json:"measurements"`
}

type safeAccelerationWitness struct {
	Schema           string                     `json:"schema"`
	Pair             int                        `json:"pair"`
	Arm              string                     `json:"arm"`
	RunID            string                     `json:"run_id"`
	Issuer           string                     `json:"issuer"`
	ProvenanceDigest string                     `json:"provenance_digest"`
	ReceiptDigest    string                     `json:"receipt_digest"`
	AcceptedTaskIDs  []string                   `json:"accepted_task_ids"`
	Violations       SafeAccelerationViolations `json:"violations"`
}

type safeAccelerationResolvedMaterial struct {
	Receipt []byte
	Witness []byte
}

// This capability has no exported construction path. Its sole constructor
// parses and re-hashes resolved receipt and witness material.
type safeAccelerationVerificationCapability struct {
	campaignDigest string
	bindings       map[string]string
	seal           string
}

type SafeAccelerationCampaign struct {
	Schema          string                   `json:"schema"`
	PublicRevision  string                   `json:"public_revision"`
	PrivateRevision string                   `json:"private_revision"`
	Baseline        SafeAccelerationIdentity `json:"baseline_identity"`
	Candidate       SafeAccelerationIdentity `json:"candidate_identity"`
	Tasks           []SafeAccelerationTask   `json:"tasks"`
	Pairs           []SafeAccelerationPair   `json:"pairs"`
	Envelope        SafeAccelerationEnvelope `json:"envelope"`
}

type SafeAccelerationReport struct {
	Schema                string                   `json:"schema"`
	Verdict               SafeAccelerationVerdict  `json:"verdict"`
	Reasons               []SafeAccelerationReason `json:"reasons,omitempty"`
	Pairs                 int                      `json:"pairs"`
	Tasks                 int                      `json:"tasks"`
	BaselineAcceptedRate  float64                  `json:"baseline_accepted_rate"`
	CandidateAcceptedRate float64                  `json:"candidate_accepted_rate"`
	BaselineP90LatencyMS  int64                    `json:"baseline_p90_latency_ms"`
	CandidateP90LatencyMS int64                    `json:"candidate_p90_latency_ms"`
	PairedRateRatio       float64                  `json:"paired_rate_ratio"`
	RateRatioLower95      float64                  `json:"rate_ratio_lower_95"`
	ConfidenceMethod      string                   `json:"confidence_method"`
}

var safeAccelerationPhases = [...]string{
	"admission", "queueing", "execution", "retries", "verification",
	"fan_in", "stragglers", "recovery", "terminal_failures",
}

// EvaluateSafeAcceleration recomputes every headline metric from the observed
// paired campaign. Invalid or incomplete evidence always abstains; complete
// evidence below the ten-times bar is a measured NO_GAIN.
func EvaluateSafeAcceleration(c SafeAccelerationCampaign, capability *safeAccelerationVerificationCapability) SafeAccelerationReport {
	r := SafeAccelerationReport{
		Schema:           SafeAccelerationSchema,
		Verdict:          SafeAccelerationAbstain,
		Pairs:            len(c.Pairs),
		Tasks:            len(c.Tasks),
		ConfidenceMethod: "paired_log_rate_ratio_t_one_sided_95",
	}
	add := func(code SafeAccelerationReasonCode, pair int) {
		r.Reasons = append(r.Reasons, SafeAccelerationReason{Code: code, Pair: pair})
	}

	if c.Schema != SafeAccelerationSchema {
		add("schema_unsupported", 0)
	}
	expectedIDs, tasksOK := safeAccelerationTaskIDs(c.Tasks)
	if len(c.Tasks) < 20 {
		add("task_count_below_20", 0)
	}
	if !tasksOK {
		add("tasks_not_unique_s0_s1", 0)
	}
	if len(c.Pairs) < 5 {
		add("matched_pair_count_below_5", 0)
	}
	if !safeAccelerationGitRevision(c.PublicRevision) {
		add("public_revision_not_exact", 0)
	}
	if !safeAccelerationGitRevision(c.PrivateRevision) {
		add("private_revision_not_exact", 0)
	}
	if !safeAccelerationIdentityComplete(c.Baseline) {
		add("baseline_identity_incomplete", 0)
	}
	if !safeAccelerationIdentityComplete(c.Candidate) {
		add("candidate_identity_incomplete", 0)
	}
	if c.Baseline == c.Candidate {
		add("candidate_identity_not_distinct", 0)
	}
	if c.Baseline.WorkloadDigest != c.Candidate.WorkloadDigest {
		add("workload_digest_mismatch", 0)
	}

	var baselineAccepted, candidateAccepted, totalUnits int
	var baselineLatencies, candidateLatencies []int64
	ratios := make([]float64, 0, len(c.Pairs))
	for i := range c.Pairs {
		pairNumber := i + 1
		p := c.Pairs[i]
		validBaseline := safeAccelerationValidateArm(p.Baseline, expectedIDs, pairNumber, "baseline", add)
		validCandidate := safeAccelerationValidateArm(p.Candidate, expectedIDs, pairNumber, "candidate", add)
		if !validBaseline || !validCandidate {
			continue
		}
		baselineAccepted += p.Baseline.AcceptedUnits
		candidateAccepted += p.Candidate.AcceptedUnits
		totalUnits += len(expectedIDs)
		baselineLatencies = append(baselineLatencies, p.Baseline.TaskLatencyMS...)
		candidateLatencies = append(candidateLatencies, p.Candidate.TaskLatencyMS...)
		baselineRate := float64(p.Baseline.AcceptedUnits) / float64(p.Baseline.FullWallMS)
		candidateRate := float64(p.Candidate.AcceptedUnits) / float64(p.Candidate.FullWallMS)
		if baselineRate <= 0 || candidateRate <= 0 {
			add("accepted_unit_rate_not_positive", pairNumber)
			continue
		}
		ratios = append(ratios, candidateRate/baselineRate)
		if !c.Envelope.AllowHigherCost && p.Candidate.CostUSD > p.Baseline.CostUSD {
			add("candidate_cost_regression", pairNumber)
		}
		if !c.Envelope.AllowHigherJoules && p.Candidate.Joules > p.Baseline.Joules {
			add("candidate_joules_regression", pairNumber)
		}
		if !c.Envelope.AllowHigherOperatorMinutes && p.Candidate.OperatorMinutes > p.Baseline.OperatorMinutes {
			add("candidate_operator_minutes_regression", pairNumber)
		}
	}
	evidence := safeAccelerationEvidence(c)
	if !safeAccelerationUniqueEvidence(evidence) {
		add("evidence_digest_replay", 0)
	}
	safeAccelerationValidateCapability(c, evidence, capability, add)
	if totalUnits > 0 {
		baselineAcceptance := float64(baselineAccepted) / float64(totalUnits)
		candidateAcceptance := float64(candidateAccepted) / float64(totalUnits)
		r.BaselineAcceptedRate = safeAccelerationRound(baselineAcceptance)
		r.CandidateAcceptedRate = safeAccelerationRound(candidateAcceptance)
		if candidateAcceptance < baselineAcceptance {
			add("candidate_acceptance_below_baseline", 0)
		}
		if candidateAcceptance < 0.80 {
			add("candidate_acceptance_below_80_percent", 0)
		}
	}
	r.BaselineP90LatencyMS = safeAccelerationP90(baselineLatencies)
	r.CandidateP90LatencyMS = safeAccelerationP90(candidateLatencies)
	if len(baselineLatencies) > 0 && r.CandidateP90LatencyMS > r.BaselineP90LatencyMS {
		add("candidate_p90_latency_regression", 0)
	}

	if len(r.Reasons) > 0 {
		return r
	}
	pairedRateRatio, lower95 := safeAccelerationPairedLower95(ratios)
	r.PairedRateRatio, r.RateRatioLower95 = safeAccelerationRound(pairedRateRatio), safeAccelerationRound(lower95)
	if safeAccelerationVerifiedVerdict(lower95) == SafeAccelerationGain10X {
		r.Verdict = SafeAccelerationGain10X
		return r
	}
	r.Verdict = SafeAccelerationNoGain
	r.Reasons = []SafeAccelerationReason{{Code: "paired_rate_ratio_lower_95_below_10"}}
	return r
}

func safeAccelerationEvidence(c SafeAccelerationCampaign) []safeAccelerationEvidenceRef {
	evidence := make([]safeAccelerationEvidenceRef, 0, len(c.Pairs)*2)
	for i, pair := range c.Pairs {
		evidence = append(evidence,
			safeAccelerationEvidenceRef{Pair: i + 1, Arm: "baseline", ReceiptDigest: pair.Baseline.ReceiptDigest, AcceptanceWitnessDigest: pair.Baseline.AcceptanceWitnessDigest},
			safeAccelerationEvidenceRef{Pair: i + 1, Arm: "candidate", ReceiptDigest: pair.Candidate.ReceiptDigest, AcceptanceWitnessDigest: pair.Candidate.AcceptanceWitnessDigest},
		)
	}
	return evidence
}

func safeAccelerationUniqueEvidence(evidence []safeAccelerationEvidenceRef) bool {
	seen := make(map[string]struct{}, len(evidence)*2)
	for _, item := range evidence {
		for _, digest := range []string{item.ReceiptDigest, item.AcceptanceWitnessDigest} {
			if _, exists := seen[digest]; exists {
				return false
			}
			seen[digest] = struct{}{}
		}
	}
	return true
}

func safeAccelerationValidateCapability(c SafeAccelerationCampaign, evidence []safeAccelerationEvidenceRef, capability *safeAccelerationVerificationCapability, add func(SafeAccelerationReasonCode, int)) {
	if capability == nil {
		add("authoritative_verification_missing", 0)
		return
	}
	digest, err := safeAccelerationCampaignDigest(c)
	if err != nil || capability.campaignDigest != digest {
		add("authoritative_campaign_binding_mismatch", 0)
		return
	}
	if len(capability.bindings) != len(evidence) || capability.seal != safeAccelerationCapabilitySeal(capability.campaignDigest, capability.bindings) {
		add("authoritative_capability_invalid", 0)
		return
	}
	for _, item := range evidence {
		if !safeAccelerationDigest(capability.bindings[safeAccelerationEvidenceKey(item.Pair, item.Arm)]) {
			add("authoritative_capability_invalid", item.Pair)
		}
	}
}

func safeAccelerationEvidenceKey(pair int, arm string) string {
	return strconv.Itoa(pair) + "/" + arm
}

func safeAccelerationCampaignDigest(c SafeAccelerationCampaign) (string, error) {
	canonical, err := json.Marshal(c) // struct order and sorted map keys are deterministic
	if err != nil {
		return "", err
	}
	return safeAccelerationHash(canonical), nil
}

func safeAccelerationHash(material []byte) string {
	sum := sha256.Sum256(material)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func safeAccelerationCapabilitySeal(campaignDigest string, bindings map[string]string) string {
	keys := make([]string, 0, len(bindings))
	for key := range bindings {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	fields := []string{"fak-safe-acceleration-capability/1", campaignDigest}
	for _, key := range keys {
		fields = append(fields, key, bindings[key])
	}
	return safeAccelerationHash([]byte(strings.Join(fields, "\x00")))
}

func safeAccelerationMaterialBinding(campaignDigest string, evidence safeAccelerationEvidenceRef, runID string) string {
	return safeAccelerationHash([]byte(strings.Join([]string{
		"fak-safe-acceleration-run-binding/1", campaignDigest, strconv.Itoa(evidence.Pair), evidence.Arm,
		evidence.ReceiptDigest, evidence.AcceptanceWitnessDigest, runID,
	}, "\x00")))
}

// safeAccelerationAuthorizeResolved is deliberately package-private. It is the
// sole capability constructor and rederives campaign measurements from parsed,
// content-addressed receipt and witness material.
func safeAccelerationAuthorizeResolved(c SafeAccelerationCampaign, materials []safeAccelerationResolvedMaterial) (*safeAccelerationVerificationCapability, error) {
	campaignDigest, err := safeAccelerationCampaignDigest(c)
	if err != nil {
		return nil, fmt.Errorf("canonical campaign: %w", err)
	}
	evidence := safeAccelerationEvidence(c)
	if len(materials) != len(evidence) || !safeAccelerationUniqueEvidence(evidence) {
		return nil, fmt.Errorf("resolved material set mismatch or replay")
	}
	seen := make(map[string]struct{}, len(evidence)*3)
	bindings := make(map[string]string, len(evidence))
	for i, item := range evidence {
		if safeAccelerationHash(materials[i].Receipt) != item.ReceiptDigest || safeAccelerationHash(materials[i].Witness) != item.AcceptanceWitnessDigest {
			return nil, fmt.Errorf("pair %d %s: resolved evidence hash mismatch", item.Pair, item.Arm)
		}
		var receipt safeAccelerationReceipt
		var witness safeAccelerationWitness
		if err := safeAccelerationDecode(materials[i].Receipt, &receipt); err != nil {
			return nil, fmt.Errorf("pair %d %s receipt: %w", item.Pair, item.Arm, err)
		}
		if err := safeAccelerationDecode(materials[i].Witness, &witness); err != nil {
			return nil, fmt.Errorf("pair %d %s witness: %w", item.Pair, item.Arm, err)
		}
		arm, identity := safeAccelerationArmAndIdentity(c, item.Pair, item.Arm)
		if receipt.Schema != "fak-safe-acceleration-receipt/1" || receipt.Pair != item.Pair || receipt.Arm != item.Arm ||
			receipt.PublicRevision != c.PublicRevision || receipt.PrivateRevision != c.PrivateRevision || receipt.Identity != identity ||
			!safeAccelerationCanonicalEqual(receipt.Measurements, safeAccelerationMeasurements(arm)) {
			return nil, fmt.Errorf("pair %d %s: receipt measurements or identity mismatch", item.Pair, item.Arm)
		}
		if witness.Schema != "fak-safe-acceleration-witness/1" || witness.Pair != item.Pair || witness.Arm != item.Arm ||
			witness.RunID != receipt.RunID || witness.ReceiptDigest != item.ReceiptDigest || witness.Violations != arm.Violations ||
			!safeAccelerationAcceptedTasks(witness.AcceptedTaskIDs, arm.TaskIDs, arm.AcceptedUnits) {
			return nil, fmt.Errorf("pair %d %s: witness outcome mismatch", item.Pair, item.Arm)
		}
		if receipt.RunID == "" || receipt.Issuer == "" || witness.Issuer == "" || receipt.Issuer == witness.Issuer ||
			!safeAccelerationDigest(receipt.ProvenanceDigest) || !safeAccelerationDigest(witness.ProvenanceDigest) ||
			receipt.ProvenanceDigest == witness.ProvenanceDigest {
			return nil, fmt.Errorf("pair %d %s: issuer or provenance is not independent", item.Pair, item.Arm)
		}
		for _, identity := range []string{item.ReceiptDigest, item.AcceptanceWitnessDigest, receipt.RunID} {
			if _, duplicate := seen[identity]; duplicate {
				return nil, fmt.Errorf("pair %d %s: evidence or run identity replay", item.Pair, item.Arm)
			}
			seen[identity] = struct{}{}
		}
		bindings[safeAccelerationEvidenceKey(item.Pair, item.Arm)] = safeAccelerationMaterialBinding(campaignDigest, item, receipt.RunID)
	}
	capability := &safeAccelerationVerificationCapability{campaignDigest: campaignDigest, bindings: bindings}
	capability.seal = safeAccelerationCapabilitySeal(campaignDigest, bindings)
	return capability, nil
}

func safeAccelerationDecode(material []byte, value any) error {
	decoder := json.NewDecoder(bytes.NewReader(material))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return fmt.Errorf("trailing JSON: %w", err)
	}
	return nil
}

func safeAccelerationArmAndIdentity(c SafeAccelerationCampaign, pair int, arm string) (SafeAccelerationArm, SafeAccelerationIdentity) {
	if arm == "baseline" {
		return c.Pairs[pair-1].Baseline, c.Baseline
	}
	return c.Pairs[pair-1].Candidate, c.Candidate
}

func safeAccelerationMeasurements(arm SafeAccelerationArm) SafeAccelerationArm {
	arm.ReceiptDigest = ""
	arm.AcceptanceWitnessDigest = ""
	arm.AcceptanceWitnessIndependent = false
	return arm
}

func safeAccelerationCanonicalEqual(a, b any) bool {
	left, leftErr := json.Marshal(a)
	right, rightErr := json.Marshal(b)
	return leftErr == nil && rightErr == nil && bytes.Equal(left, right)
}

func safeAccelerationAcceptedTasks(accepted, all []string, count int) bool {
	if len(accepted) != count {
		return false
	}
	available := make(map[string]struct{}, len(all))
	for _, task := range all {
		available[task] = struct{}{}
	}
	seen := make(map[string]struct{}, len(accepted))
	for _, task := range accepted {
		if _, ok := available[task]; !ok {
			return false
		}
		if _, duplicate := seen[task]; duplicate {
			return false
		}
		seen[task] = struct{}{}
	}
	return true
}

func safeAccelerationVerifiedVerdict(lower95 float64) SafeAccelerationVerdict {
	if lower95 >= 10 {
		return SafeAccelerationGain10X
	}
	return SafeAccelerationNoGain
}

func safeAccelerationTaskIDs(tasks []SafeAccelerationTask) ([]string, bool) {
	ids := make([]string, len(tasks))
	seen := make(map[string]struct{}, len(tasks))
	ok := true
	for i, task := range tasks {
		ids[i] = task.ID
		if task.ID == "" || task.Scope != "S0" && task.Scope != "S1" {
			ok = false
		}
		if _, exists := seen[task.ID]; exists {
			ok = false
		}
		seen[task.ID] = struct{}{}
	}
	return ids, ok
}

func safeAccelerationGitRevision(revision string) bool {
	if len(revision) != 40 || strings.TrimSpace(revision) != revision {
		return false
	}
	_, err := hex.DecodeString(revision)
	return err == nil
}

func safeAccelerationDigest(digest string) bool {
	const prefix = "sha256:"
	if !strings.HasPrefix(digest, prefix) || len(digest) != len(prefix)+64 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(digest, prefix))
	return err == nil
}

func safeAccelerationIdentityComplete(identity SafeAccelerationIdentity) bool {
	return strings.TrimSpace(identity.Source) != "" &&
		safeAccelerationDigest(identity.ArtifactDigest) &&
		safeAccelerationDigest(identity.ConfigurationDigest) &&
		safeAccelerationDigest(identity.WorkloadDigest)
}

func safeAccelerationValidateArm(arm SafeAccelerationArm, expectedIDs []string, pair int, label string, add func(SafeAccelerationReasonCode, int)) bool {
	before := 0
	addArm := func(code string) { before++; add(SafeAccelerationReasonCode(label+"_"+code), pair) }
	if !safeAccelerationSameStrings(arm.TaskIDs, expectedIDs) {
		addArm("task_ids_not_identical")
	}
	if arm.FullWallMS <= 0 {
		addArm("full_wall_not_positive")
	}
	if !safeAccelerationCompletePhases(arm.PhaseMS, arm.FullWallMS) {
		addArm("phase_accounting_incomplete")
	}
	if arm.AcceptedUnits < 0 || arm.AcceptedUnits > len(expectedIDs) {
		addArm("accepted_units_invalid")
	}
	if len(arm.TaskLatencyMS) != len(expectedIDs) || safeAccelerationHasNonPositive(arm.TaskLatencyMS) {
		addArm("task_latency_incomplete")
	}
	if arm.EvidenceKind != "observed" {
		addArm("evidence_not_observed")
	}
	if !safeAccelerationDigest(arm.ReceiptDigest) {
		addArm("receipt_digest_missing")
	}
	if !arm.AcceptanceWitnessIndependent || !safeAccelerationDigest(arm.AcceptanceWitnessDigest) || arm.AcceptanceWitnessDigest == arm.ReceiptDigest {
		addArm("acceptance_witness_not_independent")
	}
	if strings.TrimSpace(arm.Engine) == "" {
		addArm("engine_identity_missing")
	}
	if strings.TrimSpace(arm.Backend) == "" {
		addArm("backend_identity_missing")
	}
	if strings.TrimSpace(arm.Device) == "" {
		addArm("device_identity_missing")
	}
	if !arm.Native || arm.Engine != "fak-native" || arm.FallbackCount != 0 {
		addArm("native_engine_or_fallback_invalid")
	}
	for _, violation := range []struct {
		name  string
		count int
	}{
		{"policy", arm.Violations.Policy}, {"secret", arm.Violations.Secret},
		{"boundary", arm.Violations.Boundary}, {"lease", arm.Violations.Lease},
		{"scope", arm.Violations.Scope}, {"provenance", arm.Violations.Provenance},
	} {
		if violation.count != 0 {
			addArm("violation_" + violation.name)
		}
	}
	if !arm.ResourceMetricsComplete || !safeAccelerationFiniteNonNegative(arm.CostUSD) ||
		!safeAccelerationFiniteNonNegative(arm.Joules) || !safeAccelerationFiniteNonNegative(arm.OperatorMinutes) {
		addArm("resource_metrics_incomplete")
	}
	return before == 0
}

func safeAccelerationCompletePhases(phases map[string]int64, fullWallMS int64) bool {
	if len(phases) != len(safeAccelerationPhases) {
		return false
	}
	var sum int64
	for _, name := range safeAccelerationPhases {
		value, ok := phases[name]
		if !ok || value < 0 || value > fullWallMS-sum {
			return false
		}
		sum += value
	}
	return sum == fullWallMS
}

func safeAccelerationSameStrings(a, b []string) bool {
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

func safeAccelerationHasNonPositive(values []int64) bool {
	for _, value := range values {
		if value <= 0 {
			return true
		}
	}
	return false
}

func safeAccelerationFiniteNonNegative(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0
}

func safeAccelerationP90(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	copyOfValues := append([]int64(nil), values...)
	sort.Slice(copyOfValues, func(i, j int) bool { return copyOfValues[i] < copyOfValues[j] })
	index := int(math.Ceil(0.90*float64(len(copyOfValues)))) - 1
	return copyOfValues[index]
}

func safeAccelerationPairedLower95(ratios []float64) (float64, float64) {
	logs := make([]float64, len(ratios))
	var mean float64
	for i, value := range ratios {
		logs[i] = math.Log(value)
		mean += logs[i]
	}
	mean /= float64(len(logs))
	var squared float64
	for _, value := range logs {
		delta := value - mean
		squared += delta * delta
	}
	standardError := 0.0
	if len(logs) > 1 {
		standardError = math.Sqrt(squared/float64(len(logs)-1)) / math.Sqrt(float64(len(logs)))
	}
	lower := math.Exp(mean - safeAccelerationT95(len(logs)-1)*standardError)
	return math.Exp(mean), lower
}

func safeAccelerationT95(df int) float64 {
	// One-sided 95% Student-t critical values. Campaigns beyond the table use
	// the df=30 value, which is conservative relative to the asymptote.
	values := [...]float64{0, 0, 0, 0, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833,
		1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729,
		1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697}
	if df >= 1 && df < len(values) {
		return values[df]
	}
	return 1.697
}

func safeAccelerationRound(value float64) float64 {
	return math.Round(value*1e6) / 1e6
}
