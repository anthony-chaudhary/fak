package harnesscreationstudy

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

// Schema defines the canonical schema URI for harness-creation study records.
const Schema = "fak.harness-creation-study/v1alpha1"

var safeID = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)
var digestRE = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// Study defines the protocol, reference baseline, and trial run records for a harness-creation evaluation.
type Study struct {
	Schema   string   `json:"schema"`
	ID       string   `json:"id"`
	Protocol Protocol `json:"protocol"`
	Baseline Baseline `json:"baseline"`
	Runs     []Run    `json:"runs"`
}

// Protocol specifies frozen execution limits, assistance policies, and parity constraints.
type Protocol struct {
	Frozen                bool             `json:"frozen"`
	TenMinuteLimitSeconds int              `json:"ten_minute_limit_seconds"`
	AssistancePolicy      string           `json:"assistance_policy"`
	FailuresInDenominator bool             `json:"failures_in_denominator"`
	TaskDigest            string           `json:"task_digest"`
	Parity                MatchedStudySpec `json:"parity"`
}

// MatchedStudySpec configures sample sizes and ratio bounds for counterbalanced paired arm evaluations.
type MatchedStudySpec struct {
	Frozen                bool    `json:"frozen"`
	MinimumPairs          int     `json:"minimum_pairs"`
	MaxMedianElapsedRatio float64 `json:"max_median_elapsed_ratio"`
	CounterbalancedOrder  bool    `json:"counterbalanced_order"`
}

// Baseline captures the runnable, tuned reference implementation against which trials are compared.
type Baseline struct {
	ID       string `json:"id"`
	Runnable bool   `json:"runnable"`
	Tuned    bool   `json:"tuned"`
	Frozen   bool   `json:"frozen"`
	Evidence string `json:"evidence"`
}

// Run records environment parameters, timing, and outcome data for a single participant trial.
type Run struct {
	ID                    string   `json:"id"`
	ParticipantID         string   `json:"participant_id"`
	Track                 string   `json:"track"`
	Arm                   string   `json:"arm,omitempty"`
	PairID                string   `json:"pair_id,omitempty"`
	TaskDigest            string   `json:"task_digest,omitempty"`
	MachineID             string   `json:"machine_id,omitempty"`
	PairOrder             string   `json:"pair_order,omitempty"`
	ArmPosition           int      `json:"arm_position,omitempty"`
	ParticipantClass      string   `json:"participant_class"`
	Independent           bool     `json:"independent"`
	OS                    string   `json:"os,omitempty"`
	CPU                   string   `json:"cpu,omitempty"`
	NetworkState          string   `json:"network_state,omitempty"`
	CacheState            string   `json:"cache_state,omitempty"`
	Outcome               string   `json:"outcome"`
	ElapsedSeconds        float64  `json:"elapsed_seconds"`
	HelpRequests          []string `json:"help_requests,omitempty"`
	Receipt               string   `json:"receipt"`
	SourceReceipt         string   `json:"source_receipt,omitempty"`
	SourceDigest          string   `json:"source_digest,omitempty"`
	IndependentlyAuthored bool     `json:"independently_authored,omitempty"`
	ConformancePassed     bool     `json:"conformance_passed,omitempty"`
}

// Result summarizes validation findings, pass rates, and claim verdicts across study tracks.
type Result struct {
	Schema      string             `json:"schema"`
	StudyID     string             `json:"study_id"`
	BaselineOK  bool               `json:"baseline_ready"`
	Calibration int                `json:"calibration_runs"`
	TenMinute   TrackResult        `json:"ten_minute"`
	Weekend     TrackResult        `json:"weekend"`
	Parity      MatchedStudyResult `json:"parity"`
}

// MatchedStudyResult reports paired arm success counts, timing ratios, and parity claim status.
type MatchedStudyResult struct {
	CompletePairs      int      `json:"complete_pairs"`
	IncompletePairs    int      `json:"incomplete_pairs"`
	FakSuccesses       int      `json:"fak_successes"`
	BaselineSuccesses  int      `json:"baseline_successes"`
	FakFirstPairs      int      `json:"fak_first_pairs"`
	BaselineFirstPairs int      `json:"baseline_first_pairs"`
	MedianElapsedRatio *float64 `json:"median_elapsed_ratio,omitempty"`
	ClaimStatus        string   `json:"claim_status"`
	Reasons            []string `json:"reasons,omitempty"`
}

// TrackResult aggregates eligibility, success rates, and completion times for a study track.
type TrackResult struct {
	EligibleRuns         int      `json:"eligible_runs"`
	DistinctParticipants int      `json:"distinct_participants"`
	Successes            int      `json:"successes"`
	Failures             int      `json:"failures"`
	PassRate             float64  `json:"pass_rate"`
	MedianSuccessSeconds *float64 `json:"median_success_seconds,omitempty"`
	ClaimStatus          string   `json:"claim_status"`
	Reasons              []string `json:"reasons,omitempty"`
	successTimes         []float64
}

// Parse decodes and validates a raw JSON study manifest against schema, privacy, and protocol invariants.
func Parse(raw []byte) (Study, error) {
	var s Study
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return Study{}, err
	}
	if s.Schema != Schema {
		return Study{}, fmt.Errorf("schema must be %q", Schema)
	}
	if !safeID.MatchString(s.ID) {
		return Study{}, errors.New("study id must be a privacy-safe slug")
	}
	if !s.Protocol.Frozen || s.Protocol.TenMinuteLimitSeconds != 600 || s.Protocol.AssistancePolicy != "task-card-and-help-only" || !s.Protocol.FailuresInDenominator {
		return Study{}, errors.New("protocol must freeze the 600-second limit, task-card-and-help-only assistance, and failures-in-denominator rule")
	}
	if !digestRE.MatchString(s.Protocol.TaskDigest) {
		return Study{}, errors.New("protocol.task_digest must be lowercase sha256:<64 hex>")
	}
	if !s.Protocol.Parity.Frozen || s.Protocol.Parity.MinimumPairs < 1 || s.Protocol.Parity.MaxMedianElapsedRatio < 1 || !s.Protocol.Parity.CounterbalancedOrder {
		return Study{}, errors.New("protocol parity question must freeze minimum_pairs >= 1, max_median_elapsed_ratio >= 1, and counterbalanced_order")
	}
	if !safeID.MatchString(s.Baseline.ID) || strings.TrimSpace(s.Baseline.Evidence) == "" {
		return Study{}, errors.New("baseline requires a privacy-safe id and evidence reference")
	}
	seen := map[string]bool{}
	seenPairArms := map[string]bool{}
	pairParticipants := map[string]string{}
	pairOrders := map[string]string{}
	pairEnvelopes := map[string]string{}
	for i, r := range s.Runs {
		if !safeID.MatchString(r.ID) || !safeID.MatchString(r.ParticipantID) {
			return Study{}, fmt.Errorf("run %d requires privacy-safe run and participant slugs", i)
		}
		if r.Arm != "" && r.Arm != "fak" && r.Arm != "baseline" {
			return Study{}, fmt.Errorf("run %q: unknown arm %q", r.ID, r.Arm)
		}
		if (r.Arm == "") != (r.PairID == "") {
			return Study{}, fmt.Errorf("run %q: arm and pair_id must be supplied together", r.ID)
		}
		if r.PairID != "" {
			if r.PairOrder != "fak-first" && r.PairOrder != "baseline-first" {
				return Study{}, fmt.Errorf("run %q: pair_order must be fak-first or baseline-first", r.ID)
			}
			expectedPosition := 2
			if (r.PairOrder == "fak-first" && r.Arm == "fak") || (r.PairOrder == "baseline-first" && r.Arm == "baseline") {
				expectedPosition = 1
			}
			if r.ArmPosition != expectedPosition {
				return Study{}, fmt.Errorf("run %q: arm_position must be %d for %s in %s", r.ID, expectedPosition, r.Arm, r.PairOrder)
			}
		}
		if r.PairID != "" {
			if r.TaskDigest != s.Protocol.TaskDigest {
				return Study{}, fmt.Errorf("run %q: task_digest does not match protocol.task_digest", r.ID)
			}
			if !safeID.MatchString(r.MachineID) || !digestRE.MatchString(r.TaskDigest) || r.OS == "" || r.CPU == "" || r.NetworkState == "" || r.CacheState == "" {
				return Study{}, fmt.Errorf("run %q: complete comparison envelope is required", r.ID)
			}
			envelope := strings.Join([]string{r.TaskDigest, r.MachineID, r.OS, r.CPU, r.NetworkState, r.CacheState}, "\x00")
			if prior, ok := pairEnvelopes[r.PairID]; ok && prior != envelope {
				return Study{}, fmt.Errorf("pair %q comparison envelope differs between arms", r.PairID)
			}
			pairEnvelopes[r.PairID] = envelope
			if participant, ok := pairParticipants[r.PairID]; ok && participant != r.ParticipantID {
				return Study{}, fmt.Errorf("pair %q spans participants %q and %q", r.PairID, participant, r.ParticipantID)
			}
			pairParticipants[r.PairID] = r.ParticipantID
			if order, ok := pairOrders[r.PairID]; ok && order != r.PairOrder {
				return Study{}, fmt.Errorf("pair %q has conflicting order %q and %q", r.PairID, order, r.PairOrder)
			}
			pairOrders[r.PairID] = r.PairOrder
			pairKey := r.PairID + "\x00" + r.Arm
			if seenPairArms[pairKey] {
				return Study{}, fmt.Errorf("duplicate pair arm %q/%q", r.PairID, r.Arm)
			}
			seenPairArms[pairKey] = true
		}
		key := r.ParticipantID + "\x00" + r.Track + "\x00" + r.Arm
		if seen[key] {
			return Study{}, fmt.Errorf("participant %q has duplicate %s/%s attempt", r.ParticipantID, r.Track, r.Arm)
		}
		seen[key] = true
		if r.Track != "ten-minute" && r.Track != "weekend" {
			return Study{}, fmt.Errorf("run %q has invalid track", r.ID)
		}
		if r.ParticipantClass != "unfamiliar-builder" && r.ParticipantClass != "maintainer-calibration" {
			return Study{}, fmt.Errorf("run %q has invalid participant_class", r.ID)
		}
		if r.Outcome != "success" && r.Outcome != "failure" && r.Outcome != "timeout" {
			return Study{}, fmt.Errorf("run %q has invalid outcome", r.ID)
		}
		if math.IsNaN(r.ElapsedSeconds) || math.IsInf(r.ElapsedSeconds, 0) || r.ElapsedSeconds < 0 || strings.TrimSpace(r.Receipt) == "" {
			return Study{}, fmt.Errorf("run %q requires finite non-negative elapsed_seconds and receipt", r.ID)
		}
	}
	return s, nil
}

// Evaluate aggregates run outcomes and computes claim statuses across study tracks and paired arms.
func Evaluate(s Study) Result {
	baselineOK := s.Baseline.Runnable && s.Baseline.Tuned && s.Baseline.Frozen
	result := Result{Schema: Schema, StudyID: s.ID, BaselineOK: baselineOK}
	for _, r := range s.Runs {
		if r.ParticipantClass == "maintainer-calibration" || !r.Independent {
			result.Calibration++
			continue
		}
		if r.Track == "ten-minute" && r.Arm == "baseline" {
			continue
		}
		if r.Track == "ten-minute" {
			addTenMinute(&result.TenMinute, r, s.Protocol.TenMinuteLimitSeconds)
		} else {
			addWeekend(&result.Weekend, r)
		}
	}
	finish(&result.TenMinute)
	finish(&result.Weekend)
	result.TenMinute.Reasons = claimReasons(result.TenMinute, baselineOK, 2)
	result.Weekend.Reasons = claimReasons(result.Weekend, baselineOK, 1)
	if len(result.TenMinute.Reasons) == 0 {
		result.TenMinute.ClaimStatus = "supported"
	} else {
		result.TenMinute.ClaimStatus = "not_yet"
	}
	if len(result.Weekend.Reasons) == 0 {
		result.Weekend.ClaimStatus = "supported"
	} else {
		result.Weekend.ClaimStatus = "not_yet"
	}
	result.Parity = evaluateMatchedStudy(s)
	return result
}

func addTenMinute(t *TrackResult, r Run, limit int) {
	t.EligibleRuns++
	t.DistinctParticipants++
	if r.Outcome == "success" && r.ElapsedSeconds <= float64(limit) {
		t.Successes++
		addSuccessTime(t, r.ElapsedSeconds)
	} else {
		t.Failures++
	}
}

func addWeekend(t *TrackResult, r Run) {
	t.EligibleRuns++
	t.DistinctParticipants++
	if r.Outcome == "success" && r.IndependentlyAuthored && r.ConformancePassed {
		t.Successes++
		addSuccessTime(t, r.ElapsedSeconds)
	} else {
		t.Failures++
	}
}

func addSuccessTime(t *TrackResult, seconds float64) {
	t.successTimes = append(t.successTimes, seconds)
}

func finish(t *TrackResult) {
	if t.EligibleRuns > 0 {
		t.PassRate = float64(t.Successes) / float64(t.EligibleRuns)
	}
	values := t.successTimes
	t.successTimes = nil
	if len(values) == 0 {
		return
	}
	sort.Float64s(values)
	median := values[len(values)/2]
	if len(values)%2 == 0 {
		median = (values[len(values)/2-1] + median) / 2
	}
	t.MedianSuccessSeconds = &median
}

func claimReasons(t TrackResult, baselineOK bool, required int) []string {
	var reasons []string
	if !baselineOK {
		reasons = append(reasons, "tuned runnable baseline is not frozen")
	}
	if t.DistinctParticipants < required {
		reasons = append(reasons, fmt.Sprintf("need at least %d eligible independent participant(s)", required))
	}
	if t.Successes < required {
		reasons = append(reasons, fmt.Sprintf("need at least %d successful eligible run(s)", required))
	}
	return reasons
}

type parityPair struct {
	fak      *Run
	baseline *Run
}

func evaluateMatchedStudy(s Study) MatchedStudyResult {
	out := MatchedStudyResult{ClaimStatus: "not_yet"}
	pairs := map[string]*parityPair{}
	for i := range s.Runs {
		run := &s.Runs[i]
		if run.Track != "ten-minute" || run.PairID == "" || run.ParticipantClass != "unfamiliar-builder" || !run.Independent {
			continue
		}
		pair := pairs[run.PairID]
		if pair == nil {
			pair = &parityPair{}
			pairs[run.PairID] = pair
		}
		if run.Arm == "fak" {
			pair.fak = run
		} else if run.Arm == "baseline" {
			pair.baseline = run
		}
	}
	ratios := []float64{}
	for _, pair := range pairs {
		if pair.fak == nil || pair.baseline == nil {
			out.IncompletePairs++
			continue
		}
		out.CompletePairs++
		if pair.fak.PairOrder == "fak-first" {
			out.FakFirstPairs++
		} else if pair.fak.PairOrder == "baseline-first" {
			out.BaselineFirstPairs++
		}
		fakOK := pair.fak.Outcome == "success" && pair.fak.ElapsedSeconds <= float64(s.Protocol.TenMinuteLimitSeconds)
		baselineOK := pair.baseline.Outcome == "success" && pair.baseline.ElapsedSeconds <= float64(s.Protocol.TenMinuteLimitSeconds)
		if fakOK {
			out.FakSuccesses++
		}
		if baselineOK {
			out.BaselineSuccesses++
		}
		if fakOK && baselineOK && pair.baseline.ElapsedSeconds > 0 {
			ratios = append(ratios, pair.fak.ElapsedSeconds/pair.baseline.ElapsedSeconds)
		}
	}
	if len(ratios) > 0 {
		sort.Float64s(ratios)
		m := ratios[len(ratios)/2]
		if len(ratios)%2 == 0 {
			m = (ratios[len(ratios)/2-1] + ratios[len(ratios)/2]) / 2
		}
		out.MedianElapsedRatio = &m
	}
	minimum := s.Protocol.Parity.MinimumPairs
	if out.CompletePairs < minimum {
		out.Reasons = append(out.Reasons, fmt.Sprintf("need at least %d complete independent pair(s)", minimum))
		return out
	}
	if out.FakSuccesses < out.BaselineSuccesses {
		out.ClaimStatus = "refuted"
		out.Reasons = append(out.Reasons, "fak has fewer successful paired arms than baseline")
		return out
	}
	if out.IncompletePairs > 0 {
		out.Reasons = append(out.Reasons, fmt.Sprintf("%d eligible independent pair(s) still incomplete", out.IncompletePairs))
		return out
	}
	if len(ratios) < minimum {
		out.Reasons = append(out.Reasons, fmt.Sprintf("need at least %d pair(s) where both arms succeed", minimum))
		return out
	}
	if *out.MedianElapsedRatio > s.Protocol.Parity.MaxMedianElapsedRatio {
		out.ClaimStatus = "refuted"
		out.Reasons = append(out.Reasons, fmt.Sprintf("median elapsed ratio %.3f exceeds frozen bound %.3f", *out.MedianElapsedRatio, s.Protocol.Parity.MaxMedianElapsedRatio))
		return out
	}
	if out.FakFirstPairs == 0 || out.BaselineFirstPairs == 0 {
		out.Reasons = append(out.Reasons, "need complete pairs in both fak-first and baseline-first order")
		return out
	}
	out.ClaimStatus = "supported"
	return out
}
