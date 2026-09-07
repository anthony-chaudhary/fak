package armtracking

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// SchemaVersion defines the format identifier for shifting baselines and arms data.
const SchemaVersion = "fak-shifting-baselines/1"

// ArmKind classifies the role of an experimental arm.
type ArmKind string

const (
	ArmKindBaseline  ArmKind = "baseline"  // Pinned or shifting control / incumbent / reference
	ArmKindAblation  ArmKind = "ablation"  // Component or lever isolation (e.g. vDSO, normgate)
	ArmKindSweep     ArmKind = "sweep"     // Grid or hyperparameter sweep (e.g. batch size, context tokens)
	ArmKindCandidate ArmKind = "candidate" // New optimization or candidate implementation
)

// Validate checks whether an ArmKind is one of the recognized kinds.
func (k ArmKind) Validate() error {
	switch k {
	case ArmKindBaseline, ArmKindAblation, ArmKindSweep, ArmKindCandidate:
		return nil
	default:
		return fmt.Errorf("armtracking: unknown arm kind %q", k)
	}
}

// OptimizationDirection specifies whether a larger or smaller metric value is preferable.
type OptimizationDirection string

const (
	HigherIsBetter OptimizationDirection = "higher_is_better" // e.g., throughput (tok/s), cache hit rate, accuracy
	LowerIsBetter  OptimizationDirection = "lower_is_better"  // e.g., latency (ms/us), memory bytes, cost, errors
)

// Validate checks whether an OptimizationDirection is recognized.
func (d OptimizationDirection) Validate() error {
	switch d {
	case HigherIsBetter, LowerIsBetter:
		return nil
	default:
		return fmt.Errorf("armtracking: unknown optimization direction %q", d)
	}
}

// ArmMetadata records associated configuration, hardware, and provenance for an arm.
type ArmMetadata struct {
	Description   string            `json:"description,omitempty"`
	Dimension     string            `json:"dimension,omitempty"`     // e.g., target, topology, quantization, cache_control
	Feature       string            `json:"feature,omitempty"`       // e.g., vdso, radix, q4k_gemv, bp_plan
	Configuration map[string]string `json:"configuration,omitempty"` // explicit flags, levers, or knobs
	Hardware      string            `json:"hardware,omitempty"`      // host class or GPU architecture (e.g., m3pro, strix_halo, h100)
	CommitSHA     string            `json:"commit_sha,omitempty"`    // source commit provenance
	Witness       string            `json:"witness,omitempty"`       // path to artifact or reproduce command
	Tags          []string          `json:"tags,omitempty"`
}

// ArmResult captures a single evaluated measurement for an arm within a workload.
type ArmResult struct {
	ArmID            string                `json:"arm_id"`
	Workload         string                `json:"workload"`
	ArmKind          ArmKind               `json:"arm_kind"`
	PrimaryMetric    string                `json:"primary_metric"`
	PrimaryValue     float64               `json:"primary_value"`
	PrimaryUnit      string                `json:"primary_unit,omitempty"`
	Direction        OptimizationDirection `json:"direction"`
	SecondaryMetrics map[string]float64    `json:"secondary_metrics,omitempty"`
	Metadata         ArmMetadata           `json:"metadata"`
	Timestamp        time.Time             `json:"timestamp"`
	RunID            string                `json:"run_id,omitempty"`
}

// Validate checks that required fields on ArmResult are present and valid.
func (r *ArmResult) Validate() error {
	if strings.TrimSpace(r.ArmID) == "" {
		return errors.New("armtracking: arm_id cannot be empty")
	}
	if strings.TrimSpace(r.Workload) == "" {
		return errors.New("armtracking: workload cannot be empty")
	}
	if err := r.ArmKind.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(r.PrimaryMetric) == "" {
		return errors.New("armtracking: primary_metric cannot be empty")
	}
	if err := r.Direction.Validate(); err != nil {
		return err
	}
	if math.IsNaN(r.PrimaryValue) || math.IsInf(r.PrimaryValue, 0) {
		return errors.New("armtracking: primary_value must be a finite number")
	}
	return nil
}

// IsBetterThan reports whether r is strictly better than other according to Direction.
func (r *ArmResult) IsBetterThan(other *ArmResult) bool {
	if other == nil {
		return true
	}
	if r.Direction == HigherIsBetter {
		return r.PrimaryValue > other.PrimaryValue
	}
	return r.PrimaryValue < other.PrimaryValue
}

// AuditAction categorizes an entry in the append-only history log.
type AuditAction string

const (
	ActionRecordRun           AuditAction = "RECORD_RUN"            // Initial or non-improving run recorded
	ActionNewArmBest          AuditAction = "NEW_ARM_BEST"          // Arm improved upon its own prior best
	ActionNewWorkloadChampion AuditAction = "NEW_WORKLOAD_CHAMPION" // Arm achieved overall best in workload
	ActionBaselineShift       AuditAction = "BASELINE_SHIFT"        // A baseline arm shifted its frontier value
)

// AuditEntry records one historical modification, shift, or update in the registry.
type AuditEntry struct {
	EventID        string      `json:"event_id"`
	Timestamp      time.Time   `json:"timestamp"`
	ArmID          string      `json:"arm_id"`
	Workload       string      `json:"workload"`
	Action         AuditAction `json:"action"`
	Metric         string      `json:"metric"`
	PriorBestValue *float64    `json:"prior_best_value,omitempty"`
	NewValue       float64     `json:"new_value"`
	DeltaFromPrior *float64    `json:"delta_from_prior,omitempty"`
	CommitSHA      string      `json:"commit_sha,omitempty"`
	Witness        string      `json:"witness,omitempty"`
	Notes          string      `json:"notes,omitempty"`
}

// NextBestComparison models the explicit comparison between a target arm and the next best latest result.
type NextBestComparison struct {
	TargetArm       ArmResult `json:"target_arm"`
	TargetRank      int       `json:"target_rank"` // 1-based rank among latest best results in workload
	TotalArms       int       `json:"total_arms"`
	IsChampion      bool      `json:"is_champion"`      // True if TargetRank == 1
	NextBestArm     ArmResult `json:"next_best_arm"`    // If Champion: rank 2 runner-up. If not champion: rank 1 champion or immediately superior arm.
	NextBestRank    int       `json:"next_best_rank"`   // Rank of NextBestArm
	Delta           float64   `json:"delta"`            // TargetValue - NextBestValue
	SpeedupOrLift   float64   `json:"speedup_or_lift"`  // Performance ratio relative to next-best (>1.0 indicates improvement)
	PercentageDelta float64   `json:"percentage_delta"` // Percentage advantage/disadvantage against next-best
	Summary         string    `json:"summary"`          // Human-readable synthesis
	AuditEvents     int       `json:"audit_events_count"`
}

// LeaderboardRow represents one arm's current standings in a workload.
type LeaderboardRow struct {
	Rank           int         `json:"rank"`
	ArmID          string      `json:"arm_id"`
	ArmKind        ArmKind     `json:"arm_kind"`
	PrimaryValue   float64     `json:"primary_value"`
	PrimaryUnit    string      `json:"primary_unit,omitempty"`
	SpeedupVsBest  float64     `json:"speedup_vs_best"` // 1.0 for champion
	NextBestArmID  string      `json:"next_best_arm_id,omitempty"`
	NextBestValue  float64     `json:"next_best_value,omitempty"`
	NextBestMargin string      `json:"next_best_margin,omitempty"`
	CommitSHA      string      `json:"commit_sha,omitempty"`
	Witness        string      `json:"witness,omitempty"`
	RunCount       int         `json:"run_count"`
	LastUpdated    time.Time   `json:"last_updated"`
	Metadata       ArmMetadata `json:"metadata"`
}

// ArmRecord stores the current latest best result and complete run history for one arm.
type ArmRecord struct {
	LatestBest ArmResult    `json:"latest_best"`
	History    []ArmResult  `json:"history"`
	AuditLog   []AuditEntry `json:"audit_log"`
}

// WorkloadRegistry groups arms belonging to a specific workload and metric.
type WorkloadRegistry struct {
	Workload  string                `json:"workload"`
	Metric    string                `json:"metric"`
	Unit      string                `json:"unit,omitempty"`
	Direction OptimizationDirection `json:"direction"`
	Arms      map[string]*ArmRecord `json:"arms"` // keyed by ArmID
}

// Registry is the thread-safe, top-level container for all workloads, shifting arms, and audit history.
type Registry struct {
	mu        sync.RWMutex
	Schema    string                       `json:"schema"`
	Workloads map[string]*WorkloadRegistry `json:"workloads"` // keyed by workload name
}

// NewRegistry creates a new, empty shifting arms registry.
func NewRegistry() *Registry {
	return &Registry{
		Schema:    SchemaVersion,
		Workloads: make(map[string]*WorkloadRegistry),
	}
}

// Record ingests a new arm result. It preserves full history and automatically updates the
// latest best result for the arm and workload, appending audit events whenever a shift occurs.
func (r *Registry) Record(result ArmResult, notes string) (*AuditEntry, error) {
	if err := result.Validate(); err != nil {
		return nil, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if result.Timestamp.IsZero() {
		result.Timestamp = time.Now().UTC()
	}

	w, ok := r.Workloads[result.Workload]
	if !ok {
		w = &WorkloadRegistry{
			Workload:  result.Workload,
			Metric:    result.PrimaryMetric,
			Unit:      result.PrimaryUnit,
			Direction: result.Direction,
			Arms:      make(map[string]*ArmRecord),
		}
		r.Workloads[result.Workload] = w
	} else {
		// Verify metric and direction consistency
		if w.Metric != result.PrimaryMetric {
			return nil, fmt.Errorf("armtracking: metric mismatch for workload %q: registered %q, incoming %q",
				result.Workload, w.Metric, result.PrimaryMetric)
		}
		if w.Direction != result.Direction {
			return nil, fmt.Errorf("armtracking: direction mismatch for workload %q: registered %q, incoming %q",
				result.Workload, w.Direction, result.Direction)
		}
	}

	// Check prior best across the workload to detect new workload champion
	var priorWorkloadBest *ArmResult
	for _, rec := range w.Arms {
		if priorWorkloadBest == nil || rec.LatestBest.IsBetterThan(priorWorkloadBest) {
			b := rec.LatestBest
			priorWorkloadBest = &b
		}
	}

	armRec, armExists := w.Arms[result.ArmID]
	if !armExists {
		armRec = &ArmRecord{
			LatestBest: result,
			History:    []ArmResult{result},
			AuditLog:   nil,
		}
		w.Arms[result.ArmID] = armRec

		action := ActionRecordRun
		if result.ArmKind == ArmKindBaseline {
			action = ActionBaselineShift
		} else if priorWorkloadBest == nil || result.IsBetterThan(priorWorkloadBest) {
			action = ActionNewWorkloadChampion
		}

		event := r.mintAuditEventLocked(result, action, nil, result.PrimaryValue, notes)
		armRec.AuditLog = append(armRec.AuditLog, event)
		return &event, nil
	}

	// Arm exists: append to history
	priorBest := armRec.LatestBest
	armRec.History = append(armRec.History, result)

	isNewArmBest := result.IsBetterThan(&priorBest)
	if !isNewArmBest {
		// Recorded run did not surpass latest best, but is preserved in audit history
		event := r.mintAuditEventLocked(result, ActionRecordRun, &priorBest.PrimaryValue, result.PrimaryValue, notes)
		armRec.AuditLog = append(armRec.AuditLog, event)
		return &event, nil
	}

	// Update latest best
	armRec.LatestBest = result

	action := ActionNewArmBest
	if result.ArmKind == ArmKindBaseline {
		action = ActionBaselineShift
	} else if priorWorkloadBest != nil && result.IsBetterThan(priorWorkloadBest) {
		action = ActionNewWorkloadChampion
	}

	event := r.mintAuditEventLocked(result, action, &priorBest.PrimaryValue, result.PrimaryValue, notes)
	armRec.AuditLog = append(armRec.AuditLog, event)
	return &event, nil
}

func (r *Registry) mintAuditEventLocked(res ArmResult, action AuditAction, prior *float64, val float64, notes string) AuditEntry {
	var delta *float64
	if prior != nil {
		d := val - *prior
		delta = &d
	}

	hashInput := fmt.Sprintf("%s:%s:%s:%f:%d", res.Workload, res.ArmID, action, val, res.Timestamp.UnixNano())
	sum := sha256.Sum256([]byte(hashInput))
	eventID := "evt-" + hex.EncodeToString(sum[:8])

	return AuditEntry{
		EventID:        eventID,
		Timestamp:      res.Timestamp,
		ArmID:          res.ArmID,
		Workload:       res.Workload,
		Action:         action,
		Metric:         res.PrimaryMetric,
		PriorBestValue: prior,
		NewValue:       val,
		DeltaFromPrior: delta,
		CommitSHA:      res.Metadata.CommitSHA,
		Witness:        res.Metadata.Witness,
		Notes:          notes,
	}
}

// CompareToNextBest retrieves the latest best result for armID in workload and compares it against
// the next best latest result among all peer arms in that workload.
func (r *Registry) CompareToNextBest(workload, armID string) (*NextBestComparison, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	w, ok := r.Workloads[workload]
	if !ok {
		return nil, fmt.Errorf("armtracking: workload %q not found", workload)
	}

	targetRec, ok := w.Arms[armID]
	if !ok {
		return nil, fmt.Errorf("armtracking: arm %q not found in workload %q", armID, workload)
	}

	if len(w.Arms) < 2 {
		// Single arm in workload, no peer exists
		target := targetRec.LatestBest
		return &NextBestComparison{
			TargetArm:       target,
			TargetRank:      1,
			TotalArms:       1,
			IsChampion:      true,
			NextBestArm:     target,
			NextBestRank:    1,
			Delta:           0,
			SpeedupOrLift:   1.0,
			PercentageDelta: 0.0,
			Summary:         fmt.Sprintf("arm %q is the only arm in workload %q (value: %g %s)", armID, workload, target.PrimaryValue, target.PrimaryUnit),
			AuditEvents:     len(targetRec.AuditLog),
		}, nil
	}

	// Sort all latest best arms according to workload direction
	type rankedArm struct {
		arm ArmResult
	}
	ranked := make([]rankedArm, 0, len(w.Arms))
	for _, arm := range w.Arms {
		ranked = append(ranked, rankedArm{arm: arm.LatestBest})
	}

	sort.Slice(ranked, func(i, j int) bool {
		if w.Direction == HigherIsBetter {
			return ranked[i].arm.PrimaryValue > ranked[j].arm.PrimaryValue
		}
		return ranked[i].arm.PrimaryValue < ranked[j].arm.PrimaryValue
	})

	targetRank := 1
	for idx, rk := range ranked {
		if rk.arm.ArmID == armID {
			targetRank = idx + 1
			break
		}
	}

	target := targetRec.LatestBest
	var nextBest ArmResult
	var nextBestRank int
	isChampion := targetRank == 1

	if isChampion {
		// Champion compares to runner-up (#2)
		nextBest = ranked[1].arm
		nextBestRank = 2
	} else {
		// Non-champion compares to the rank immediately ahead of it (#targetRank - 1)
		nextBest = ranked[targetRank-2].arm
		nextBestRank = targetRank - 1
	}

	cmp := buildNextBestComparison(target, targetRank, nextBest, nextBestRank, len(ranked), w.Direction)
	cmp.AuditEvents = len(targetRec.AuditLog)
	return &cmp, nil
}

func buildNextBestComparison(target ArmResult, targetRank int, nextBest ArmResult, nextBestRank int, total int, dir OptimizationDirection) NextBestComparison {
	delta := target.PrimaryValue - nextBest.PrimaryValue

	var speedup float64
	var pctChange float64

	if dir == HigherIsBetter {
		// Throughput/accuracy: higher is better
		if nextBest.PrimaryValue != 0 {
			speedup = target.PrimaryValue / nextBest.PrimaryValue
			pctChange = ((target.PrimaryValue - nextBest.PrimaryValue) / nextBest.PrimaryValue) * 100.0
		} else {
			speedup = 1.0
			pctChange = 0.0
		}
	} else {
		// Latency/memory: lower is better
		if target.PrimaryValue != 0 {
			speedup = nextBest.PrimaryValue / target.PrimaryValue
		} else {
			speedup = 1.0
		}
		if nextBest.PrimaryValue != 0 {
			pctChange = ((target.PrimaryValue - nextBest.PrimaryValue) / nextBest.PrimaryValue) * 100.0
		} else {
			pctChange = 0.0
		}
	}

	var summary strings.Builder
	isChamp := targetRank == 1
	if isChamp {
		if pctChange >= 0 {
			fmt.Fprintf(&summary, "%s (Rank %d/%d) leads runner-up %s (Rank %d) by %.2f× (+%.1f%%) on %s (%g vs %g %s)",
				target.ArmID, targetRank, total, nextBest.ArmID, nextBestRank, speedup, pctChange, target.PrimaryMetric,
				target.PrimaryValue, nextBest.PrimaryValue, target.PrimaryUnit)
		} else {
			fmt.Fprintf(&summary, "%s (Rank %d/%d) leads runner-up %s (Rank %d) by %.2f× (%.1f%%) on %s (%g vs %g %s)",
				target.ArmID, targetRank, total, nextBest.ArmID, nextBestRank, speedup, pctChange, target.PrimaryMetric,
				target.PrimaryValue, nextBest.PrimaryValue, target.PrimaryUnit)
		}
	} else {
		fmt.Fprintf(&summary, "%s (Rank %d/%d) trails superior %s (Rank %d) by %.2f× (%.1f%%) on %s (%g vs %g %s)",
			target.ArmID, targetRank, total, nextBest.ArmID, nextBestRank, speedup, pctChange, target.PrimaryMetric,
			target.PrimaryValue, nextBest.PrimaryValue, target.PrimaryUnit)
	}

	return NextBestComparison{
		TargetArm:       target,
		TargetRank:      targetRank,
		TotalArms:       total,
		IsChampion:      isChamp,
		NextBestArm:     nextBest,
		NextBestRank:    nextBestRank,
		Delta:           delta,
		SpeedupOrLift:   speedup,
		PercentageDelta: pctChange,
		Summary:         summary.String(),
	}
}

// Leaderboard generates a ranked view of latest best results for a given workload.
func (r *Registry) Leaderboard(workload string) ([]LeaderboardRow, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	w, ok := r.Workloads[workload]
	if !ok {
		return nil, fmt.Errorf("armtracking: workload %q not found", workload)
	}

	if len(w.Arms) == 0 {
		return nil, nil
	}

	// Sort arms
	type armItem struct {
		rec *ArmRecord
	}
	items := make([]armItem, 0, len(w.Arms))
	for _, arm := range w.Arms {
		items = append(items, armItem{rec: arm})
	}

	sort.Slice(items, func(i, j int) bool {
		if w.Direction == HigherIsBetter {
			return items[i].rec.LatestBest.PrimaryValue > items[j].rec.LatestBest.PrimaryValue
		}
		return items[i].rec.LatestBest.PrimaryValue < items[j].rec.LatestBest.PrimaryValue
	})

	championVal := items[0].rec.LatestBest.PrimaryValue
	rows := make([]LeaderboardRow, len(items))

	for i, it := range items {
		rank := i + 1
		arm := it.rec.LatestBest

		var speedupVsBest float64
		if w.Direction == HigherIsBetter {
			if championVal != 0 {
				speedupVsBest = arm.PrimaryValue / championVal
			} else {
				speedupVsBest = 1.0
			}
		} else {
			if arm.PrimaryValue != 0 {
				speedupVsBest = championVal / arm.PrimaryValue
			} else {
				speedupVsBest = 1.0
			}
		}

		var nextBestArmID string
		var nextBestVal float64
		var margin string

		if len(items) > 1 {
			if rank == 1 {
				runnerUp := items[1].rec.LatestBest
				nextBestArmID = runnerUp.ArmID
				nextBestVal = runnerUp.PrimaryValue
				ratio := 1.0
				if w.Direction == HigherIsBetter && runnerUp.PrimaryValue != 0 {
					ratio = arm.PrimaryValue / runnerUp.PrimaryValue
					margin = fmt.Sprintf("+%.1f%% vs #2 %s", (ratio-1.0)*100.0, runnerUp.ArmID)
				} else if w.Direction == LowerIsBetter && arm.PrimaryValue != 0 {
					ratio = runnerUp.PrimaryValue / arm.PrimaryValue
					margin = fmt.Sprintf("%.2f× faster vs #2 %s", ratio, runnerUp.ArmID)
				}
			} else {
				superior := items[i-1].rec.LatestBest
				nextBestArmID = superior.ArmID
				nextBestVal = superior.PrimaryValue
				if w.Direction == HigherIsBetter && superior.PrimaryValue != 0 {
					pct := ((arm.PrimaryValue - superior.PrimaryValue) / superior.PrimaryValue) * 100.0
					margin = fmt.Sprintf("%.1f%% vs #%d %s", pct, i, superior.ArmID)
				} else if w.Direction == LowerIsBetter && arm.PrimaryValue != 0 {
					ratio := superior.PrimaryValue / arm.PrimaryValue
					margin = fmt.Sprintf("%.2f× vs #%d %s", ratio, i, superior.ArmID)
				}
			}
		}

		rows[i] = LeaderboardRow{
			Rank:           rank,
			ArmID:          arm.ArmID,
			ArmKind:        arm.ArmKind,
			PrimaryValue:   arm.PrimaryValue,
			PrimaryUnit:    arm.PrimaryUnit,
			SpeedupVsBest:  speedupVsBest,
			NextBestArmID:  nextBestArmID,
			NextBestValue:  nextBestVal,
			NextBestMargin: margin,
			CommitSHA:      arm.Metadata.CommitSHA,
			Witness:        arm.Metadata.Witness,
			RunCount:       len(it.rec.History),
			LastUpdated:    arm.Timestamp,
			Metadata:       arm.Metadata,
		}
	}

	return rows, nil
}

// AuditHistory retrieves the complete audit history for a workload, or scoped to a specific armID.
func (r *Registry) AuditHistory(workload string, armID string) ([]AuditEntry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	w, ok := r.Workloads[workload]
	if !ok {
		return nil, fmt.Errorf("armtracking: workload %q not found", workload)
	}

	var events []AuditEntry
	if armID != "" {
		arm, ok := w.Arms[armID]
		if !ok {
			return nil, fmt.Errorf("armtracking: arm %q not found in workload %q", armID, workload)
		}
		events = append(events, arm.AuditLog...)
	} else {
		for _, arm := range w.Arms {
			events = append(events, arm.AuditLog...)
		}
	}

	sort.Slice(events, func(i, j int) bool {
		return events[i].Timestamp.Before(events[j].Timestamp)
	})

	return events, nil
}

// WorkloadsList returns all registered workload names sorted alphabetically.
func (r *Registry) WorkloadsList() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.Workloads))
	for name := range r.Workloads {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// GetArm returns the latest best result and run count for a given arm in a workload.
func (r *Registry) GetArm(workload, armID string) (*ArmResult, int, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	w, ok := r.Workloads[workload]
	if !ok {
		return nil, 0, false
	}
	arm, ok := w.Arms[armID]
	if !ok {
		return nil, 0, false
	}
	res := arm.LatestBest
	return &res, len(arm.History), true
}
