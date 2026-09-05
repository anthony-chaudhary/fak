package director

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/supervisoragent"
)

// RollupEngine aggregates multi-agent worker states and active lane leases in-process
// to compile high-frequency zero-self-report DirectorDigests.
type RollupEngine struct {
	mu             sync.RWMutex
	workers        map[string]WorkerDigestRow
	leases         map[string]LeaseSnapshot
	startTime      int64
	stallTimeoutMs int64
	nowFunc        func() int64
}

// NewRollupEngine creates a new in-process RollupEngine.
func NewRollupEngine() *RollupEngine {
	return &RollupEngine{
		workers:   make(map[string]WorkerDigestRow),
		leases:    make(map[string]LeaseSnapshot),
		startTime: time.Now().UnixMilli(),
		nowFunc: func() int64 {
			return time.Now().UnixMilli()
		},
	}
}

// RecordWorker records or updates a worker's digest row.
func (e *RollupEngine) RecordWorker(row WorkerDigestRow) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if row.State == "" && row.RunID != "" {
		row.State = WorkerHealthy
	}
	if e.stallTimeoutMs > 0 && row.LastWitnessMs > 0 && row.State != WorkerDone {
		now := e.now()
		if now-row.LastWitnessMs > e.stallTimeoutMs {
			row.State = WorkerStalled
		}
	}
	e.workers[row.RunID] = row
}

// RecordLease records or updates an active lane lease snapshot.
func (e *RollupEngine) RecordLease(lease LeaseSnapshot) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.leases[lease.Lane] = lease
}

// GetWorker returns a worker row by RunID.
func (e *RollupEngine) GetWorker(runID string) (WorkerDigestRow, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	row, ok := e.workers[runID]
	return row, ok
}

// GetLease returns a lease snapshot by Lane.
func (e *RollupEngine) GetLease(lane string) (LeaseSnapshot, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	lease, ok := e.leases[lane]
	return lease, ok
}

// RemoveWorker removes a worker from tracking.
func (e *RollupEngine) RemoveWorker(runID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.workers, runID)
}

// RemoveLease removes a lease from tracking.
func (e *RollupEngine) RemoveLease(lane string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.leases, lane)
}

// Reset clears all tracked workers and leases.
func (e *RollupEngine) Reset() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.workers = make(map[string]WorkerDigestRow)
	e.leases = make(map[string]LeaseSnapshot)
}

// SetStartTime overrides the engine's start timestamp.
func (e *RollupEngine) SetStartTime(ts int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.startTime = ts
}

// SetStallTimeoutMs configures the inactivity threshold before marking a worker stalled.
func (e *RollupEngine) SetStallTimeoutMs(ms int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.stallTimeoutMs = ms
}

// SetNowFunc overrides the clock function for deterministic testing.
func (e *RollupEngine) SetNowFunc(fn func() int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.nowFunc = fn
}

func (e *RollupEngine) now() int64 {
	if e.nowFunc != nil {
		return e.nowFunc()
	}
	return time.Now().UnixMilli()
}

// CompileDigest builds the current DirectorDigest across all tracked workers and leases.
func (e *RollupEngine) CompileDigest() DirectorDigest {
	e.mu.RLock()
	defer e.mu.RUnlock()

	now := e.now()
	totalWorkers := len(e.workers)
	activeWorkers := 0
	stalledWorkers := 0
	completedWorkers := 0
	blockedWorkers := 0
	totalCommits := 0
	sumVelocity := 0.0

	workers := make([]WorkerDigestRow, 0, totalWorkers)
	for _, w := range e.workers {
		if e.stallTimeoutMs > 0 && w.LastWitnessMs > 0 && w.State != WorkerDone {
			if now-w.LastWitnessMs > e.stallTimeoutMs {
				w.State = WorkerStalled
			}
		}

		switch w.State {
		case WorkerHealthy:
			activeWorkers++
		case WorkerStalled:
			stalledWorkers++
		case WorkerDone:
			completedWorkers++
		case WorkerBlocked:
			blockedWorkers++
		default:
			// Fail-safe: unknown state counts as blocked (needs attention)
			blockedWorkers++
		}

		totalCommits += w.VerifiedCommits
		sumVelocity += w.VelocityScore
		workers = append(workers, w)
	}

	leases := make([]LeaseSnapshot, 0, len(e.leases))
	for _, l := range e.leases {
		leases = append(leases, l)
	}

	// Deterministic ordering
	sort.Slice(workers, func(i, j int) bool {
		return workers[i].RunID < workers[j].RunID
	})
	sort.Slice(leases, func(i, j int) bool {
		return leases[i].Lane < leases[j].Lane
	})

	var blockRate, stallRate float64
	if totalWorkers > 0 {
		blockRate = float64(blockedWorkers) / float64(totalWorkers)
		stallRate = float64(stalledWorkers) / float64(totalWorkers)
	}

	// Calculate commits per hour
	var commitsPerHour float64
	elapsedHours := float64(now-e.startTime) / 3600000.0
	if elapsedHours > 0.0001 && totalCommits > 0 {
		commitsPerHour = float64(totalCommits) / elapsedHours
	} else if sumVelocity > 0 {
		commitsPerHour = sumVelocity
	} else {
		commitsPerHour = float64(totalCommits)
	}

	digest := DirectorDigest{
		Schema:           DigestSchema,
		Timestamp:        now,
		TotalWorkers:     totalWorkers,
		ActiveWorkers:    activeWorkers,
		StalledWorkers:   stalledWorkers,
		CompletedWorkers: completedWorkers,
		FleetVelocity: FleetVelocityScore{
			TotalCommits:   totalCommits,
			CommitsPerHour: commitsPerHour,
			BlockRate:      blockRate,
			StallRate:      stallRate,
		},
		Workers: workers,
		Leases:  leases,
	}

	digest.RollupHash = computeRollupHash(&digest)
	return digest
}

func computeRollupHash(d *DirectorDigest) string {
	h := sha256.New()
	fmt.Fprintf(h, "schema=%s|ts=%d|tot=%d|act=%d|stl=%d|cmp=%d|cph=%.4f|blk=%.4f|stl=%.4f\n",
		d.Schema, d.Timestamp, d.TotalWorkers, d.ActiveWorkers, d.StalledWorkers, d.CompletedWorkers,
		d.FleetVelocity.CommitsPerHour, d.FleetVelocity.BlockRate, d.FleetVelocity.StallRate)
	for _, w := range d.Workers {
		fmt.Fprintf(h, "w:%s|%s|%s|%s|%d|%d|%d|%.4f|%d\n",
			w.RunID, w.Lane, w.Issue, w.State, w.StepCount, w.VerifiedCommits, w.TreeTouches, w.VelocityScore, w.LastWitnessMs)
	}
	for _, l := range d.Leases {
		fmt.Fprintf(h, "l:%s|%s|%s|%s|%s\n",
			l.Lane, l.LaneKind, strings.Join(l.Tree, ","), l.Holder, l.Mode)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// EvaluateFleetSteering maps DirectorDigest metrics to closed supervisor actions.
func (e *RollupEngine) EvaluateFleetSteering(d DirectorDigest) []SteeringRecommendation {
	return EvaluateFleetSteering(d)
}

// EvaluateFleetSteering maps DirectorDigest metrics to closed supervisor actions:
// spawn, replace, replan, hold, widen, escalate.
func EvaluateFleetSteering(d DirectorDigest) []SteeringRecommendation {
	var recs []SteeringRecommendation

	// 1. Critical fleet-wide thresholds -> escalate
	if d.TotalWorkers >= 2 && d.FleetVelocity.StallRate >= 0.5 {
		recs = append(recs, SteeringRecommendation{
			Action: supervisoragent.ActionEscalate,
			Reason: ReasonFleetHighStallRate,
			SupervisorAction: supervisoragent.EscalateAction{
				Class:      "director",
				Severity:   "operator",
				ReasonCode: ReasonFleetHighStallRate,
			},
		})
	}
	if d.TotalWorkers >= 2 && d.FleetVelocity.BlockRate >= 0.5 {
		recs = append(recs, SteeringRecommendation{
			Action: supervisoragent.ActionEscalate,
			Reason: ReasonFleetHighBlockRate,
			SupervisorAction: supervisoragent.EscalateAction{
				Class:      "director",
				Severity:   "operator",
				ReasonCode: ReasonFleetHighBlockRate,
			},
		})
	}

	// 2. Per-worker steering evaluations
	for _, w := range d.Workers {
		switch w.State {
		case WorkerStalled:
			// Stalled worker needs replacement
			recs = append(recs, SteeringRecommendation{
				Action: supervisoragent.ActionReplace,
				RunID:  w.RunID,
				Issue:  w.Issue,
				Lane:   w.Lane,
				Reason: ReasonWorkerStalled,
				SupervisorAction: supervisoragent.ReplaceAction{
					RunID: w.RunID,
					Issue: w.Issue,
					Lane:  w.Lane,
				},
			})

		case WorkerBlocked:
			// Blocked worker needs replan / redispatch
			recs = append(recs, SteeringRecommendation{
				Action: supervisoragent.ActionRedispatch,
				RunID:  w.RunID,
				Issue:  w.Issue,
				Lane:   w.Lane,
				Reason: ReasonWorkerBlocked,
				SupervisorAction: supervisoragent.RedispatchAction{
					Issue: w.Issue,
					Lane:  w.Lane,
				},
			})

		case WorkerHealthy:
			// Check for thrashing runaway worker (escalate)
			if w.StepCount >= 50 && w.VerifiedCommits == 0 && w.TreeTouches >= 10 {
				recs = append(recs, SteeringRecommendation{
					Action: supervisoragent.ActionEscalate,
					RunID:  w.RunID,
					Issue:  w.Issue,
					Lane:   w.Lane,
					Reason: ReasonWorkerThrashing,
					SupervisorAction: supervisoragent.EscalateAction{
						RunID:      w.RunID,
						Issue:      w.Issue,
						Class:      "director",
						Severity:   "operator",
						ReasonCode: ReasonWorkerThrashing,
					},
				})
			} else if w.TreeTouches >= 25 && w.VerifiedCommits == 0 {
				// Tree expansion needed (widen)
				recs = append(recs, SteeringRecommendation{
					Action: supervisoragent.ActionWiden,
					RunID:  w.RunID,
					Issue:  w.Issue,
					Lane:   w.Lane,
					Reason: ReasonLaneWiden,
					SupervisorAction: supervisoragent.WidenAction{
						Lane: w.Lane,
					},
				})
			}
		}
	}

	// 3. Idle lease check (spawn)
	activeLanes := make(map[string]bool)
	for _, w := range d.Workers {
		if w.State != WorkerDone {
			activeLanes[w.Lane] = true
		}
	}
	for _, l := range d.Leases {
		if l.Holder == "" && !activeLanes[l.Lane] && l.Lane != "" {
			recs = append(recs, SteeringRecommendation{
				Action: supervisoragent.ActionSpawn,
				Lane:   l.Lane,
				Tree:   l.Tree,
				Reason: ReasonLaneIdle,
				SupervisorAction: supervisoragent.SpawnAction{
					Lane: l.Lane,
				},
			})
		}
	}

	// 4. Default: hold when no corrective actions are needed
	if len(recs) == 0 {
		reason := ReasonFleetHealthy
		if d.TotalWorkers == 0 {
			reason = ReasonFleetIdle
		}
		recs = append(recs, SteeringRecommendation{
			Action:           supervisoragent.ActionHold,
			Reason:           reason,
			SupervisorAction: supervisoragent.HoldAction{},
		})
	}

	return recs
}

// CompileAndEvaluate is a convenience method that compiles a digest and evaluates steering.
func (e *RollupEngine) CompileAndEvaluate() (DirectorDigest, []SteeringRecommendation) {
	d := e.CompileDigest()
	return d, e.EvaluateFleetSteering(d)
}
