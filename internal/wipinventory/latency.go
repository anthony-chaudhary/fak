package wipinventory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ProtectionLatencySchema is the stable schema token for source-work protection latency reports.
const ProtectionLatencySchema = "fak-protection-latency/1"

// ProtectionLatencyReport summarizes protection latency intervals and SLO compliance.
type ProtectionLatencyReport struct {
	Schema                     string              `json:"schema"`
	ObservedAt                 time.Time           `json:"observed_at"`
	Repo                       string              `json:"repo"`
	TotalSourcePaths           int                 `json:"total_source_paths"`
	Outcomes                   map[string]int      `json:"outcomes"`
	Surfaces                   map[string]int      `json:"surfaces"`
	P50Seconds                 float64             `json:"p50_seconds"`
	P95Seconds                 float64             `json:"p95_seconds"`
	MaxSeconds                 float64             `json:"max_seconds"`
	ProtectedWithinBudgetRatio float64             `json:"protected_within_budget_ratio"`
	StaleRefusalCount          int                 `json:"stale_refusal_count"`
	UnknownClockCount          int                 `json:"unknown_clock_count"`
	PathSamples                []PathLatencySample `json:"path_samples"`
	SLOVerdict                 string              `json:"slo_verdict"`
	Errors                     []string            `json:"errors,omitempty"`
}

// JSON encodes the report with indentation.
func (r ProtectionLatencyReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// PathLatencySample captures protection interval details for one source path.
type PathLatencySample struct {
	Path           string     `json:"path"`
	FirstSeen      *time.Time `json:"first_seen,omitempty"`
	ProtectedAt    *time.Time `json:"protected_at,omitempty"`
	LatencySeconds *float64   `json:"latency_seconds,omitempty"`
	Outcome        string     `json:"outcome"`
	Surface        string     `json:"surface"`
	ClockKnown     bool       `json:"clock_known"`
}

// LatencyOptions configures the latency measurement pass.
type LatencyOptions struct {
	Now         time.Time
	Budget      time.Duration
	Runner      Runner
	WorkerRoot  string
	CommitLimit int
	Stat        func(path string) (os.FileInfo, error)
}

type landedCommit struct {
	SHA           string
	CommitterUnix int64
	AuthorUnix    int64
}

// MeasureProtectionLatency derives protection intervals from working-tree status,
// existing refs/fak/wip/* checkpoints, and recent git history.
func MeasureProtectionLatency(ctx context.Context, repo string, opts LatencyOptions) (*ProtectionLatencyReport, error) {
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}

	if repo == "" {
		repo = "."
	}
	abs, err := filepath.Abs(repo)
	if err == nil {
		repo = abs
	}
	cleanRepo := filepath.ToSlash(filepath.Clean(repo))

	now := opts.Now.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}
	budget := opts.Budget
	if budget <= 0 {
		budget = time.Hour
	}
	runner := opts.Runner
	if runner == nil {
		runner = GitRunner{}
	}
	workerRoot := opts.WorkerRoot
	if workerRoot == "" {
		workerRoot = defaultWorkerRoot()
	}
	commitLimit := opts.CommitLimit
	if commitLimit <= 0 {
		commitLimit = 20
	}
	statFn := opts.Stat
	if statFn == nil {
		statFn = os.Stat
	}

	var errorsList []string

	// 1. Collect ignored files.
	ignoredMap := make(map[string]bool)
	if out, err := runner.Run(cleanRepo, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"); err == nil {
		for _, raw := range bytes.Split(out, []byte{0}) {
			if len(raw) > 0 {
				path := filepath.ToSlash(string(raw))
				ignoredMap[path] = true
				if strings.HasSuffix(path, "/") {
					ignoredMap[strings.TrimSuffix(path, "/")] = true
				}
			}
		}
	} else {
		errorsList = append(errorsList, "ignored_visibility: "+err.Error())
	}

	// 2. Collect existing checkpoints under refs/fak/wip.
	dummyReport := &Report{}
	cps, _ := checkpoints(cleanRepo, runner, dummyReport)
	errorsList = append(errorsList, dummyReport.Errors...)

	checkpointsByPath := make(map[string]*Checkpoint)
	for i := range cps {
		cp := &cps[i]
		for _, p := range cp.allPaths {
			norm := filepath.ToSlash(p)
			if existing, ok := checkpointsByPath[norm]; !ok || cp.Unix < existing.Unix {
				checkpointsByPath[norm] = cp
			}
		}
	}

	// 3. Collect registered worker worktrees.
	dummyReport2 := &Report{}
	wts, _ := worktrees(cleanRepo, workerRoot, runner, dummyReport2)
	errorsList = append(errorsList, dummyReport2.Errors...)

	workerPaths := make(map[string]bool)
	for _, wt := range wts {
		for _, s := range wt.Tracked.Samples {
			workerPaths[filepath.ToSlash(s)] = true
		}
		for _, s := range wt.Untracked.Samples {
			workerPaths[filepath.ToSlash(s)] = true
		}
	}

	// 4. Collect working tree status from main.
	workingTreeStatus := make(map[string]string)
	if out, err := runner.Run(cleanRepo, "status", "--porcelain=v1", "-z", "--untracked-files=all"); err == nil {
		for _, raw := range bytes.Split(out, []byte{0}) {
			if len(raw) < 4 {
				continue
			}
			line := string(raw)
			statusCode := line[:2]
			name := filepath.ToSlash(line[3:])

			// Deleted paths do not count as active source work.
			if strings.HasPrefix(line, " D") || strings.HasPrefix(line, "D ") {
				continue
			}
			if !isSourcePath(cleanRepo, name, ignoredMap, statFn) {
				continue
			}
			workingTreeStatus[name] = statusCode
		}
	} else {
		errorsList = append(errorsList, "working_tree_status: "+err.Error())
	}

	// 5. Collect landed commits from git log.
	landedByPath := make(map[string]landedCommit)
	if out, err := runner.Run(cleanRepo, "log", fmt.Sprintf("-n%d", commitLimit), "--format=commit %H%x00%ct%x00%at", "--name-status"); err == nil {
		var curCommit landedCommit
		for _, row := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			row = strings.TrimRight(row, "\r")
			if row == "" {
				continue
			}
			if strings.HasPrefix(row, "commit ") {
				meta := strings.TrimPrefix(row, "commit ")
				parts := strings.Split(meta, "\x00")
				if len(parts) >= 3 {
					ct, _ := strconv.ParseInt(parts[1], 10, 64)
					at, _ := strconv.ParseInt(parts[2], 10, 64)
					curCommit = landedCommit{
						SHA:           parts[0],
						CommitterUnix: ct,
						AuthorUnix:    at,
					}
				}
				continue
			}
			parts := strings.SplitN(row, "\t", 2)
			if len(parts) == 2 {
				status := parts[0]
				path := filepath.ToSlash(parts[1])
				if status == "D" {
					continue
				}
				if !isSourcePath(cleanRepo, path, ignoredMap, statFn) {
					continue
				}
				if _, ok := landedByPath[path]; !ok {
					landedByPath[path] = curCommit
				}
			}
		}
	} else {
		errorsList = append(errorsList, "git_log: "+err.Error())
	}

	// 6. Aggregate unique candidate source paths.
	allCandidatePaths := make(map[string]bool)
	for p := range workingTreeStatus {
		allCandidatePaths[p] = true
	}
	for p := range checkpointsByPath {
		if isSourcePath(cleanRepo, p, ignoredMap, statFn) {
			allCandidatePaths[p] = true
		}
	}
	for p := range workerPaths {
		if isSourcePath(cleanRepo, p, ignoredMap, statFn) {
			allCandidatePaths[p] = true
		}
	}
	for p := range landedByPath {
		allCandidatePaths[p] = true
	}

	sortedPaths := make([]string, 0, len(allCandidatePaths))
	for p := range allCandidatePaths {
		sortedPaths = append(sortedPaths, p)
	}
	sort.Strings(sortedPaths)

	samples := make([]PathLatencySample, 0, len(sortedPaths))
	for _, p := range sortedPaths {
		sample := PathLatencySample{Path: p}

		// Priority 1: Checkpointed
		if cp, ok := checkpointsByPath[p]; ok {
			sample.Outcome = "checkpointed"
			sample.Surface = "checkpoint"
			sample.ClockKnown = true
			cpTime := time.Unix(cp.Unix, 0).UTC()
			sample.ProtectedAt = &cpTime

			info, err := statFn(filepath.Join(cleanRepo, filepath.FromSlash(p)))
			if err == nil {
				mtime := info.ModTime().UTC()
				sample.FirstSeen = &mtime
				if cpTime.After(mtime) {
					diff := cpTime.Sub(mtime).Seconds()
					sample.LatencySeconds = &diff
				} else {
					lat := 0.0
					sample.LatencySeconds = &lat
				}
			} else {
				sample.FirstSeen = &cpTime
				lat := 0.0
				sample.LatencySeconds = &lat
			}
			samples = append(samples, sample)
			continue
		}

		// Priority 2: Worker isolated
		if workerPaths[p] {
			sample.Outcome = "worker_isolated"
			sample.Surface = "detached_worker"
			sample.ClockKnown = true
			sample.ProtectedAt = &now
			sample.FirstSeen = &now
			lat := 0.0
			sample.LatencySeconds = &lat
			samples = append(samples, sample)
			continue
		}

		// Priority 3: Working tree uncommitted/unprotected
		if _, ok := workingTreeStatus[p]; ok {
			sample.Outcome = "unprotected"
			sample.Surface = "shared_trunk"
			sample.ProtectedAt = nil

			info, err := statFn(filepath.Join(cleanRepo, filepath.FromSlash(p)))
			if err != nil {
				sample.ClockKnown = false
				sample.FirstSeen = nil
				sample.LatencySeconds = nil
			} else {
				sample.ClockKnown = true
				mtime := info.ModTime().UTC()
				sample.FirstSeen = &mtime
				age := now.Sub(mtime).Seconds()
				if age < 0 {
					age = 0
				}
				sample.LatencySeconds = &age
			}
			samples = append(samples, sample)
			continue
		}

		// Priority 4: Landed in git log
		if lc, ok := landedByPath[p]; ok {
			sample.Outcome = "landed"
			sample.Surface = "shared_trunk"
			sample.ClockKnown = true
			commitTime := time.Unix(lc.CommitterUnix, 0).UTC()
			sample.ProtectedAt = &commitTime

			if lc.AuthorUnix > 0 {
				authorTime := time.Unix(lc.AuthorUnix, 0).UTC()
				sample.FirstSeen = &authorTime
				diff := float64(lc.CommitterUnix - lc.AuthorUnix)
				if diff < 0 {
					diff = 0
				}
				sample.LatencySeconds = &diff
			} else {
				sample.FirstSeen = &commitTime
				lat := 0.0
				sample.LatencySeconds = &lat
			}
			samples = append(samples, sample)
			continue
		}
	}

	report := ComputeProtectionLatency(now, cleanRepo, budget, samples)
	if len(errorsList) > 0 {
		sort.Strings(errorsList)
		report.Errors = errorsList
	}
	return report, nil
}

// ComputeProtectionLatency computes aggregated metrics and SLO compliance from a list of samples.
func ComputeProtectionLatency(observedAt time.Time, repo string, budget time.Duration, samples []PathLatencySample) *ProtectionLatencyReport {
	if budget <= 0 {
		budget = time.Hour
	}
	if samples == nil {
		samples = []PathLatencySample{}
	}

	outcomes := map[string]int{
		"checkpointed":    0,
		"worker_isolated": 0,
		"landed":          0,
		"unprotected":     0,
	}
	surfaces := map[string]int{
		"shared_trunk":    0,
		"checkpoint":      0,
		"detached_worker": 0,
	}

	staleRefusalCount := 0
	unknownClockCount := 0
	protectedWithinBudgetCount := 0
	var latencies []float64

	for _, s := range samples {
		outcomes[s.Outcome]++
		surfaces[s.Surface]++

		if !s.ClockKnown || s.LatencySeconds == nil {
			unknownClockCount++
			continue
		}

		lat := *s.LatencySeconds
		latencies = append(latencies, lat)

		if s.Outcome == "unprotected" {
			if lat > budget.Seconds() {
				staleRefusalCount++
			}
		} else {
			if lat <= budget.Seconds() {
				protectedWithinBudgetCount++
			}
		}
	}

	total := len(samples)
	var ratio float64
	if total == 0 {
		ratio = 1.0
	} else {
		ratio = float64(protectedWithinBudgetCount) / float64(total)
	}

	sort.Float64s(latencies)
	var p50, p95, maxSec float64
	if len(latencies) > 0 {
		p50 = nearestRank(latencies, 50)
		p95 = nearestRank(latencies, 95)
		maxSec = latencies[len(latencies)-1]
	}

	var verdict string
	switch {
	case staleRefusalCount > 0 || ratio < 0.95:
		verdict = "VIOLATION"
	case ratio < 1.0 || outcomes["unprotected"] > 0:
		verdict = "NEEDS_ATTENTION"
	default:
		verdict = "PASS"
	}

	return &ProtectionLatencyReport{
		Schema:                     ProtectionLatencySchema,
		ObservedAt:                 observedAt.UTC(),
		Repo:                       repo,
		TotalSourcePaths:           total,
		Outcomes:                   outcomes,
		Surfaces:                   surfaces,
		P50Seconds:                 p50,
		P95Seconds:                 p95,
		MaxSeconds:                 maxSec,
		ProtectedWithinBudgetRatio: ratio,
		StaleRefusalCount:          staleRefusalCount,
		UnknownClockCount:          unknownClockCount,
		PathSamples:                samples,
		SLOVerdict:                 verdict,
	}
}

func nearestRank(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0.0
	}
	if p < 0 {
		p = 0
	}
	if p > 100 {
		p = 100
	}
	n := float64(len(sorted))
	rank := int(math.Ceil((p / 100.0) * n))
	if rank < 1 {
		rank = 1
	}
	if rank > len(sorted) {
		rank = len(sorted)
	}
	return sorted[rank-1]
}

func isSourcePath(cleanRepo, rel string, ignoredMap map[string]bool, statFn func(string) (os.FileInfo, error)) bool {
	rel = filepath.ToSlash(filepath.Clean(rel))
	if rel == "." || rel == "" {
		return false
	}
	if ignoredMap[rel] {
		return false
	}
	for ign := range ignoredMap {
		if strings.HasSuffix(ign, "/") && strings.HasPrefix(rel, ign) {
			return false
		}
	}
	segs := strings.Split(rel, "/")
	for _, s := range segs {
		low := strings.ToLower(s)
		if low == "vendor" || low == "node_modules" || low == "third_party" || low == "thirdparty" {
			return false
		}
		if strings.HasPrefix(s, ".") || strings.HasPrefix(s, "_scratch") || low == "scratch" || low == "tmp" || low == "temp" {
			return false
		}
	}
	base := strings.ToLower(segs[len(segs)-1])
	if strings.HasSuffix(base, ".pb.go") || strings.HasSuffix(base, "_generated.go") ||
		strings.HasSuffix(base, ".gen.go") || strings.HasPrefix(base, "zz_generated") {
		return false
	}
	for _, ext := range []string{".exe", ".test", ".dll", ".so", ".dylib", ".a", ".o", ".out"} {
		if strings.HasSuffix(base, ext) {
			return false
		}
	}

	fullPath := filepath.Join(cleanRepo, filepath.FromSlash(rel))
	if f, err := os.Open(fullPath); err == nil {
		buf := make([]byte, 1024)
		n, _ := f.Read(buf)
		f.Close()
		content := string(buf[:n])
		if strings.Contains(content, "DO NOT EDIT") && (strings.Contains(content, "Code generated") || strings.Contains(content, "GENERATED")) {
			return false
		}
	}
	return true
}

// Float64Ptr returns a pointer to the given float64 value.
func Float64Ptr(v float64) *float64 {
	return &v
}

// TimePtr returns a pointer to the given time.Time value in UTC.
func TimePtr(t time.Time) *time.Time {
	utc := t.UTC()
	return &utc
}
