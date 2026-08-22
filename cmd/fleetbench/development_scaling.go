package main

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

const (
	developmentFixtureSchema = "fak-development-dispatch-receipts/1"
	developmentReportSchema  = "fak-fleet-development-scaling/1"
)

var developmentSeatGrid = []int{1, 4, 8, 16, 30}

//go:embed testdata/issue-8477-development-receipts.json
var canonicalDevelopmentFixture []byte

type developmentFixture struct {
	Schema   string                  `json:"schema"`
	Workload developmentWorkload     `json:"workload"`
	Arms     []developmentArmFixture `json:"arms"`
}

type developmentWorkload struct {
	ID    string                `json:"id"`
	Items []developmentWorkItem `json:"items"`
}

type developmentWorkItem struct {
	ID           string   `json:"id"`
	Dependencies []string `json:"dependencies,omitempty"`
}

type developmentArmFixture struct {
	Seats    int                  `json:"seats"`
	Receipts []developmentReceipt `json:"receipts"`
}

type developmentReceipt struct {
	ID           string                       `json:"id"`
	WorkID       string                       `json:"work_id"`
	Worker       string                       `json:"worker"`
	Attempt      int                          `json:"attempt"`
	Admitted     bool                         `json:"admitted"`
	StartMS      int64                        `json:"start_ms"`
	EndMS        int64                        `json:"end_ms"`
	Outcome      string                       `json:"outcome"`
	ClosureStamp *developmentClosureStamp     `json:"witness,omitempty"`
	Accounting   developmentReceiptAccounting `json:"accounting"`
}

type developmentClosureStamp struct {
	Kind        string `json:"kind"`
	Reference   string `json:"reference"`
	Independent bool   `json:"independent"`
}

type developmentReceiptAccounting struct {
	ExecutionMS      int64 `json:"execution_ms"`
	QueueMS          int64 `json:"queue_ms"`
	CollisionMS      int64 `json:"collision_ms"`
	DependencyWaitMS int64 `json:"dependency_wait_ms"`
	RetryMS          int64 `json:"retry_ms"`
	LandingWaitMS    int64 `json:"landing_wait_ms"`
	VerificationMS   int64 `json:"verification_wait_ms"`
}

type developmentWorkloadIdentity struct {
	ID            string `json:"id"`
	Items         int    `json:"items"`
	ReceiptDigest string `json:"receipt_digest"`
}

type developmentLossBuckets struct {
	TotalWorkerMS           int64 `json:"total_worker_ms"`
	UsefulExecutionMS       int64 `json:"useful_execution_ms"`
	DuplicateExecutionMS    int64 `json:"duplicate_execution_ms"`
	UnverifiableExecutionMS int64 `json:"unverifiable_execution_ms"`
	QueueMS                 int64 `json:"queue_ms"`
	CollisionMS             int64 `json:"collision_ms"`
	DependencyWaitMS        int64 `json:"dependency_wait_ms"`
	RetryMS                 int64 `json:"retry_ms"`
	LandingWaitMS           int64 `json:"landing_wait_ms"`
	VerificationMS          int64 `json:"verification_wait_ms"`
}

type developmentArmResult struct {
	Seats                int                    `json:"seats"`
	AdmittedWorkers      int                    `json:"admitted_workers"`
	AcceptedClosures     int                    `json:"accepted_closures"`
	Attempts             int                    `json:"attempts"`
	DuplicateAttempts    int                    `json:"duplicate_attempts"`
	UnverifiableAttempts int                    `json:"unverifiable_attempts"`
	DroppedItems         int                    `json:"dropped_items"`
	MakespanMS           int64                  `json:"makespan_ms"`
	CriticalPathMS       int64                  `json:"critical_path_ms"`
	WIPAreaWorkerMS      int64                  `json:"wip_area_worker_ms"`
	RetryAttempts        int                    `json:"retry_attempts"`
	CollisionEvents      int                    `json:"collision_events"`
	Speedup              float64                `json:"speedup"`
	ParallelEfficiency   float64                `json:"parallel_efficiency"`
	DominantLimiter      string                 `json:"dominant_limiter"`
	Losses               developmentLossBuckets `json:"losses"`
}

type developmentReport struct {
	Schema          string                      `json:"schema"`
	Workload        developmentWorkloadIdentity `json:"workload"`
	BaselineSeats   int                         `json:"baseline_seats"`
	Arms            []developmentArmResult      `json:"arms"`
	DominantLimiter string                      `json:"dominant_limiter"`
	NextExperiment  string                      `json:"next_experiment"`
}

func runDevelopmentScaling(selfcheck bool, jsonOut, reportOut io.Writer) error {
	report, err := analyzeDevelopmentFixture(canonicalDevelopmentFixture)
	if err != nil {
		return err
	}
	if selfcheck {
		if err := runDevelopmentNegativeChecks(); err != nil {
			return err
		}
	}
	if err := writeDevelopmentSummary(reportOut, report, selfcheck); err != nil {
		return err
	}
	enc := json.NewEncoder(jsonOut)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func analyzeDevelopmentFixture(body []byte) (developmentReport, error) {
	fixture, err := decodeDevelopmentFixture(body)
	if err != nil {
		return developmentReport{}, err
	}
	items, err := validateDevelopmentWorkload(fixture)
	if err != nil {
		return developmentReport{}, err
	}
	arms := append([]developmentArmFixture(nil), fixture.Arms...)
	sort.Slice(arms, func(i, j int) bool { return arms[i].Seats < arms[j].Seats })
	if len(arms) != len(developmentSeatGrid) {
		return developmentReport{}, fmt.Errorf("fixture arms must be exactly %v", developmentSeatGrid)
	}
	results := make([]developmentArmResult, 0, len(arms))
	for i, arm := range arms {
		if arm.Seats != developmentSeatGrid[i] {
			return developmentReport{}, fmt.Errorf("fixture arms must be exactly %v", developmentSeatGrid)
		}
		result, err := analyzeDevelopmentArm(items, arm)
		if err != nil {
			return developmentReport{}, fmt.Errorf("%d-seat arm: %w", arm.Seats, err)
		}
		results = append(results, result)
	}
	baseline := results[0]
	if baseline.AcceptedClosures == 0 || baseline.MakespanMS == 0 {
		return developmentReport{}, errors.New("one-seat baseline requires witnessed closures and positive makespan")
	}
	var aggregate developmentLossBuckets
	for i := range results {
		results[i].Speedup = roundDevelopment(
			(float64(results[i].AcceptedClosures) / float64(results[i].MakespanMS)) /
				(float64(baseline.AcceptedClosures) / float64(baseline.MakespanMS)),
		)
		results[i].ParallelEfficiency = roundDevelopment(results[i].Speedup / float64(results[i].Seats))
		if results[i].Speedup > float64(results[i].Seats)+1e-9 || results[i].ParallelEfficiency > 1+1e-9 {
			return developmentReport{}, fmt.Errorf("%d-seat arm has impossible superlinear accounting: speedup %.6f exceeds seats", results[i].Seats, results[i].Speedup)
		}
		results[i].DominantLimiter, _ = dominantDevelopmentLimiter(results[i].Losses)
		aggregate.add(results[i].Losses)
	}
	dominant, _ := dominantDevelopmentLimiter(aggregate)
	sum := sha256.Sum256(body)
	return developmentReport{
		Schema: developmentReportSchema,
		Workload: developmentWorkloadIdentity{
			ID: fixture.Workload.ID, Items: len(fixture.Workload.Items),
			ReceiptDigest: "sha256:" + hex.EncodeToString(sum[:]),
		},
		BaselineSeats:   1,
		Arms:            results,
		DominantLimiter: dominant,
		NextExperiment:  nextDevelopmentExperiment(dominant),
	}, nil
}

func decodeDevelopmentFixture(body []byte) (developmentFixture, error) {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	var fixture developmentFixture
	if err := dec.Decode(&fixture); err != nil {
		return developmentFixture{}, fmt.Errorf("decode receipt fixture: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return developmentFixture{}, errors.New("decode receipt fixture: trailing JSON value")
	}
	if fixture.Schema != developmentFixtureSchema {
		return developmentFixture{}, fmt.Errorf("fixture schema must be %q", developmentFixtureSchema)
	}
	return fixture, nil
}

func validateDevelopmentWorkload(fixture developmentFixture) (map[string]developmentWorkItem, error) {
	if strings.TrimSpace(fixture.Workload.ID) == "" || len(fixture.Workload.Items) == 0 {
		return nil, errors.New("workload identity and items are required")
	}
	items := make(map[string]developmentWorkItem, len(fixture.Workload.Items))
	for _, item := range fixture.Workload.Items {
		if strings.TrimSpace(item.ID) == "" {
			return nil, errors.New("workload item id is required")
		}
		if _, exists := items[item.ID]; exists {
			return nil, fmt.Errorf("duplicate workload item %q", item.ID)
		}
		items[item.ID] = item
	}
	for _, item := range fixture.Workload.Items {
		for _, dep := range item.Dependencies {
			if dep == item.ID {
				return nil, fmt.Errorf("workload item %q depends on itself", item.ID)
			}
			if _, ok := items[dep]; !ok {
				return nil, fmt.Errorf("workload item %q has unknown dependency %q", item.ID, dep)
			}
		}
	}
	state := make(map[string]uint8, len(items))
	var visit func(string) error
	visit = func(id string) error {
		if state[id] == 1 {
			return fmt.Errorf("workload dependency cycle at %q", id)
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		for _, dep := range items[id].Dependencies {
			if err := visit(dep); err != nil {
				return err
			}
		}
		state[id] = 2
		return nil
	}
	for id := range items {
		if err := visit(id); err != nil {
			return nil, err
		}
	}
	return items, nil
}

func analyzeDevelopmentArm(items map[string]developmentWorkItem, arm developmentArmFixture) (developmentArmResult, error) {
	if arm.Seats <= 0 || len(arm.Receipts) == 0 {
		return developmentArmResult{}, errors.New("positive seats and receipts are required")
	}
	seenReceipts := make(map[string]struct{}, len(arm.Receipts))
	represented := make(map[string]bool, len(items))
	winners := make(map[string]developmentReceipt, len(items))
	for _, receipt := range arm.Receipts {
		if _, exists := seenReceipts[receipt.ID]; receipt.ID == "" || exists {
			return developmentArmResult{}, fmt.Errorf("receipt id %q is empty or duplicated", receipt.ID)
		}
		seenReceipts[receipt.ID] = struct{}{}
		if _, ok := items[receipt.WorkID]; !ok {
			return developmentArmResult{}, fmt.Errorf("receipt %q names unknown work item %q", receipt.ID, receipt.WorkID)
		}
		represented[receipt.WorkID] = true
		if err := validateDevelopmentReceipt(receipt); err != nil {
			return developmentArmResult{}, fmt.Errorf("receipt %q: %w", receipt.ID, err)
		}
		if receipt.Outcome == "closed" {
			winner, exists := winners[receipt.WorkID]
			if !exists || receipt.EndMS < winner.EndMS || receipt.EndMS == winner.EndMS && receipt.ID < winner.ID {
				winners[receipt.WorkID] = receipt
			}
		}
	}
	for id := range items {
		if !represented[id] {
			return developmentArmResult{}, fmt.Errorf("hidden dropped work: item %q has no dispatch receipt", id)
		}
	}

	result := developmentArmResult{Seats: arm.Seats, Attempts: len(arm.Receipts), AcceptedClosures: len(winners)}
	workers := make(map[string]struct{})
	minStart, maxEnd := arm.Receipts[0].StartMS, arm.Receipts[0].EndMS
	for _, receipt := range arm.Receipts {
		if receipt.Admitted {
			workers[receipt.Worker] = struct{}{}
		}
		if receipt.StartMS < minStart {
			minStart = receipt.StartMS
		}
		if receipt.EndMS > maxEnd {
			maxEnd = receipt.EndMS
		}
		if receipt.Attempt > 1 {
			result.RetryAttempts++
		}
		if receipt.Accounting.CollisionMS > 0 {
			result.CollisionEvents++
		}
		winner, isWinner := winners[receipt.WorkID]
		switch {
		case isWinner && winner.ID == receipt.ID:
			result.Losses.UsefulExecutionMS += receipt.Accounting.ExecutionMS
		case receipt.Outcome == "duplicate" || receipt.Outcome == "closed":
			result.DuplicateAttempts++
			result.Losses.DuplicateExecutionMS += receipt.Accounting.ExecutionMS
		default:
			result.UnverifiableAttempts++
			result.Losses.UnverifiableExecutionMS += receipt.Accounting.ExecutionMS
		}
		result.Losses.TotalWorkerMS += receipt.EndMS - receipt.StartMS
		result.Losses.QueueMS += receipt.Accounting.QueueMS
		result.Losses.CollisionMS += receipt.Accounting.CollisionMS
		result.Losses.DependencyWaitMS += receipt.Accounting.DependencyWaitMS
		result.Losses.RetryMS += receipt.Accounting.RetryMS
		result.Losses.LandingWaitMS += receipt.Accounting.LandingWaitMS
		result.Losses.VerificationMS += receipt.Accounting.VerificationMS
	}
	result.AdmittedWorkers = len(workers)
	if result.AdmittedWorkers > arm.Seats {
		return developmentArmResult{}, fmt.Errorf("admitted workers %d exceed seats %d", result.AdmittedWorkers, arm.Seats)
	}
	for id := range items {
		if _, ok := winners[id]; !ok {
			result.DroppedItems++
		}
	}
	result.MakespanMS = maxEnd - minStart
	result.WIPAreaWorkerMS = result.Losses.TotalWorkerMS
	if result.Losses.sum() != result.Losses.TotalWorkerMS {
		return developmentArmResult{}, fmt.Errorf("hidden dropped work-time: buckets sum to %d, total worker time is %d", result.Losses.sum(), result.Losses.TotalWorkerMS)
	}
	critical, err := developmentCriticalPath(items, winners)
	if err != nil {
		return developmentArmResult{}, err
	}
	result.CriticalPathMS = critical
	if result.CriticalPathMS > result.MakespanMS {
		return developmentArmResult{}, fmt.Errorf("critical path %dms exceeds makespan %dms", result.CriticalPathMS, result.MakespanMS)
	}
	return result, nil
}

func validateDevelopmentReceipt(receipt developmentReceipt) error {
	if receipt.Worker == "" || receipt.Attempt < 1 || receipt.StartMS < 0 || receipt.EndMS <= receipt.StartMS {
		return errors.New("worker, positive attempt, and ordered non-negative timestamps are required")
	}
	if !receipt.Admitted {
		return errors.New("canonical benchmark receipts must record admitted attempts")
	}
	switch receipt.Outcome {
	case "closed":
		if receipt.ClosureStamp == nil || !receipt.ClosureStamp.Independent || receipt.ClosureStamp.Kind == "" || receipt.ClosureStamp.Reference == "" {
			return errors.New("closed work is missing an independent receipt witness")
		}
	case "duplicate", "failed", "unverifiable", "dropped":
	default:
		return fmt.Errorf("unknown outcome %q", receipt.Outcome)
	}
	values := []int64{
		receipt.Accounting.ExecutionMS, receipt.Accounting.QueueMS, receipt.Accounting.CollisionMS,
		receipt.Accounting.DependencyWaitMS, receipt.Accounting.RetryMS,
		receipt.Accounting.LandingWaitMS, receipt.Accounting.VerificationMS,
	}
	var accounted int64
	for _, value := range values {
		if value < 0 {
			return errors.New("accounting buckets must be non-negative")
		}
		accounted += value
	}
	if elapsed := receipt.EndMS - receipt.StartMS; accounted != elapsed {
		return fmt.Errorf("accounting buckets total %dms, receipt elapsed time is %dms", accounted, elapsed)
	}
	return nil
}

func developmentCriticalPath(items map[string]developmentWorkItem, winners map[string]developmentReceipt) (int64, error) {
	memo := make(map[string]int64, len(items))
	var longest func(string) (int64, error)
	longest = func(id string) (int64, error) {
		if value, ok := memo[id]; ok {
			return value, nil
		}
		winner, ok := winners[id]
		if !ok {
			memo[id] = 0
			return 0, nil
		}
		var dependencyPath int64
		for _, dep := range items[id].Dependencies {
			dependencyWinner, ok := winners[dep]
			if !ok {
				return 0, fmt.Errorf("witnessed closure %q depends on unwitnessed item %q", id, dep)
			}
			if dependencyWinner.EndMS > winner.StartMS {
				return 0, fmt.Errorf("receipt %q starts before dependency %q closes", winner.ID, dependencyWinner.ID)
			}
			path, err := longest(dep)
			if err != nil {
				return 0, err
			}
			if path > dependencyPath {
				dependencyPath = path
			}
		}
		memo[id] = dependencyPath + winner.EndMS - winner.StartMS
		return memo[id], nil
	}
	var critical int64
	for id := range items {
		path, err := longest(id)
		if err != nil {
			return 0, err
		}
		if path > critical {
			critical = path
		}
	}
	return critical, nil
}

func (b developmentLossBuckets) sum() int64 {
	return b.UsefulExecutionMS + b.DuplicateExecutionMS + b.UnverifiableExecutionMS +
		b.QueueMS + b.CollisionMS + b.DependencyWaitMS + b.RetryMS + b.LandingWaitMS + b.VerificationMS
}

func (b *developmentLossBuckets) add(other developmentLossBuckets) {
	b.TotalWorkerMS += other.TotalWorkerMS
	b.UsefulExecutionMS += other.UsefulExecutionMS
	b.DuplicateExecutionMS += other.DuplicateExecutionMS
	b.UnverifiableExecutionMS += other.UnverifiableExecutionMS
	b.QueueMS += other.QueueMS
	b.CollisionMS += other.CollisionMS
	b.DependencyWaitMS += other.DependencyWaitMS
	b.RetryMS += other.RetryMS
	b.LandingWaitMS += other.LandingWaitMS
	b.VerificationMS += other.VerificationMS
}

func dominantDevelopmentLimiter(losses developmentLossBuckets) (string, int64) {
	candidates := []struct {
		name  string
		value int64
	}{
		{"duplicate_execution_ms", losses.DuplicateExecutionMS},
		{"unverifiable_execution_ms", losses.UnverifiableExecutionMS},
		{"queue_ms", losses.QueueMS},
		{"collision_ms", losses.CollisionMS},
		{"dependency_wait_ms", losses.DependencyWaitMS},
		{"retry_ms", losses.RetryMS},
		{"landing_wait_ms", losses.LandingWaitMS},
		{"verification_wait_ms", losses.VerificationMS},
	}
	winner := candidates[0]
	for _, candidate := range candidates[1:] {
		if candidate.value > winner.value {
			winner = candidate
		}
	}
	return winner.name, winner.value
}

func nextDevelopmentExperiment(limiter string) string {
	switch limiter {
	case "duplicate_execution_ms":
		return "tighten ticket intent and exact-tree admission, then replay the 16/30-seat arms"
	case "unverifiable_execution_ms":
		return "repair independent effect receipts before adding workers"
	case "queue_ms":
		return "increase admission only after shortening the queue on the same fixed workload"
	case "collision_ms":
		return "narrow exact-tree packets, then replay the 16/30-seat arms"
	case "dependency_wait_ms":
		return "split the longest dependency chain before adding seats"
	case "retry_ms":
		return "remove the leading retry cause, then replay the same receipt fixture"
	case "landing_wait_ms":
		return "shorten the serialized landing queue before increasing concurrency"
	case "verification_wait_ms":
		return "parallelize independent witness readback without widening write leases"
	default:
		return "increase fixed-workload depth before increasing seats"
	}
}

func writeDevelopmentSummary(w io.Writer, report developmentReport, selfcheck bool) error {
	if _, err := fmt.Fprintf(w, "FLEET DEVELOPMENT SCALING  workload=%s  receipts=%s\n", report.Workload.ID, shortDevelopmentDigest(report.Workload.ReceiptDigest)); err != nil {
		return err
	}
	if _, err := fmt.Fprintln(w, "SEATS ADMIT CLOSE MAKE CPATH WIP RET COLL QUEUE DEP LAND VERIFY SPEEDUP EFF"); err != nil {
		return err
	}
	for _, arm := range report.Arms {
		if _, err := fmt.Fprintf(w, "%5d %5d %5d %4d %5d %4d %3d %4d %5d %3d %4d %6d %7.3f %4.3f\n",
			arm.Seats, arm.AdmittedWorkers, arm.AcceptedClosures, arm.MakespanMS, arm.CriticalPathMS,
			arm.WIPAreaWorkerMS, arm.RetryAttempts, arm.CollisionEvents, arm.Losses.QueueMS, arm.Losses.DependencyWaitMS,
			arm.Losses.LandingWaitMS, arm.Losses.VerificationMS, arm.Speedup, arm.ParallelEfficiency); err != nil {
			return err
		}
	}
	_, limiterMS := dominantDevelopmentLimiter(sumDevelopmentLosses(report.Arms))
	if _, err := fmt.Fprintf(w, "dominant limiter: %s (%d worker-ms)\nnext experiment: %s\n", report.DominantLimiter, limiterMS, report.NextExperiment); err != nil {
		return err
	}
	if selfcheck {
		_, err := fmt.Fprintln(w, "selfcheck: PASS (reconciled accounting; superlinear, missing-witness, and hidden-drop guards fired)")
		return err
	}
	return nil
}

func sumDevelopmentLosses(arms []developmentArmResult) developmentLossBuckets {
	var total developmentLossBuckets
	for _, arm := range arms {
		total.add(arm.Losses)
	}
	return total
}

func shortDevelopmentDigest(digest string) string {
	if len(digest) <= 19 {
		return digest
	}
	return digest[:19]
}

func roundDevelopment(value float64) float64 {
	return math.Round(value*1e6) / 1e6
}

type developmentNegativeCase struct {
	name    string
	wantErr string
	fixture developmentFixture
}

func runDevelopmentNegativeChecks() error {
	for _, tc := range developmentNegativeFixtures() {
		body, err := json.Marshal(tc.fixture)
		if err != nil {
			return fmt.Errorf("selfcheck %s: encode fixture: %w", tc.name, err)
		}
		_, err = analyzeDevelopmentFixture(body)
		if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
			return fmt.Errorf("selfcheck %s: got %v, want error containing %q", tc.name, err, tc.wantErr)
		}
	}
	return nil
}

func developmentNegativeFixtures() []developmentNegativeCase {
	superlinear := minimalDevelopmentFixture()
	superlinear.Arms[1].Receipts[0].EndMS = 10
	superlinear.Arms[1].Receipts[0].Accounting.ExecutionMS = 10

	missingReceipts := minimalDevelopmentFixture()
	missingReceipts.Arms[1].Receipts = nil

	missingStamp := minimalDevelopmentFixture()
	missingStamp.Arms[1].Receipts[0].ClosureStamp = nil

	hiddenDrop := minimalDevelopmentFixture()
	hiddenDrop.Workload.Items = append(hiddenDrop.Workload.Items, developmentWorkItem{ID: "work-2"})

	return []developmentNegativeCase{
		{name: "impossible-superlinear", wantErr: "impossible superlinear accounting", fixture: superlinear},
		{name: "missing-receipts", wantErr: "positive seats and receipts are required", fixture: missingReceipts},
		{name: "missing-receipt-witness", wantErr: "missing an independent receipt witness", fixture: missingStamp},
		{name: "hidden-dropped-work", wantErr: "hidden dropped work", fixture: hiddenDrop},
	}
}

func minimalDevelopmentFixture() developmentFixture {
	durations := []int64{100, 50, 40, 30, 25}
	arms := make([]developmentArmFixture, len(developmentSeatGrid))
	for i, seats := range developmentSeatGrid {
		duration := durations[i]
		arms[i] = developmentArmFixture{Seats: seats, Receipts: []developmentReceipt{{
			ID: "receipt-" + fmt.Sprint(seats), WorkID: "work-1", Worker: "worker-1", Attempt: 1, Admitted: true,
			StartMS: 0, EndMS: duration, Outcome: "closed",
			ClosureStamp: &developmentClosureStamp{Kind: "origin-main-ancestry", Reference: "sha256:witness", Independent: true},
			Accounting:   developmentReceiptAccounting{ExecutionMS: duration},
		}}}
	}
	return developmentFixture{
		Schema:   developmentFixtureSchema,
		Workload: developmentWorkload{ID: "negative-selfcheck", Items: []developmentWorkItem{{ID: "work-1"}}},
		Arms:     arms,
	}
}
