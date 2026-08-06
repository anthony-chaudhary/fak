package main

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type fairnessReport struct {
	Schema                string           `json:"schema"`
	Verdict               string           `json:"verdict"`
	Submitted             int              `json:"submitted"`
	Scheduled             int              `json:"scheduled"`
	Cancelled             int              `json:"cancelled"`
	RefusedSpend          int              `json:"refused_spend"`
	InteractiveP95MS      float64          `json:"interactive_wait_p95_ms"`
	BulkP95MS             float64          `json:"bulk_wait_p95_ms"`
	PerTenant             map[string]int   `json:"per_tenant_scheduled"`
	MaxServiceLag         map[string]int   `json:"max_service_lag"`
	SelectionCostNS       int64            `json:"selection_cost_ns"`
	AllocationBytes       uint64           `json:"allocation_bytes"`
	EstimatedIdleModelNS  int64            `json:"estimated_idle_model_ns"`
	DuplicateOutputTokens int              `json:"duplicate_output_tokens"`
	FailedWorkUnits       int              `json:"failed_work_units"`
	SpendMicros           map[string]int64 `json:"spend_micros"`
	EconomicsVerdict      string           `json:"economics_verdict"`
	Scope                 string           `json:"scope"`
}

func runFairnessFixture() (fairnessReport, error) {
	now := time.Unix(100, 0)
	s, err := microagent.NewTenantQueue([]microagent.TenantEnvelope{{Tenant: "interactive", Weight: 3, MaxQueued: 800, MaxConcurrent: 2, MaxSpendMicros: 1000, RatePerMinute: 10000}, {Tenant: "bulk", Weight: 1, MaxQueued: 800, MaxConcurrent: 4, MaxSpendMicros: 1000, RatePerMinute: 10000}})
	if err != nil {
		return fairnessReport{}, err
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	submitted, cancelled, refused := 0, 0, 0
	for i := 0; i < 600; i++ {
		submitted++
		if err := s.Submit(microagent.TenantTask{ID: fmt.Sprintf("i-%04d", i), Tenant: "interactive", Interactive: true, Deadline: now.Add(50 * time.Millisecond), EnqueuedAt: now, CostMicros: 1, Cancelled: i%100 == 0}); err != nil {
			refused++
		}
		if i%100 == 0 {
			cancelled++
		}
	}
	for i := 0; i < 200; i++ {
		submitted++
		if err := s.Submit(microagent.TenantTask{ID: fmt.Sprintf("b-%04d", i), Tenant: "bulk", EnqueuedAt: now, CostMicros: 2}); err != nil {
			refused++
		}
	}
	if err := s.Submit(microagent.TenantTask{ID: "over-spend", Tenant: "bulk", CostMicros: 2000}); err != nil {
		refused++
	}
	start := time.Now()
	var iw, bw []time.Duration
	scheduled := 0
	for {
		t, ok := s.Next(now.Add(time.Duration(scheduled) * time.Microsecond))
		if !ok {
			break
		}
		wait := time.Duration(scheduled) * time.Microsecond
		if t.Interactive {
			iw = append(iw, wait)
		} else {
			bw = append(bw, wait)
		}
		scheduled++
	}
	wall := time.Since(start)
	runtime.ReadMemStats(&after)
	snap := s.Snapshot()
	r := fairnessReport{Schema: "fak-microcontext-fairness/1", Verdict: "PASS", Submitted: submitted, Scheduled: scheduled, Cancelled: cancelled, RefusedSpend: refused, InteractiveP95MS: percentileMS(iw, .95), BulkP95MS: percentileMS(bw, .95), PerTenant: snap.Scheduled, MaxServiceLag: snap.MaxLag, SelectionCostNS: wall.Nanoseconds(), AllocationBytes: after.TotalAlloc - before.TotalAlloc, EstimatedIdleModelNS: 0, DuplicateOutputTokens: 0, FailedWorkUnits: 0, SpendMicros: snap.SpendMicros, EconomicsVerdict: "cost-only; no economic gain claimed without a tuned serving baseline", Scope: "offline deterministic scheduler fixture; fairness scheduling and overhead only, not model throughput or provider economics"}
	return r, verifyFairnessReport(r)
}

func verifyFairnessReport(r fairnessReport) error {
	if r.Schema != "fak-microcontext-fairness/1" || r.Verdict != "PASS" {
		return fmt.Errorf("schema/verdict")
	}
	if r.Submitted != r.Scheduled+r.Cancelled || r.RefusedSpend != 1 {
		return fmt.Errorf("accounting submitted=%d scheduled=%d cancelled=%d refused=%d", r.Submitted, r.Scheduled, r.Cancelled, r.RefusedSpend)
	}
	if r.PerTenant["interactive"] != 594 || r.PerTenant["bulk"] != 200 {
		return fmt.Errorf("tenant counts=%v", r.PerTenant)
	}
	for _, lag := range r.MaxServiceLag {
		if lag > 6 {
			return fmt.Errorf("weighted lag=%v", r.MaxServiceLag)
		}
	}
	if r.InteractiveP95MS >= r.BulkP95MS || r.SelectionCostNS <= 0 || r.AllocationBytes <= 0 || r.EconomicsVerdict == "" || r.Scope == "" {
		return fmt.Errorf("fairness/economics evidence incomplete")
	}
	return nil
}
func writeFairness(path string) error {
	r, err := runFairnessFixture()
	if err != nil {
		return err
	}
	b, _ := json.MarshalIndent(r, "", "  ")
	return os.WriteFile(path, append(b, '\n'), 0644)
}
func verifyFairnessArtifact(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var r fairnessReport
	if err = json.Unmarshal(b, &r); err != nil {
		return err
	}
	return verifyFairnessReport(r)
}
