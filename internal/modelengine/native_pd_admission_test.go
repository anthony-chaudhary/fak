package modelengine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s lane=default
func TestPDTransferCostAdmission(t *testing.T) {
	for _, tc := range []struct {
		name          string
		ceiling       time.Duration
		cost          NativePDCostEstimate
		estimatorErr  error
		nilEstimator  bool
		defaultPolicy bool
		allowed       bool
		resident      bool
	}{
		{name: "at ceiling", ceiling: 6 * time.Millisecond, cost: NativePDCostEstimate{Queue: time.Millisecond, Transfer: 2 * time.Millisecond, Prefill: 3 * time.Millisecond}, allowed: true},
		{name: "over ceiling", ceiling: 5 * time.Millisecond, cost: NativePDCostEstimate{Queue: time.Millisecond, Transfer: 2 * time.Millisecond, Prefill: 3 * time.Millisecond}},
		{name: "negative queue", ceiling: time.Second, cost: NativePDCostEstimate{Queue: -1}},
		{name: "negative transfer", ceiling: time.Second, cost: NativePDCostEstimate{Transfer: -1}},
		{name: "negative prefill", ceiling: time.Second, cost: NativePDCostEstimate{Prefill: -1}},
		{name: "duration overflow", ceiling: time.Duration(1<<63 - 1), cost: NativePDCostEstimate{Queue: time.Duration(1<<63 - 1), Transfer: 1}},
		{name: "zero ceiling", cost: NativePDCostEstimate{}},
		{name: "negative ceiling", ceiling: -1},
		{name: "missing estimator", ceiling: time.Second, nilEstimator: true},
		{name: "estimator unavailable", ceiling: time.Second, estimatorErr: errors.New("uncalibrated")},
		{name: "zero cost", ceiling: time.Second, allowed: true},
		{name: "default unchanged", defaultPolicy: true, allowed: true},
		{name: "resident route still budgets full prefill", ceiling: time.Millisecond, cost: NativePDCostEstimate{Prefill: 2 * time.Millisecond}, resident: true},
		{name: "resident route accepted at ceiling", ceiling: time.Millisecond, cost: NativePDCostEstimate{Prefill: time.Millisecond}, resident: true, allowed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := model.NewSynthetic(SyntheticConfig())
			sessions, calls := 0, 0
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var estimator NativePDCostEstimator = func(got context.Context, tokens int, worker string) (NativePDCostEstimate, error) {
				calls++
				if got != ctx || tokens != 3 || worker != "decode-0" {
					t.Errorf("estimator inputs: tokens=%d worker=%q context matches=%v", tokens, worker, got == ctx)
				}
				return tc.cost, tc.estimatorErr
			}
			if tc.nilEstimator {
				estimator = nil
			}
			opts := []NativePDOption{WithNativePDSessionHook(func(*model.Session) { sessions++ })}
			if !tc.defaultPolicy {
				opts = append(opts, WithNativePDCostAdmission(tc.ceiling, estimator))
			}
			index := newNativePDLocalResidency()
			if tc.resident {
				index.Observe("decode-0", []string{NativePDPrefixKey([]int{3, 7, 11})})
			}
			cluster := NewNativePDClusterWithResidency(m, 1, index, opts...)
			defer cluster.Close()
			admit, err := cluster.Admit(ctx, NativePDRequest{Prompt: []int{3, 7, 11}, ModelID: "synthetic-test", TokenizerID: "byte", Lease: "lease-14253", Taint: abi.TaintTrusted, Scope: abi.ScopeFleet, AdmissionVerdict: cachemeta.AdmissionAllow, AdmittedBy: "admission-test"})
			if tc.resident && cluster.localityHits != 1 {
				t.Fatal("fixture did not select a resident route")
			}
			if tc.allowed {
				if err != nil {
					t.Fatal(err)
				}
				drainPDRequest(t, admit.Request)
				if sessions == 0 || cluster.PrefillWorker().Stats().PrefillRequests != 1 {
					t.Fatal("accepted request did not use native prefill")
				}
			} else {
				if !errors.Is(err, ErrNativePDCostAdmission) {
					t.Fatalf("error=%v, want typed cost refusal", err)
				}
				if tc.estimatorErr != nil && !errors.Is(err, tc.estimatorErr) {
					t.Fatalf("estimator error identity lost: %v", err)
				}
				if sessions != 0 || cluster.PrefillWorker().Stats().PrefillRequests != 0 {
					t.Fatal("rejected request allocated a session or ran prefill")
				}
				worker, _ := cluster.DecodeWorker("decode-0")
				if worker.Stats().DecodeImports != 0 || worker.Stats().SchedulerAdmits != 0 {
					t.Fatal("rejected request reached decode admission")
				}
			}
			wantCalls := 1
			if tc.defaultPolicy || tc.nilEstimator || tc.ceiling <= 0 {
				wantCalls = 0
			}
			if calls != wantCalls {
				t.Fatalf("estimator calls=%d want=%d", calls, wantCalls)
			}
		})
	}
}
