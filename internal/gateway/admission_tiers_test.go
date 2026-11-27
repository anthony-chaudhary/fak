package gateway

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

// fak-test:runtime fast est=10ms lane=default
func TestTierAdmissionNoHoLBlocking(t *testing.T) {
	plan, err := kvbudget.AutotuneTiersFromTokenBudget(384*1024, kvbudget.ProfileUltracode)
	if err != nil {
		t.Fatalf("AutotuneTiersFromTokenBudget: %v", err)
	}
	// Cap L1 master concurrency at 1 slot so a second L1 orchestrator queues
	// without cannibalizing L2/L3 headroom.
	plan.Specs[0].MaxSlots = 1
	pool, err := kvbudget.NewTierPool(plan.TotalTokenCapacity, plan.Specs)
	if err != nil {
		t.Fatalf("NewTierPool: %v", err)
	}

	ctl := NewAdmissionController(DefaultAdmissionPolicy())
	ctl.SetTierPool(pool)

	// 1. Admit 1x128k L1 master slot.
	v1, masterLease1, _ := ctl.OfferTiered(SeqRequest{
		TraceID: "master-1",
		Tokens:  131072,
	})
	if v1 != VerdictAdmitted {
		t.Fatalf("master-1 verdict = %v, want %v", v1, VerdictAdmitted)
	}

	// 2. Offer 2nd L1 master request -> queued because L1 is full.
	v2, _, verdict2 := ctl.OfferTiered(SeqRequest{
		TraceID: "master-2",
		Tokens:  131072,
	})
	if v2 != VerdictQueued {
		t.Fatalf("master-2 verdict = %v, want %v", v2, VerdictQueued)
	}
	advice := BuildTierAdvice(verdict2, 131072)
	if advice.Code != "DOWNGRADE_TO_WORKER" {
		t.Fatalf("master-2 advice.Code = %q, want DOWNGRADE_TO_WORKER", advice.Code)
	}

	// 3. Offer 4x32k L2 worker requests: must NOT be HoL-blocked by master-2!
	workerLeases := make([]kvbudget.TierLease, 0, 4)
	for i := 0; i < 4; i++ {
		vw, wLease, _ := ctl.OfferTiered(SeqRequest{
			TraceID: fmt.Sprintf("worker-%d", i),
			Tokens:  32768,
		})
		if vw != VerdictAdmitted {
			t.Fatalf("worker-%d verdict = %v, want %v (HoL blocked by master-2)", i, vw, VerdictAdmitted)
		}
		workerLeases = append(workerLeases, wLease)
	}

	// 4. Offer 32x4k L3 micro requests: must NOT be HoL-blocked by master-2!
	microLeases := make([]kvbudget.TierLease, 0, 32)
	for i := 0; i < 32; i++ {
		vm, mLease, _ := ctl.OfferTiered(SeqRequest{
			TraceID: fmt.Sprintf("micro-%d", i),
			Tokens:  4096,
		})
		if vm != VerdictAdmitted {
			t.Fatalf("micro-%d verdict = %v, want %v (HoL blocked by master-2)", i, vm, VerdictAdmitted)
		}
		microLeases = append(microLeases, mLease)
	}

	// 5. ScheduleTiered while master-1 is still active leaves master-2 queued.
	if promoted := ctl.ScheduleTiered(); len(promoted) != 0 {
		t.Fatalf("ScheduleTiered promoted %d requests while master-1 active, want 0", len(promoted))
	}

	// 6. Complete master-1 -> ScheduleTiered promotes master-2.
	ctl.CompleteTiered(masterLease1)
	promoted := ctl.ScheduleTiered()
	if len(promoted) != 1 || promoted[0].Request.TraceID != "master-2" {
		t.Fatalf("ScheduleTiered after master-1 release = %+v, want [master-2]", promoted)
	}

	// Clean up remaining leases (exercise Complete cleanup on worker-0).
	if !ctl.Complete("worker-0") {
		t.Fatal("ctl.Complete(worker-0) = false, want true")
	}
	ctl.CompleteTiered(promoted[0].Lease)
	for _, l := range workerLeases[1:] {
		ctl.CompleteTiered(l)
	}
	for _, l := range microLeases {
		ctl.CompleteTiered(l)
	}

	snap := ctl.TierPoolSnapshot()
	if snap.FreeTokens != snap.TotalTokens {
		t.Fatalf("FreeTokens = %d, TotalTokens = %d after releasing all leases", snap.FreeTokens, snap.TotalTokens)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestTierAdmissionNilPoolPreservesLegacyBehavior(t *testing.T) {
	ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, TokenBudget: 1024, MaxWaiting: 4})
	if snap := ctl.TierPoolSnapshot(); snap.Enabled {
		t.Fatalf("expected TierPoolSnapshot.Enabled=false when nil pool")
	}
	v1, _, _ := ctl.OfferTiered(SeqRequest{TraceID: "req-1", Tokens: 512})
	if v1 != VerdictAdmitted {
		t.Fatalf("req-1 verdict = %v, want %v", v1, VerdictAdmitted)
	}
	v2, _, _ := ctl.OfferTiered(SeqRequest{TraceID: "req-2", Tokens: 512})
	if v2 != VerdictQueued {
		t.Fatalf("req-2 verdict = %v, want %v", v2, VerdictQueued)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBuildTierAdviceCodes(t *testing.T) {
	cases := []struct {
		name    string
		verdict kvbudget.TierVerdict
		tokens  int
		want    string
	}{
		{
			name: "downgrade to worker",
			verdict: kvbudget.TierVerdict{
				Tier:       kvbudget.TierMaster,
				AdviceCode: "TIER_EXHAUSTED_DOWNGRADE_AVAILABLE",
				AvailableTiers: []kvbudget.TierStatus{
					{Name: kvbudget.TierMaster, MaxContext: 131072, AvailableNow: 0},
					{Name: kvbudget.TierWorker, MaxContext: 32768, AvailableNow: 4},
					{Name: kvbudget.TierMicro, MaxContext: 4096, AvailableNow: 32},
				},
			},
			tokens: 131072,
			want:   "DOWNGRADE_TO_WORKER",
		},
		{
			name: "split to micro",
			verdict: kvbudget.TierVerdict{
				Tier:       kvbudget.TierWorker,
				AdviceCode: "TIER_EXHAUSTED_DOWNGRADE_AVAILABLE",
				AvailableTiers: []kvbudget.TierStatus{
					{Name: kvbudget.TierMaster, MaxContext: 131072, AvailableNow: 0},
					{Name: kvbudget.TierWorker, MaxContext: 32768, AvailableNow: 0},
					{Name: kvbudget.TierMicro, MaxContext: 4096, AvailableNow: 16},
				},
			},
			tokens: 32768,
			want:   "SPLIT_TO_MICRO",
		},
		{
			name: "compact context",
			verdict: kvbudget.TierVerdict{
				Tier:       kvbudget.TierMicro,
				CapTokens:  4096,
				AdviceCode: "EXCEEDS_TIER_MAX_CONTEXT",
				AvailableTiers: []kvbudget.TierStatus{
					{Name: kvbudget.TierMicro, MaxContext: 4096, AvailableNow: 0},
				},
			},
			tokens: 8192,
			want:   "COMPACT_CONTEXT",
		},
		{
			name: "wait for slot",
			verdict: kvbudget.TierVerdict{
				Tier:       kvbudget.TierMicro,
				AdviceCode: "POOL_EXHAUSTED",
				AvailableTiers: []kvbudget.TierStatus{
					{Name: kvbudget.TierMaster, MaxContext: 131072, AvailableNow: 0},
					{Name: kvbudget.TierWorker, MaxContext: 32768, AvailableNow: 0},
					{Name: kvbudget.TierMicro, MaxContext: 4096, AvailableNow: 0},
				},
			},
			tokens: 4096,
			want:   "WAIT_FOR_SLOT",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := BuildTierAdvice(tc.verdict, tc.tokens)
			if got.Code != tc.want {
				t.Fatalf("BuildTierAdvice().Code = %q, want %q", got.Code, tc.want)
			}
		})
	}
}
