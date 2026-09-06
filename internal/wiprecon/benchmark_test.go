package wiprecon

import (
	"fmt"
	"path/filepath"
	"testing"
)

var (
	sinkDecision      Decision
	sinkAdoptDecision AdoptDecision
	sinkDecisions     []Decision
	sinkRows          []ReclaimRow
	sinkReceipt       Receipt
	sinkString        string
)

// BenchmarkDecide measures per-candidate classification across decision rules.
func BenchmarkDecide(b *testing.B) {
	cands := []Candidate{
		{Session: "s-live", Owner: OwnerLive, Landed: true, Applies: true},
		{Session: "s-landed", Owner: OwnerCrashed, Landed: true},
		{Session: "s-reclaim", Owner: OwnerCrashed, Landed: false, Applies: true},
		{Session: "s-quarantine", Owner: OwnerCrashed, Landed: false, Applies: false},
		{Session: "s-diverged", Owner: OwnerCrashed, Applies: true, DivergedPaths: 2},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkDecision = Decide(cands[i%len(cands)])
	}
}

// BenchmarkReconcile measures batch candidate reconciliation and deterministic sort.
func BenchmarkReconcile(b *testing.B) {
	for _, size := range []int{10, 50, 200} {
		cands := make([]Candidate, size)
		for i := 0; i < size; i++ {
			cands[i] = Candidate{
				Session: fmt.Sprintf("session-%04d", (i*37)%size),
				Owner:   OwnerCrashed,
				Landed:  i%3 == 0,
				Applies: i%3 == 1,
			}
		}
		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkDecisions = Reconcile(cands)
			}
		})
	}
}

// BenchmarkRankReclaim measures actionable-first and base-drift ordering.
func BenchmarkRankReclaim(b *testing.B) {
	for _, size := range []int{10, 50, 200} {
		rows := make([]ReclaimRow, size)
		for i := 0; i < size; i++ {
			rows[i] = ReclaimRow{
				Session:       fmt.Sprintf("sess-%04d", (i*23)%size),
				TrunkDistance: (i * 17) % 500,
				AgeHours:      float64((i * 13) % 200),
			}
			if i%4 == 0 {
				rows[i].AdoptedBy = "peer"
			}
			if i%7 == 0 {
				rows[i].TrunkDistance = DriftUnknown
			}
		}
		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkRows = RankReclaim(rows)
			}
		})
	}
}

// BenchmarkUnownedReclaim measures worklist filtering to claimable rows.
func BenchmarkUnownedReclaim(b *testing.B) {
	rows := make([]ReclaimRow, 100)
	for i := 0; i < len(rows); i++ {
		rows[i] = ReclaimRow{
			Session:   fmt.Sprintf("sess-%04d", i),
			AdoptedBy: "",
		}
		if i%3 == 0 {
			rows[i].AdoptedBy = "peer"
		} else if i%3 == 1 {
			rows[i].AdoptedBy = "self"
			rows[i].AdoptedMine = true
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRows = UnownedReclaim(rows)
	}
}

// BenchmarkDecideAdopt measures ownership decisions across grant, held, resume, and takeover.
func BenchmarkDecideAdopt(b *testing.B) {
	reqGrant := AdoptRequest{
		Session:       "crashed-1",
		Action:        ActReclaim,
		CheckpointRef: "refs/fak/wip/crashed-1",
		CheckpointSHA: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		DeltaDigest:   "sha256:digest",
		Successor:     "rescuer-1",
		Now:           1_000_000,
		TTLSeconds:    900,
	}
	incumbent := Receipt{
		Session:       "crashed-1",
		CheckpointRef: "refs/fak/wip/crashed-1",
		CheckpointSHA: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Successor:     "rescuer-prior",
		Phase:         PhaseMaterialized,
		AdoptedAt:     999_000,
		RenewedAt:     999_500,
		TTLSeconds:    900,
		Attempt:       1,
	}
	reqResume := reqGrant
	reqResume.Successor = "rescuer-prior"

	reqTakeover := reqGrant
	reqTakeover.Now = 1_005_000

	b.Run("Grant", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkAdoptDecision = DecideAdopt(nil, reqGrant)
		}
	})

	b.Run("Held", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkAdoptDecision = DecideAdopt(&incumbent, reqGrant)
		}
	})

	b.Run("Resume", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkAdoptDecision = DecideAdopt(&incumbent, reqResume)
		}
	})

	b.Run("Takeover", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkAdoptDecision = DecideAdopt(&incumbent, reqTakeover)
		}
	})
}

// BenchmarkApplyAdopt measures mutating and producing updated adoption receipts.
func BenchmarkApplyAdopt(b *testing.B) {
	req := AdoptRequest{
		Session:       "crashed-1",
		Action:        ActReclaim,
		CheckpointRef: "refs/fak/wip/crashed-1",
		CheckpointSHA: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		DeltaDigest:   "sha256:digest",
		Successor:     "rescuer-1",
		Now:           1_000_000,
		TTLSeconds:    900,
	}
	cur := Receipt{
		Session:       "crashed-1",
		CheckpointRef: "refs/fak/wip/crashed-1",
		CheckpointSHA: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Successor:     "rescuer-prior",
		Phase:         PhaseMaterialized,
		AdoptedAt:     990_000,
		RenewedAt:     990_000,
		TTLSeconds:    900,
		Attempt:       1,
	}

	b.Run("Grant", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkReceipt, _ = ApplyAdopt(nil, req, AdoptGrant)
		}
	})

	b.Run("Resume", func(b *testing.B) {
		resumeReq := req
		resumeReq.Successor = cur.Successor
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkReceipt, _ = ApplyAdopt(&cur, resumeReq, AdoptResume)
		}
	})

	b.Run("Takeover", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkReceipt, _ = ApplyAdopt(&cur, req, AdoptTakeover)
		}
	})
}

// BenchmarkReceiptEncodeDecode measures serialization and parsing of durable receipts.
func BenchmarkReceiptEncodeDecode(b *testing.B) {
	r := Receipt{
		Session:       "crashed-session",
		CheckpointRef: "refs/fak/wip/crashed-session",
		CheckpointSHA: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		DeltaDigest:   "sha256:deadbeef",
		Successor:     "rescuer-alpha",
		Phase:         PhaseMaterialized,
		Target:        filepath.Join(b.TempDir(), "target"),
		AdoptedAt:     1_700_000_000,
		RenewedAt:     1_700_000_100,
		TTLSeconds:    900,
		Attempt:       2,
		Audit: []AuditEvent{
			{At: 1_700_000_000, Event: EventAdopted, Actor: "rescuer-alpha"},
			{At: 1_700_000_050, Event: EventMaterialized, Actor: "rescuer-alpha", Detail: "4 files"},
		},
	}
	encoded, err := EncodeReceipt(r)
	if err != nil {
		b.Fatalf("EncodeReceipt: %v", err)
	}

	b.Run("Encode", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkString, _ = EncodeReceipt(r)
		}
	})

	b.Run("Decode", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sinkReceipt, _ = DecodeReceipt(encoded)
		}
	})
}

// BenchmarkMarkPhase measures phase advancement and audit trail append.
func BenchmarkMarkPhase(b *testing.B) {
	r := Receipt{
		Session:       "crashed-session",
		CheckpointRef: "refs/fak/wip/crashed-session",
		CheckpointSHA: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Successor:     "rescuer-alpha",
		Phase:         PhaseAdopted,
		AdoptedAt:     1_700_000_000,
		RenewedAt:     1_700_000_000,
		TTLSeconds:    900,
		Attempt:       1,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkReceipt = MarkPhase(r, PhaseMaterialized, int64(1_700_000_000+i), EventMaterialized, "worktree prepared")
	}
}
