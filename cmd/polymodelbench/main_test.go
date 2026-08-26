package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestSelfcheck runs the three witnesses in-process (quiet) so CI proves the
// host-many / decode-one / lossless-cache-led-MTP claims, not just `go build`.
func TestSelfcheck(t *testing.T) {
	if !hostMany(true) {
		t.Error("hostMany: residency budget/pin invariant failed")
	}
	if !decodeOne(true) {
		t.Error("decodeOne: serial decode-lane invariant failed")
	}
	if !peagleParallelDepth(true) {
		t.Error("peagleParallelDepth: parallel-depth shape/accounting or token identity regressed")
	}
	if !cacheLedMTP(true) {
		t.Error("cacheLedMTP: greedy speculative decode is not lossless (bit-exact KV rollback regressed)")
	}
}

func TestPEagleShapeReceipt(t *testing.T) {
	const n, depths = 24, 4
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	draft := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64))
	r := measurePEagleShape(target, draft, bytesToIDs([]byte("parallel depth shape witness")), n, depths)
	if !r.TokenIdenticalToGreedy {
		t.Fatal("parallel-depth output is not token-identical to greedy")
	}
	if r.Engine != peagleEngine || r.TargetProvenance == "" || r.DraftSourceProvenance == "" {
		t.Fatalf("receipt lacks actual engine/provenance: %+v", r)
	}
	if r.LogicalDraftCalls != r.TargetVerifyRounds || r.SequentialDraftStepsAvoided != r.ProposedTokens {
		t.Fatalf("parallel call/cost accounting inconsistent: %+v", r)
	}
	if len(r.AcceptanceProfile) != depths || r.MeanAcceptanceLength <= 1 {
		t.Fatalf("acceptance witness incomplete: %+v", r)
	}
}
