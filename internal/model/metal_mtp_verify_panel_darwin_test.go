//go:build darwin && arm64 && cgo

package model

import (
	"reflect"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func newMetalMTPPanelFixture(t *testing.T) *Model {
	t.Helper()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	return m
}

func newPreparedMetalMTPPanelSession(t *testing.T, m *Model) (*Session, []float32) {
	t.Helper()
	s := m.NewSession()
	s.Q4K, s.MetalQ4K = true, true
	s.captureTargetHidden = true
	if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		s.Close()
		t.Fatal(err)
	}
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % m.Cfg.VocabSize
	}
	before := s.Prefill(prompt)
	if executed, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		s.Close()
		t.Fatalf("finalize resident GDN state: executed=%v err=%v", executed, err)
	}
	return s, before
}

func TestMetalMTPP4ReceiptDoesNotOverclaimIncompleteGraph(t *testing.T) {
	tests := []struct {
		name          string
		graph         metalgemm.GraphReceipt
		wantWaits     int
		wantReadbacks int
	}{
		{name: "submit failure"},
		{name: "commit without completed wait", graph: metalgemm.GraphReceipt{Committed: true}},
		{name: "completed without terminal readback", graph: metalgemm.GraphReceipt{Committed: true, CompletedWait: true}, wantWaits: 1},
		{name: "complete terminal observation", graph: metalgemm.GraphReceipt{Committed: true, CompletedWait: true, HostReadbacks: 1}, wantWaits: 1, wantReadbacks: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := qwen35MetalMTPVerifyPanelReceipt(4, 32, tt.graph, &metalQwen35MTPCheckpoint{})
			if r.TerminalWaits != tt.wantWaits || r.TerminalReadbacks != tt.wantReadbacks ||
				r.Committed != tt.graph.Committed || r.CompletedWait != tt.graph.CompletedWait {
				t.Fatalf("receipt=%+v graph=%+v", r, tt.graph)
			}
		})
	}
}

func TestMetalMTPP4ExactResidentEntrySeam(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := newMetalMTPPanelFixture(t)
	s, _ := newPreparedMetalMTPPanelSession(t, m)
	defer s.Close()
	if !s.Q4K || !s.MetalQ4K || s.qwen35HAL == nil || s.qwen35HAL.decodePath != Qwen35MetalGDNDecodeForwardPath {
		t.Fatalf("session is not the exact resident Q4K/Metal entry shape: Q4K=%v MetalQ4K=%v HAL=%+v", s.Q4K, s.MetalQ4K, s.qwen35HAL)
	}
	if qwen35MTPMetalP4Verify == nil {
		t.Fatal("exact resident Q4K/Metal P4 entry seam is not installed")
	}
	rows, checkpoint, receipt, accepted, err := qwen35MTPMetalP4Verify(s, []int{3, 5, 7, 11})
	if checkpoint != nil {
		defer checkpoint.Close()
	}
	if err != nil || !accepted || len(rows) != qwen35MetalMTPVerifyPanelTokens || checkpoint == nil {
		t.Fatalf("entry seam accepted=%v rows=%d checkpoint=%T err=%v", accepted, len(rows), checkpoint, err)
	}
	if receipt.Path != Qwen35MetalMTPVerifyPanelPath || receipt.TargetVerificationOperations != 1 || !receipt.OneOperation {
		t.Fatalf("entry seam receipt=%+v", receipt)
	}
}

func TestMetalMTPP4PanelSingleFenceTerminalPackAndDigest(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := newMetalMTPPanelFixture(t)
	s, _ := newPreparedMetalMTPPanelSession(t, m)
	defer s.Close()
	before := exactQwen38SessionDigest(t, s)
	snapshot, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	draft := []int{3, 5, 7, 11}
	result, accepted, err := s.Qwen35MetalMTPVerifyPanel(draft)
	if err != nil || !accepted {
		t.Fatalf("P4 panel accepted=%v err=%v", accepted, err)
	}
	if result.Checkpoint == nil {
		t.Fatal("P4 panel omitted explicit checkpoint owner")
	}
	r := result.Receipt
	if r.Schema != Qwen35MetalMTPVerifyPanelSchema || r.Path != Qwen35MetalMTPVerifyPanelPath ||
		r.Tokens != 4 || r.CommandBuffers != 1 || r.TerminalWaits != 1 || r.TerminalReadbacks != 1 ||
		r.IntermediateWaits != 0 || r.IntermediateReadbacks != 0 || !r.Committed || !r.CompletedWait {
		t.Fatalf("P4 graph receipt=%+v", r)
	}
	wantLinear := linearQwen35Layers(m.Cfg)
	if r.CheckpointLayers != wantLinear || r.DeviceCheckpointCopies != 2*wantLinear ||
		r.HostStateUploads != 0 || r.HostStateReadbacks != 0 || r.CheckpointBufferSwaps != 0 {
		t.Fatalf("P4 checkpoint receipt=%+v, want layers=%d", r, wantLinear)
	}
	if len(result.Logits) != 4 || len(result.RawHidden) != 4 || len(result.KV) != m.Cfg.NumLayers-wantLinear {
		t.Fatalf("P4 terminal shapes logits=%d hidden=%d kv=%d", len(result.Logits), len(result.RawHidden), len(result.KV))
	}
	for row := 0; row < 4; row++ {
		if len(result.Logits[row]) != m.Cfg.VocabSize || len(result.RawHidden[row]) != m.Cfg.HiddenSize {
			t.Fatalf("P4 row %d widths logits=%d hidden=%d", row, len(result.Logits[row]), len(result.RawHidden[row]))
		}
	}
	if r.StateDigestDomain != Qwen35MetalMTPStateDigestDomain || len(r.StateSHA256) != 64 ||
		r.StateSHA256 != qwen35MetalMTPStateDigest(r.Base, draft, joinQwen35Rows(result.RawHidden, m.Cfg.HiddenSize), result.KV) {
		t.Fatalf("P4 state digest domain=%q sha=%q", r.StateDigestDomain, r.StateSHA256)
	}
	if err := result.Checkpoint.Restore(); err != nil {
		t.Fatal(err)
	}
	if err := snapshot.Restore(s); err != nil {
		t.Fatal(err)
	}
	result.Checkpoint.Close()
	snapshot.Close()
	if after := exactQwen38SessionDigest(t, s); after != before {
		t.Fatal("explicit P4 checkpoint + PrefixSnapshot did not restore target state")
	}
}

func TestMetalMTPP4TransactionAcceptanceMatrix(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := newMetalMTPPanelFixture(t)
	draft := []int{13, 17, 19, 23}
	for accepted := 0; accepted <= len(draft); accepted++ {
		accepted := accepted
		t.Run(string(rune('0'+accepted)), func(t *testing.T) {
			baselineBuffers := metalgemm.GDNLiveBufferCount()
			target, before := newPreparedMetalMTPPanelSession(t, m)
			serial, _ := newPreparedMetalMTPPanelSession(t, m)
			defer target.Close()
			defer serial.Close()
			tx, err := beginQwen35MTPTargetTransaction(target, before)
			if err != nil {
				t.Fatal(err)
			}
			rows, err := tx.Verify(draft)
			if err != nil {
				t.Fatal(err)
			}
			if tx.receipt.Path != Qwen35MetalMTPVerifyPanelPath || !tx.receipt.OneOperation || tx.checkpoint == nil {
				t.Fatalf("Metal-first transaction receipt=%+v checkpoint=%T", tx.receipt, tx.checkpoint)
			}
			serialRows := make([][]float32, len(draft))
			for i, token := range draft {
				serialRows[i] = append([]float32(nil), serial.Step(token)...)
				assertCosineAtLeast(t, "P4 target logits", serialRows[i], rows[i], Qwen35GDNParityCosineMin)
				if argmax(serialRows[i]) != argmax(rows[i]) {
					t.Fatalf("row %d argmax=%d want %d", i, argmax(rows[i]), argmax(serialRows[i]))
				}
			}
			// Reset the serial oracle to the accepted prefix only.
			serial.Close()
			serial, _ = newPreparedMetalMTPPanelSession(t, m)
			var wantBoundary []float32
			for _, token := range draft[:accepted] {
				wantBoundary = serial.Step(token)
			}
			if accepted == 0 {
				wantBoundary = before
			}
			boundary, err := tx.Commit(accepted)
			if err != nil {
				t.Fatal(err)
			}
			if !tx.closed || tx.checkpoint != nil || tx.closeCount != 1 {
				t.Fatalf("transaction lifecycle closed=%v checkpoint=%T closes=%d", tx.closed, tx.checkpoint, tx.closeCount)
			}
			assertCosineAtLeast(t, "accepted boundary", wantBoundary, boundary, Qwen35GDNParityCosineMin)
			if accepted < len(draft) {
				if got, want := exactQwen38SessionDigest(t, target), exactQwen38SessionDigest(t, serial); got != want {
					t.Fatalf("r=%d did not restore and replay the exact accepted prefix", accepted)
				}
				if !metalMTPHiddenStateEqual(target, serial) {
					t.Fatalf("r=%d target-hidden replay differs", accepted)
				}
			} else {
				compareExactQwen38SessionState(t, accepted, serial, target)
			}
			bonus := argmax(boundary)
			wantNext, gotNext := serial.Step(bonus), target.Step(bonus)
			assertCosineAtLeast(t, "post-transaction bonus", wantNext, gotNext, Qwen35GDNParityCosineMin)
			if argmax(wantNext) != argmax(gotNext) {
				t.Fatalf("r=%d bonus argmax=%d want %d", accepted, argmax(gotNext), argmax(wantNext))
			}
			target.Close()
			serial.Close()
			if got := metalgemm.GDNLiveBufferCount(); got != baselineBuffers {
				t.Fatalf("r=%d retained GDN buffers=%d want %d", accepted, got, baselineBuffers)
			}
		})
	}
}

func metalMTPHiddenStateEqual(a, b *Session) bool {
	if !slices.Equal(a.targetHiddenTokens, b.targetHiddenTokens) || len(a.targetHidden) != len(b.targetHidden) {
		return false
	}
	for i := range a.targetHidden {
		if !slices.Equal(a.targetHidden[i], b.targetHidden[i]) {
			return false
		}
	}
	return true
}

func TestMetalMTPP4PostSubmitFailureRestoresStateWithoutFallback(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := newMetalMTPPanelFixture(t)
	baselineBuffers := metalgemm.GDNLiveBufferCount()
	s, beforeLogits := newPreparedMetalMTPPanelSession(t, m)
	defer s.Close()
	before := exactQwen38SessionDigest(t, s)
	beforeHidden := cloneTargetHidden(s.targetHidden)
	beforeTokens := append([]int(nil), s.targetHiddenTokens...)
	b := s.qwen35HAL.sequenceBackend.(*metalQwen35GDNSequenceBackend)
	b.injectMTPPanelPostSubmitFailure = true
	tx, err := beginQwen35MTPTargetTransaction(s, beforeLogits)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := tx.Verify([]int{29, 31, 37, 41})
	if err == nil || rows != nil {
		t.Fatalf("injected post-submit failure rows=%v err=%v", rows, err)
	}
	if tx.receipt.Path != Qwen35MetalMTPVerifyPanelPath || tx.receipt.TargetVerificationOperations != 1 || tx.receipt.OneOperation {
		t.Fatalf("accepted Metal failure fell through: %+v", tx.receipt)
	}
	if !tx.closed || tx.checkpoint != nil || tx.closeCount != 1 {
		t.Fatalf("failed transaction lifecycle closed=%v checkpoint=%T closes=%d", tx.closed, tx.checkpoint, tx.closeCount)
	}
	if after := exactQwen38SessionDigest(t, s); after != before || !reflect.DeepEqual(s.targetHidden, beforeHidden) || !reflect.DeepEqual(s.targetHiddenTokens, beforeTokens) {
		t.Fatal("post-submit failure did not restore GDN, KV, lineage, and target-hidden state")
	}
	s.Close()
	if got := metalgemm.GDNLiveBufferCount(); got != baselineBuffers {
		t.Fatalf("post-submit failure retained GDN buffers=%d want %d", got, baselineBuffers)
	}
}
