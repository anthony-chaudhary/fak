package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// writeServeStreamedV41ReserveGGUF serializes a DeepSeek-V4.1-shaped (deepseek41) header whose
// routed-expert slabs (experts) and dense side (denseRows x 256 Q2_K attention rows) are DECLARED at
// aegis scale but never materialized: only the header is written. The planner and the checkpoint
// tier's stageability probe read tensor geometry from the header alone, so the fit math runs at real
// scale (hundreds of GiB of routed slabs) without the test writing a payload.
func writeServeStreamedV41ReserveGGUF(t *testing.T, experts, denseRows uint64) string {
	t.Helper()
	type tensor struct {
		name  string
		dims  []uint64
		typ   ggufload.TensorType
		bytes uint64
	}
	q2 := func(rows uint64) uint64 { return rows * 84 } // one 256-wide Q2_K block per row
	tensors := []tensor{
		{"token_embd.weight", []uint64{256, 1}, ggufload.TensorQ2_K, q2(1)},
		{"output.weight", []uint64{256, 1}, ggufload.TensorQ6_K, 210},
		{"output_norm.weight", []uint64{256}, ggufload.TensorF32, 1024},
		{"blk.0.attn_q_a.weight", []uint64{256, 256}, ggufload.TensorQ2_K, q2(256)},
		{"blk.0.attn_q_b.weight", []uint64{256, denseRows}, ggufload.TensorQ2_K, q2(denseRows)},
		{"blk.0.attn_kv.weight", []uint64{256, 32}, ggufload.TensorQ2_K, q2(32)},
		{"blk.0.attn_output_a.weight", []uint64{256, 256}, ggufload.TensorQ2_K, q2(256)},
		{"blk.0.attn_output_b.weight", []uint64{256, 256}, ggufload.TensorQ2_K, q2(256)},
		{"blk.0.attn_sinks.weight", []uint64{8}, ggufload.TensorF32, 32},
		{"blk.0.ffn_gate_shexp.weight", []uint64{256, 256}, ggufload.TensorQ2_K, q2(256)},
		{"blk.0.ffn_up_shexp.weight", []uint64{256, 256}, ggufload.TensorQ2_K, q2(256)},
		{"blk.0.ffn_down_shexp.weight", []uint64{256, 256}, ggufload.TensorQ2_K, q2(256)},
		{"blk.0.ffn_gate_exps.weight", []uint64{256, 256, experts}, ggufload.TensorQ2_K, q2(256 * experts)},
		{"blk.0.ffn_up_exps.weight", []uint64{256, 256, experts}, ggufload.TensorQ2_K, q2(256 * experts)},
		{"blk.0.ffn_down_exps.weight", []uint64{256, 256, experts}, ggufload.TensorQ2_K, q2(256 * experts)},
	}
	keys := []struct {
		key   string
		value uint64
	}{
		{"embedding_length", 256}, {"block_count", 1}, {"attention.head_count", 8},
		{"attention.head_count_kv", 1}, {"attention.key_length", 32}, {"feed_forward_length", 256},
		{"context_length", 2048}, {"expert_count", experts}, {"expert_used_count", 2},
		{"expert_feed_forward_length", 256}, {"attention.q_lora_rank", 256}, {"attention.kv_lora_rank", 32},
		{"attention.qk_nope_head_dim", 16}, {"attention.qk_rope_head_dim", 16}, {"attention.value_length_mla", 16},
		{"attention.output_group_count", 1}, {"attention.output_lora_rank", 256},
	}
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, uint64(len(tensors)), uint64(3+len(keys)))
	writeKVStringForTest(&b, "general.architecture", "deepseek41")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVFloat32ForTest(&b, "deepseek41.attention.layer_norm_rms_epsilon", 1e-6)
	for _, key := range keys {
		writeKVUint64ForTest(&b, "deepseek41."+key.key, key.value)
	}
	var offset uint64
	for _, tn := range tensors {
		writeTensorInfoForTest(&b, tn.name, tn.dims, uint32(tn.typ), offset)
		offset = (offset + tn.bytes + 31) &^ 31
	}
	padToAlignmentForTest(&b, 32)
	path := filepath.Join(t.TempDir(), "v41-streamed-reserve.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// serveStreamedV41AegisFixture opens the aegis-shaped V4.1 header: ~183 GiB of declared routed
// slabs (the DeepSeek-V4.1 Flash Q2_K set never fits a 30.4 GiB host, so the streamed arm is always
// selected) and a ~10 GiB dense side (larger than any combined dense bound, so the bounded
// gguf-host-dense-streamed row binds exactly as on aegis).
func serveStreamedV41AegisFixture(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	t.Setenv("FAK_STREAM_Q4K", "1")
	ws, err := ggufload.OpenWeights(writeServeStreamedV41ReserveGGUF(t, 3_050_000, 128_000_000))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ws.Close() })
	return ws
}

func serveStreamedRowBytes(plan compute.MemoryPlan, detail string) (int64, int) {
	var total int64
	rows := 0
	for _, d := range plan {
		if d.Detail == detail {
			total += d.Bytes
			rows++
		}
	}
	return total, rows
}

// G5 (fak-private ticket ds41-streamed-resident-margin-omits-transient-20261121): on aegis
// (30.4 GiB host, believed-free ~22.4 GiB) every fit-commit back-off rung refused FitTooBig by ~1 GiB:
// expert 21,656,830,095 B + dense 2,165,683,009 B + v41 transient 1,294,860,288 B = 23.39 GiB against
// 22.38 GiB. The expert bound (0.9*avail) and the combined dense bound (0.9*(avail-expert)) were taken
// from the GROSS host fit, so they alone claimed ~0.99*avail before the transient row was appended.
// This pins that the streamed plan's host-scoped demand (expert + dense + transient) lands within the
// host budget that judges it, at the aegis readings, on a Strix-Halo-sized control, and on every
// back-off fraction serve_fit_commit.go walks.
// fak-test:justify why=regression when=changed:cmd/fak/serve_model_fit_streamed.go
// fak-test:runtime fast est=50ms lane=default
func TestServeStreamedV41HostRowsFitAfterTransientReserve(t *testing.T) {
	const gib = float64(1 << 30)
	ws := serveStreamedV41AegisFixture(t)
	cfg, err := ws.File.Config()
	if err != nil || !cfg.IsDeepSeekV41() || !serveStreamedExpertsCapable(ws) {
		t.Fatalf("fixture must be a stageable V4.1 artifact: cfg err=%v", err)
	}
	transient := cfg.V41TransientResidentBytes(serveV41TransientCacheBudget(ws))
	if transient <= 0 {
		t.Fatal("fixture must carry a positive V4.1 transient charge")
	}
	for _, availGiB := range []float64{22.48, 22.38, 48.6} {
		for _, frac := range []float64{1.0, 0.99, 0.98, 0.96, 0.92, 0.84, 0.68} {
			fit := serveFitBudget{Base: int64(availGiB * frac * gib), Headroom: serveGGUFHostHeadroom}
			avail := fit.avail()
			plan, streamed, err := serveStreamedCPUOffloadPlanForPool(ws, nil, 1, 16, fit, false)
			if err != nil || !streamed {
				t.Fatalf("avail=%.2fGiB x%.2f: streamed=%v err=%v, want the streamed arm", availGiB, frac, streamed, err)
			}
			expert, _ := serveStreamedRowBytes(plan, "gguf-host-expert-offload-streamed")
			dense, _ := serveStreamedRowBytes(plan, "gguf-host-dense-streamed")
			charged, rows := serveStreamedRowBytes(plan, "v41-forward-transient-scratch")
			if expert <= 0 || dense <= 0 || rows != 1 || charged != transient {
				t.Fatalf("avail=%.2fGiB x%.2f: expert=%d dense=%d transient=%d (rows %d), want positive bounded rows and one %d-byte transient",
					availGiB, frac, expert, dense, charged, rows, transient)
			}
			if host := plan.HostTotal(); host > avail {
				t.Fatalf("avail=%.2fGiB x%.2f: host demand %d (expert %d + dense %d + transient %d) exceeds host budget %d by %d bytes",
					availGiB, frac, host, expert, dense, charged, avail, host-avail)
			}
		}
	}
}

// G5 plan-json/fit-commit parity: the context-sizing arm (serveCPUOffloadSizingStreamedPlan, the
// weights buildServeSizingArtifact reports) must carry the v41-forward-transient-scratch row the
// fit-commit plan charges, with the SAME host-scoped total, and the load path's append must not
// charge it a second time. Before the fix the sizing plan omitted the row, so plan-json said "fits"
// while fit-commit refused.
// fak-test:justify why=regression when=changed:cmd/fak/serve_model_fit.go
// fak-test:runtime fast est=50ms lane=default
func TestServeStreamedV41SizingPlanChargesTransientLikeFitCommit(t *testing.T) {
	ws := serveStreamedV41AegisFixture(t)
	aegisAvailGiB := 22.48
	hostFit := serveFitBudget{Base: int64(aegisAvailGiB * float64(1<<30)), Headroom: serveGGUFHostHeadroom}
	sizing, streamed := serveCPUOffloadSizingStreamedPlan(ws, nil, 1, hostFit)
	if !streamed {
		t.Fatal("sizing arm did not select the streamed plan")
	}
	if _, rows := serveStreamedRowBytes(sizing, "v41-forward-transient-scratch"); rows != 1 {
		t.Fatalf("sizing plan carries %d v41-forward-transient-scratch rows, want 1 (plan-json must see the fit-commit host demand)", rows)
	}
	commit, streamed, err := serveStreamedCPUOffloadPlanForAperture(ws, nil, 1, 16, hostFit, hostFit, false, false)
	if err != nil || !streamed {
		t.Fatalf("fit-commit plan: streamed=%v err=%v", streamed, err)
	}
	if got, want := sizing.HostTotal(), commit.HostTotal(); got != want {
		t.Fatalf("sizing host total %d != fit-commit host total %d", got, want)
	}
	cfg, _ := ws.File.Config()
	again := appendServeGGUFDevicePlan(ws, nil, sizing, 16, hostFit)
	if _, rows := serveStreamedRowBytes(again, "v41-forward-transient-scratch"); rows != 1 {
		t.Fatalf("appendServeGGUFDevicePlan over the sizing plan charged %d transient rows, want 1", rows)
	}
	if _, rows := serveStreamedRowBytes(appendServeV41TransientCharge(sizing, cfg, serveV41TransientCacheBudget(ws)), "v41-forward-transient-scratch"); rows != 1 {
		t.Fatalf("appendServeV41TransientCharge is not idempotent: %d rows", rows)
	}
}

// A non-V4.1 artifact has no transient reserve, so its streamed bound stays the historical gross
// (1-margin)*avail derivation byte-for-byte.
// fak-test:justify why=regression when=changed:cmd/fak/serve_model_fit_streamed.go
// fak-test:runtime fast est=20ms lane=default
func TestServeStreamedNonV41BoundUnchangedByReserve(t *testing.T) {
	path := writeServeStreamedScaledSynthGGUF(t, "nonv41-reserve.gguf", 64, 256)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	fit := serveFitBudget{Base: 1 << 20}
	_, streamed, selected, err := serveStreamedCPUOffloadCorePlan(ws, nil, 1, 0, fit, false, false)
	if err != nil || !selected {
		t.Fatalf("control did not select the streamed arm: selected=%v err=%v", selected, err)
	}
	if _, rows := serveStreamedRowBytes(streamed, "v41-forward-transient-scratch"); rows != 0 {
		t.Fatal("non-V4.1 plan gained a V4.1 transient row")
	}
	ok, bound, err := serveStreamedCPUOffloadPathDecision(path, nil, 1, 0, fit, false)
	if err != nil || !ok || bound != serveCPUOffloadStreamedResidentBound(fit) {
		t.Fatalf("non-V4.1 decision bound=%d ok=%v err=%v, want the gross %d", bound, ok, err, serveCPUOffloadStreamedResidentBound(fit))
	}
}
