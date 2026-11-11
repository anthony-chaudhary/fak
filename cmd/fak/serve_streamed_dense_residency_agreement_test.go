package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serve_streamed_dense_residency_agreement_test.go - plan-json/runtime AGREEMENT witness for the
// streamed-dense residency carve-out (fak#13205 follow-on; private ticket aegis-27b-model-capacity).
// The admission side (serveQ4KFitOptions -> EstimateQ4KLoadMemoryPlan, emitted by --plan-json via
// buildServeSizingArtifact) and the load side (serveResidentQ4KLoadOptions +
// serveLoadStreamedDenseOptions -> LoadModelQ4KStreamedDense) each build their OWN option list from
// the SAME backend/artifact predicates. This test asserts the two lists agree on WHICH eligible
// streamed-dense tensors stay device-resident (the pin predicate, per tensor over the full canon
// chain), that they carry the same policy class, and that the pinned estimate's fold matches what
// the loader actually materializes: every pinned operand resident Q4_K, nothing else lazy-divergent.
// A divergence between the two sides is exactly how an admitted-but-unexecutable plan reaches the
// warmup preflight ("attention_qg must have a live default-device allocation").

// serveStreamedDenseMarkerBackend is serveCapBackend advertising the qwen35-hybrid sequence-prefill
// capability seam (qwen35SequencePrefillMarker): the marker plus the qualified path identity is what
// wires WithStreamedDenseDeviceResident into serveResidentQ4KLoadOptions.
type serveStreamedDenseMarkerBackend struct {
	serveCapBackend
}

func (b serveStreamedDenseMarkerBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}

const (
	streamedDenseAgreementDim   = 256
	streamedDenseAgreementVocab = 4
	// One [256,256] Q4_K tensor: 256 super-blocks x 144 B.
	streamedDenseAgreementLargeBytes = int64(36864)
	// The [256,vocab=4] Q4_K head: 4 super-blocks x 144 B.
	streamedDenseAgreementSmallBytes = int64(576)
	// The [512,256] Q4_K gated-q projection: 512 super-blocks x 144 B.
	streamedDenseAgreementQGBytes = int64(73728)
)

// streamedDenseAgreementEligible is the bounded-streamed-dense eligible set of the fixture: the six
// identity-layout dense matmuls (v/up/down of both layers) plus the untied lm_head. token_embd is NOT
// eligible (not a quant matmul weight); the self_attn q/k projections are NOT eligible (rotary/gated
// unpermute, not identity) - they stay device-charged in every variant, which is why the pin's
// attention_qg exclusion only matters through the fold it shrinks.
var streamedDenseAgreementEligible = []string{
	"model.layers.0.self_attn.v_proj.weight",
	"model.layers.0.mlp.up_proj.weight",
	"model.layers.0.mlp.down_proj.weight",
	"model.layers.1.self_attn.v_proj.weight",
	"model.layers.1.mlp.up_proj.weight",
	"model.layers.1.mlp.down_proj.weight",
	"lm_head.weight",
}

const streamedDenseAgreementWantEligible = int64(6*streamedDenseAgreementLargeBytes + streamedDenseAgreementSmallBytes)

// writeServeStreamedDenseHybridGGUF writes the 10-tensor fixture: two layers of a dim=256/vocab=4
// checkpoint, all Q4_K, zero payloads (valid: d=min=0 dequantizes to zeros; the test checks ROUTING
// and RESIDENCY, not values). arch "qwen35" with full_attention_interval=2 yields LayerTypes
// [linear_attention, full_attention] (a hybrid carrying a real attention_qg operand at layer 1);
// arch "llama" is the non-hybrid control whose pin predicate matches nothing. The qwen35
// full-attention layer's q_proj stacks query+gate (gated rotary), so its out-dim is
// 2*heads*headDim = 512, not the hidden width the llama control keeps. The layer-1 q_proj
// carries the arch's own out-dim (qwen35: gated rotary, 2 x heads x headDim; llama: heads x
// headDim); stored GGUF dims are [in, out], which the parser reverses into the [out, in] shape
// the loader validates.
func writeServeStreamedDenseHybridGGUF(t *testing.T, arch string) string {
	t.Helper()
	const dim, vocab = streamedDenseAgreementDim, streamedDenseAgreementVocab
	large := uint64(streamedDenseAgreementLargeBytes)
	small := uint64(streamedDenseAgreementSmallBytes)

	var b bytes.Buffer
	if arch == "qwen35" {
		writeMinimalHeaderForTest(&b, 10, 11)
		writeKVStringForTest(&b, "general.architecture", "qwen35")
		writeKVUint32ForTest(&b, "general.alignment", 32)
		writeKVUint32ForTest(&b, "qwen35.embedding_length", dim)
		writeKVUint32ForTest(&b, "qwen35.block_count", 2)
		writeKVUint32ForTest(&b, "qwen35.attention.head_count", 1)
		writeKVUint32ForTest(&b, "qwen35.attention.head_count_kv", 1)
		writeKVUint32ForTest(&b, "qwen35.full_attention_interval", 2)
		writeKVUint32ForTest(&b, "qwen35.attention.key_length", dim)
		writeKVUint32ForTest(&b, "qwen35.feed_forward_length", dim)
		writeKVUint32ForTest(&b, "qwen35.context_length", 16)
		writeKVFloat32ForTest(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)
	} else {
		writeMinimalHeaderForTest(&b, 10, 9)
		writeKVStringForTest(&b, "general.architecture", "llama")
		writeKVUint32ForTest(&b, "general.alignment", 32)
		writeKVUint32ForTest(&b, "llama.embedding_length", dim)
		writeKVUint32ForTest(&b, "llama.block_count", 2)
		writeKVUint32ForTest(&b, "llama.attention.head_count", 1)
		writeKVUint32ForTest(&b, "llama.attention.key_length", dim)
		writeKVUint32ForTest(&b, "llama.feed_forward_length", dim)
		writeKVUint32ForTest(&b, "llama.context_length", 16)
		writeKVFloat32ForTest(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)
	}

	// qwen35 gated rotary: q_proj out-dim = 2 x heads(1) x headDim; the llama control carries
	// the plain heads x headDim out-dim.
	qOut, attnQBytes := dim, large
	if arch == "qwen35" {
		qOut, attnQBytes = 2*dim, uint64(streamedDenseAgreementQGBytes)
	}
	type fixtureTensor struct {
		name  string
		dims  [2]uint64 // stored GGUF order [in, out]; the parser reverses to [out, in]
		bytes uint64
	}
	tensors := []fixtureTensor{
		{"token_embd.weight", [2]uint64{dim, vocab}, small},
		{"output.weight", [2]uint64{dim, vocab}, small},
		{"blk.0.attn_v.weight", [2]uint64{dim, dim}, large},
		{"blk.0.ffn_up.weight", [2]uint64{dim, dim}, large},
		{"blk.0.ffn_down.weight", [2]uint64{dim, dim}, large},
		{"blk.1.attn_q.weight", [2]uint64{dim, uint64(qOut)}, attnQBytes},
		{"blk.1.attn_k.weight", [2]uint64{dim, dim}, large},
		{"blk.1.attn_v.weight", [2]uint64{dim, dim}, large},
		{"blk.1.ffn_up.weight", [2]uint64{dim, dim}, large},
		{"blk.1.ffn_down.weight", [2]uint64{dim, dim}, large},
	}
	var off uint64
	for _, ft := range tensors {
		writeTensorInfoForTest(&b, ft.name, ft.dims[:], uint32(ggufload.TensorQ4_K), off)
		off += ft.bytes
	}
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, int(off)))

	path := filepath.Join(t.TempDir(), arch+"-streamed-dense-agreement.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// streamedDenseAgreementSizingBytes sums one plan-json demand set by Detail.
func streamedDenseAgreementSizingBytes(rows []serveSizingDemandRow) map[string]int64 {
	out := map[string]int64{}
	for _, r := range rows {
		out[r.Detail] += r.Bytes
	}
	return out
}

// TestServeStreamedDenseResidencyAgreement is the cross-check: the plan-json admission options and
// the runtime load options must name the SAME device-resident set, and the loader must materialize
// exactly that set. It fails on any disagreement between the two sides.
//
// fak-test:runtime fast est=100ms
func TestServeStreamedDenseResidencyAgreement(t *testing.T) {
	if compute.Default() == nil {
		t.Skip("no reference backend registered; device arm not exercisable")
	}
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "1")

	be := serveStreamedDenseMarkerBackend{serveCapBackend{
		Backend: compute.Default(), total: 16 << 30, free: 16 << 30, known: true, uploadDtype: true,
	}}
	fit := serveDeviceFitBudget(be)
	hostFit := serveStreamedHostFit(be, &fit)
	hostProbeable := hostFit.avail() > 0

	t.Run("hybrid-pin-agrees-plan-and-runtime", func(t *testing.T) {
		path := writeServeStreamedDenseHybridGGUF(t, "qwen35")
		ws, err := ggufload.OpenWeights(path)
		if err != nil {
			t.Fatal(err)
		}
		defer ws.Close()
		cfg, err := ws.File.Config()
		if err != nil {
			t.Fatalf("Config: %v", err)
		}
		if !cfg.IsQwen35Hybrid() {
			t.Fatalf("precondition broken: fixture is not a qwen35 hybrid (layer_types=%v)", cfg.LayerTypes)
		}
		quant := ggufload.ClassifyTensorQuant(ws.File.Tensors)

		// The two sides' option lists, built exactly as production builds them.
		planOpts := serveQ4KFitOptions(path, ws, be, serveLoadArmResidentQ4K, fit)
		loadOpts := append(append([]ggufload.Q4KLoadOption(nil),
			serveResidentQ4KLoadOptions(be, path, true, quant)...),
			serveLoadStreamedDenseOptions(hostFit, true, false, 0)...)

		planPin := ggufload.StreamedDensePin(planOpts)
		loadPin := ggufload.StreamedDensePin(loadOpts)
		if planPin == nil || loadPin == nil {
			t.Fatalf("marker backend + hybrid checkpoint must wire the device-residency pin on BOTH sides: plan=%v runtime=%v", planPin != nil, loadPin != nil)
		}

		// Per-tensor agreement over the full canon chain (CanonicalTensorNameArch ->
		// QuantSourceTensorName), the exact chain both the estimate fold and the loader dispatch run.
		for _, info := range ws.File.Tensors {
			canon, ok := ggufload.CanonicalTensorNameArch(info.Name, cfg.ModelType)
			if !ok {
				t.Fatalf("fixture tensor %s has no canonical mapping; premise broken", info.Name)
			}
			resolved, keep := fakmodel.QuantSourceTensorName(cfg, canon)
			if !keep {
				continue
			}
			if planPin(resolved) != loadPin(resolved) {
				t.Fatalf("residency disagreement on %s (%s): plan-json pins=%v runtime pins=%v", resolved, info.Name, planPin(resolved), loadPin(resolved))
			}
		}
		// The attention_qg kernel operand (layer 1 full-attention q_proj) is pinned on both sides.
		const attnQG = "model.layers.1.self_attn.q_proj.weight"
		if !planPin(attnQG) || !loadPin(attnQG) {
			t.Fatalf("attention_qg operand %s: plan-json pins=%v runtime pins=%v, want both pinned", attnQG, planPin(attnQG), loadPin(attnQG))
		}

		// Policy-class agreement through the shared option-application surface.
		effPlan := ggufload.ApplyQ4KLoadOptions(planOpts)
		effLoad := ggufload.ApplyQ4KLoadOptions(loadOpts)
		if effPlan.StreamedDenseQ4K != effLoad.StreamedDenseQ4K {
			t.Fatalf("streamed-dense policy diverges: plan-json=%v runtime=%v", effPlan.StreamedDenseQ4K, effLoad.StreamedDenseQ4K)
		}
		if hostProbeable && (effPlan.StreamedDenseBounded != effLoad.StreamedDenseBounded || !effPlan.StreamedDenseBounded) {
			t.Fatalf("bounded-form divergence on a probeable host: plan-json bounded=%v runtime bounded=%v", effPlan.StreamedDenseBounded, effLoad.StreamedDenseBounded)
		}

		// Estimate fold: the unpinned baseline folds the whole eligible side into ONE host row; the
		// pinned plan estimate folds NOTHING (every eligible tensor is a preflight operand here), and
		// the move is re-scope, not re-size.
		unpinned, err := ws.EstimateQ4KLoadMemoryPlan(ggufload.WithStreamedDenseQ4KWorkingSet(1 << 20))
		if err != nil {
			t.Fatalf("unpinned estimate: %v", err)
		}
		if got := serveMemoryBytesByDetail(unpinned)["gguf-host-dense-streamed"]; got != streamedDenseAgreementWantEligible {
			t.Fatalf("RED premise broken: unpinned host row = %d, want the full eligible side %d", got, streamedDenseAgreementWantEligible)
		}
		pinned, err := ws.EstimateQ4KLoadMemoryPlan(planOpts...)
		if err != nil {
			t.Fatalf("pinned plan estimate: %v", err)
		}
		if hostProbeable {
			if got := serveMemoryBytesByDetail(pinned)["gguf-host-dense-streamed"]; got != 0 {
				t.Fatalf("pinned plan host row = %d, want 0: every eligible operand stays out of the host fold", got)
			}
		}
		if got, want := pinned.DeviceTotal()-unpinned.DeviceTotal(), streamedDenseAgreementWantEligible; got != want {
			t.Fatalf("device delta (pinned - unpinned) = %d, want the packed bytes of the eligible side %d", got, want)
		}
		if pinned.Total() != unpinned.Total() {
			t.Fatalf("plan total moved: pinned %d vs unpinned %d; the pin must re-scope, not re-size", pinned.Total(), unpinned.Total())
		}

		// Runtime: the loader materializes EXACTLY the pinned set - every eligible operand resident
		// Q4_K (HasQ4K && !Q4KLazy), the q/k projections never lazy either way.
		m, err := ggufload.LoadModelQ4KStreamedDense(path, nil, loadOpts...)
		if err != nil {
			t.Fatalf("streamed-dense load: %v", err)
		}
		defer m.CloseWeights()
		for _, canon := range streamedDenseAgreementEligible {
			if !m.HasQ4K(canon) || m.Q4KLazy(canon) {
				t.Fatalf("pinned %s: resident=%v lazy=%v, want materialized resident Q4_K (plan and runtime disagree)", canon, m.HasQ4K(canon), m.Q4KLazy(canon))
			}
		}
		for _, canon := range []string{"model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight"} {
			if m.Q4KLazy(canon) {
				t.Fatalf("%s went checkpoint-backed under the pin; the runtime diverged from the plan", canon)
			}
		}

		// End-to-end plan-json: the --plan-json artifact carries the SAME residency set - on a
		// probeable host its bounded dense row is empty, matching the direct pinned estimate.
		art, err := buildServeSizingArtifact(ws, be, false, 0, path, 0)
		if err != nil {
			t.Fatalf("plan-json artifact: %v", err)
		}
		if hostProbeable {
			if got := streamedDenseAgreementSizingBytes(art.Demands)["gguf-host-dense-streamed"]; got != 0 {
				t.Fatalf("plan-json dense host row = %d, want 0 (the pinned estimate folds nothing)", got)
			}
		}
	})

	t.Run("llama-control-no-pin", func(t *testing.T) {
		path := writeServeStreamedDenseHybridGGUF(t, "llama")
		ws, err := ggufload.OpenWeights(path)
		if err != nil {
			t.Fatal(err)
		}
		defer ws.Close()
		cfg, err := ws.File.Config()
		if err != nil {
			t.Fatalf("Config: %v", err)
		}
		if cfg.IsQwen35Hybrid() {
			t.Fatal("precondition broken: llama control must not be a qwen35 hybrid")
		}
		quant := ggufload.ClassifyTensorQuant(ws.File.Tensors)

		planOpts := serveQ4KFitOptions(path, ws, be, serveLoadArmResidentQ4K, fit)
		loadOpts := append(append([]ggufload.Q4KLoadOption(nil),
			serveResidentQ4KLoadOptions(be, path, true, quant)...),
			serveLoadStreamedDenseOptions(hostFit, true, false, 0)...)
		if p, l := ggufload.StreamedDensePin(planOpts), ggufload.StreamedDensePin(loadOpts); p != nil || l != nil {
			t.Fatalf("non-hybrid arch must pin nothing on either side: plan-json=%v runtime=%v", p != nil, l != nil)
		}

		est, err := ws.EstimateQ4KLoadMemoryPlan(ggufload.WithStreamedDenseQ4KWorkingSet(1 << 20))
		if err != nil {
			t.Fatalf("llama estimate: %v", err)
		}
		if got := serveMemoryBytesByDetail(est)["gguf-host-dense-streamed"]; got != streamedDenseAgreementWantEligible {
			t.Fatalf("llama host row = %d, want the full eligible side %d (no pin to shrink it)", got, streamedDenseAgreementWantEligible)
		}

		m, err := ggufload.LoadModelQ4KStreamedDense(path, nil, loadOpts...)
		if err != nil {
			t.Fatalf("llama streamed-dense load: %v", err)
		}
		defer m.CloseWeights()
		for _, canon := range streamedDenseAgreementEligible {
			if !m.HasQ4K(canon) || !m.Q4KLazy(canon) {
				t.Fatalf("llama %s: resident=%v lazy=%v, want checkpoint-backed (unpinned streamed route)", canon, m.HasQ4K(canon), m.Q4KLazy(canon))
			}
		}
	})
}
