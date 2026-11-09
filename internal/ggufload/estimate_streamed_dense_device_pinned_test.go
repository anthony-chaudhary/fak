package ggufload

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// estimate_streamed_dense_device_pinned_test.go - regression for the qwen35-hybrid
// streamed-dense admission bug: the bounded streamed-dense policy folded EVERY eligible dense
// k-quant into one bounded HOST working set, including the weight operands the executing
// backend's sequence-prefill kernel validates for a live default-device allocation. Under VRAM
// pressure the HAL spilled those operands to managed memory and the warmup preflight failed
// closed on the first one ("attention_qg must have a live default-device allocation; managed
// memory is refused"). WithStreamedDenseDeviceResident pins exactly those operands back onto the
// device charge in BOTH the estimate and the loader, so the fit guard judges what the kernel
// actually needs instead of a plan the kernel refuses.

const pinnedFixtureFFNBytes = int64(256 * 144)  // one [256,256] Q4_K tensor: 256 super-blocks x 144 B
const pinnedFixtureLMHeadBytes = int64(4 * 144) // the fixture's [256,vocab=4] Q4_K head: 4 super-blocks x 144 B

// qwen35PinnedTestConfig is the parsed config of buildQwen35GGUFFixture("qwen35"): two layers,
// full_attention_interval=4, so layer 0 is linear_attention and IsQwen35Hybrid is true.
func qwen35PinnedTestConfig(t *testing.T, ws *WeightSource) model.Config {
	t.Helper()
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("precondition broken: fixture is not a qwen35 hybrid (layer_types=%v)", cfg.LayerTypes)
	}
	return cfg
}

// TestQwen35StreamedDenseDevicePinnedKeepsPreflightOperandsOnDevice is the ESTIMATE half: every
// quantized matmul operand of the qwen35 sequence preflight (here: lm_head, v_proj, up_proj,
// down_proj) is excluded from the bounded host fold when the pin is declared, and lands back on
// the DEVICE charge at its exact packed bytes. The plan TOTAL is unchanged when the bound exceeds
// the eligible side - only the SCOPE moves from host to device - which is precisely the
// invariant the fit guard needs: it now refuses on the device aperture instead of admitting a
// plan whose kernel operand the HAL spills to managed memory.
func TestQwen35StreamedDenseDevicePinnedKeepsPreflightOperandsOnDevice(t *testing.T) {
	const dim, vocab = 256, 4
	path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ4_K, TensorQ4_K, false, false)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	cfg := qwen35PinnedTestConfig(t, ws)
	pin := func(canon string) bool { return model.Qwen35SequencePrefillDeviceResident(cfg, canon) }

	// Premise: all four quantized operands are preflight-required AND streamed-dense eligible.
	eligible := []string{
		"lm_head.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	}
	for _, canon := range eligible {
		if !pin(canon) {
			t.Fatalf("premise broken: %s is not device-residency-pinned", canon)
		}
	}
	const wantEligible = int64(3*pinnedFixtureFFNBytes + pinnedFixtureLMHeadBytes)
	const bound = int64(1 << 20) // above the eligible side, so the host row caps at the dense side

	unpinned, err := ws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("unpinned estimate: %v", err)
	}
	if got := memoryPlanBytesByDetail(unpinned)["gguf-host-dense-streamed"]; got != wantEligible {
		t.Fatalf("RED premise broken: unpinned host row = %d, want the full eligible side %d", got, wantEligible)
	}

	pinned, err := ws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(bound), WithStreamedDenseDeviceResident(pin))
	if err != nil {
		t.Fatalf("pinned estimate: %v", err)
	}
	if got := memoryPlanBytesByDetail(pinned)["gguf-host-dense-streamed"]; got != 0 {
		t.Fatalf("pinned host row = %d, want 0: every preflight operand must stay out of the host fold", got)
	}
	if got, want := pinned.DeviceTotal()-unpinned.DeviceTotal(), wantEligible; got != want {
		t.Fatalf("device delta (pinned - unpinned) = %d, want the packed bytes of the four pinned operands %d", got, want)
	}
	if got, want := pinned.Total(), unpinned.Total(); got != want {
		t.Fatalf("plan total moved: pinned %d vs unpinned %d; the pin must re-scope, not re-size", got, want)
	}

	// Partial pin: the streamed set SHRINKS by exactly the pinned tensor's packed bytes.
	partial := func(canon string) bool { return canon == "model.layers.0.self_attn.v_proj.weight" }
	half, err := ws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(bound), WithStreamedDenseDeviceResident(partial))
	if err != nil {
		t.Fatalf("partial-pin estimate: %v", err)
	}
	if got := memoryPlanBytesByDetail(half)["gguf-host-dense-streamed"]; got != wantEligible-pinnedFixtureFFNBytes {
		t.Fatalf("partial-pin host row = %d, want the three unpinned operands %d", got, wantEligible-pinnedFixtureFFNBytes)
	}
	if got, want := half.DeviceTotal()-unpinned.DeviceTotal(), pinnedFixtureFFNBytes; got != want {
		t.Fatalf("partial-pin device delta = %d, want one pinned operand %d", got, want)
	}

	// Control: on a non-hybrid arch the pin predicate matches nothing, so the plan is byte-identical
	// with and without the option - the pin is scoped to the qwen35-hybrid contract, not global.
	llamaPath := buildQwen35GGUFFixture(t, "llama", dim, vocab, TensorQ4_K, TensorQ4_K, false, false)
	lws, err := OpenWeights(llamaPath)
	if err != nil {
		t.Fatal(err)
	}
	defer lws.Close()
	llamaCfg, err := lws.File.Config()
	if err != nil {
		t.Fatalf("llama Config: %v", err)
	}
	llamaPin := func(canon string) bool { return model.Qwen35SequencePrefillDeviceResident(llamaCfg, canon) }
	a, err := lws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("llama unpinned estimate: %v", err)
	}
	bb, err := lws.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(bound), WithStreamedDenseDeviceResident(llamaPin))
	if err != nil {
		t.Fatalf("llama pinned estimate: %v", err)
	}
	if a.Total() != bb.Total() || a.DeviceTotal() != bb.DeviceTotal() || a.HostTotal() != bb.HostTotal() {
		t.Fatalf("non-hybrid control moved: unpinned (%d/%d/%d) vs pinned (%d/%d/%d)",
			a.Total(), a.DeviceTotal(), a.HostTotal(), bb.Total(), bb.DeviceTotal(), bb.HostTotal())
	}
}

// TestUnifiedHostResidencyPlanCarriesStreamedDensePinThrough is the RESIDENCY-BOUND half:
// UnifiedHostResidencyPlan rebuilds its own option list from the probed request, so the pin must
// travel with that reconstruction - otherwise the shared-pool host-residency guard would judge a
// different (un-pinned) plan than the one the loader builds.
func TestUnifiedHostResidencyPlanCarriesStreamedDensePinThrough(t *testing.T) {
	const dim, vocab = 256, 4
	path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ4_K, TensorQ4_K, false, false)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	cfg := qwen35PinnedTestConfig(t, ws)
	pin := func(canon string) bool { return model.Qwen35SequencePrefillDeviceResident(cfg, canon) }
	const wantEligible = int64(3*pinnedFixtureFFNBytes + pinnedFixtureLMHeadBytes)
	const bound = int64(1 << 20)

	unpinned, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(bound))
	if err != nil {
		t.Fatalf("unpinned residency plan: %v", err)
	}
	if got := memoryPlanBytesByDetail(unpinned)["gguf-host-dense-streamed"]; got != wantEligible {
		t.Fatalf("unpinned host row = %d, want %d", got, wantEligible)
	}
	pinned, err := ws.UnifiedHostResidencyPlan(WithStreamedDenseQ4KWorkingSet(bound), WithStreamedDenseDeviceResident(pin))
	if err != nil {
		t.Fatalf("pinned residency plan: %v", err)
	}
	if got := memoryPlanBytesByDetail(pinned)["gguf-host-dense-streamed"]; got != 0 {
		t.Fatalf("pinned host row = %d, want 0: the pin did not survive the option reconstruction", got)
	}
	if got, want := pinned.DeviceTotal()-unpinned.DeviceTotal(), wantEligible; got != want {
		t.Fatalf("residency device delta = %d, want %d", got, want)
	}
}

// qwen35PinnedStreamedFixture is the in-memory LOADER fixture: two Q4_K identity-eligible dense
// matmuls (v_proj, down_proj) of a one-layer qwen35-hybrid checkpoint, zero-payload Q4_K blocks
// (valid: d=min=0 dequantizes to zeros; the test checks ROUTING, not values).
func qwen35PinnedStreamedFixture(t *testing.T) (*WeightSource, *rangeReadCounter, model.Config) {
	t.Helper()
	const off = int64(pinnedFixtureFFNBytes)
	blob := append([]byte(nil), make([]byte, pinnedFixtureFFNBytes)...)
	blob = append(blob, make([]byte, pinnedFixtureFFNBytes)...)
	reader := &rangeReadCounter{r: bytes.NewReader(blob), lo: 0, hi: int64(len(blob))}
	meta := map[string]Value{
		"general.architecture":                    {Type: TypeString, Value: "qwen35"},
		"general.name":                            {Type: TypeString, Value: "Qwen3.8-27B-Q4_K_M"},
		"qwen35.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
		"qwen35.block_count":                      {Type: TypeUint32, Value: uint32(1)},
		"qwen35.attention.head_count":             {Type: TypeUint32, Value: uint32(1)},
		"qwen35.attention.head_count_kv":          {Type: TypeUint32, Value: uint32(1)},
		"qwen35.attention.key_length":             {Type: TypeUint32, Value: uint32(256)},
		"qwen35.feed_forward_length":              {Type: TypeUint32, Value: uint32(256)},
		"qwen35.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
		"qwen35.full_attention_interval":          {Type: TypeUint32, Value: uint32(4)},
	}
	file := &File{Metadata: meta, Tensors: []TensorInfo{
		{Name: "blk.0.attn_v.weight", Type: TensorQ4_K, Dims: []uint64{256, 256}},
		{Name: "blk.0.ffn_down.weight", Type: TensorQ4_K, Dims: []uint64{256, 256}, FileOffset: off},
	}}
	ws, err := NewWeightSource(file, reader, int64(len(blob)))
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("precondition broken: fixture is not a qwen35 hybrid (layer_types=%v)", cfg.LayerTypes)
	}
	return ws, reader, cfg
}

// TestQwen35StreamedDenseDevicePinnedLoaderMaterializesPinnedTensors is the LOADER-RETENTION
// half of the witness: the SAME pin the estimate honors must make the loader MATERIALIZE the
// pinned operands (read their payloads, hold them in the resident store, not lazy) while leaving
// unpinned eligible tensors checkpoint-backed. Plan and loader agreeing on the streamed set is
// the invariant; a divergence between them is exactly how the admitted-but-unexecutable plan
// reached the warmup preflight.
func TestQwen35StreamedDenseDevicePinnedLoaderMaterializesPinnedTensors(t *testing.T) {
	const (
		vProj = "model.layers.0.self_attn.v_proj.weight"
		down  = "model.layers.0.mlp.down_proj.weight"
	)
	t.Run("unpinned-stays-lazy", func(t *testing.T) {
		ws, reader, _ := qwen35PinnedStreamedFixture(t)
		m, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedDenseQ4K(true))
		if err != nil {
			t.Fatal(err)
		}
		if got := reader.overlaps.Load(); got != 0 {
			t.Fatalf("unpinned streamed build made %d payload reads, want zero (both operands lazy)", got)
		}
		if !m.Q4KLazy(vProj) || !m.Q4KLazy(down) {
			t.Fatalf("unpinned: lazy(v_proj)=%v lazy(down_proj)=%v, want both checkpoint-backed", m.Q4KLazy(vProj), m.Q4KLazy(down))
		}
	})

	t.Run("fully-pinned-materializes-all", func(t *testing.T) {
		ws, reader, cfg := qwen35PinnedStreamedFixture(t)
		pin := func(canon string) bool { return model.Qwen35SequencePrefillDeviceResident(cfg, canon) }
		m, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedDenseQ4K(true), WithStreamedDenseDeviceResident(pin))
		if err != nil {
			t.Fatal(err)
		}
		if got := reader.overlaps.Load(); got == 0 {
			t.Fatal("fully-pinned build made no payload reads; the pinned operands were not materialized")
		}
		for _, name := range []string{vProj, down} {
			if !m.HasQ4K(name) || m.Q4KLazy(name) {
				t.Fatalf("pinned %s: resident=%v lazy=%v, want materialized resident Q4_K", name, m.HasQ4K(name), m.Q4KLazy(name))
			}
		}
	})

	t.Run("partial-pin-shrinks-streamed-set", func(t *testing.T) {
		ws, reader, _ := qwen35PinnedStreamedFixture(t)
		// Count ONLY the down_proj range: if the partial pin (v_proj only) leaked, down_proj would
		// still be read and held lazy; if the pin over-reached, down_proj would be read here.
		reader.lo, reader.hi = int64(pinnedFixtureFFNBytes), int64(2*pinnedFixtureFFNBytes)
		partial := func(canon string) bool { return canon == vProj }
		m, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedDenseQ4K(true), WithStreamedDenseDeviceResident(partial))
		if err != nil {
			t.Fatal(err)
		}
		if got := reader.overlaps.Load(); got != 0 {
			t.Fatalf("partial pin read the unpinned down_proj payload %d times; the streamed set did not shrink correctly", got)
		}
		if !m.HasQ4K(vProj) || m.Q4KLazy(vProj) {
			t.Fatalf("pinned v_proj: resident=%v lazy=%v, want materialized", m.HasQ4K(vProj), m.Q4KLazy(vProj))
		}
		if !m.Q4KLazy(down) {
			t.Fatal("unpinned down_proj must stay checkpoint-backed under a partial pin")
		}
	})
}

// TestStreamedDenseLazyMirrorsExcludeDevicePinned binds the two host-load-peak mirrors
// (streamedDenseLazy, mappedResidentFileBacked) to the same exclusion: a pinned tensor is never a
// range descriptor in the peak simulation either, so the host peak charges it as an owned copy
// exactly as the materializing loader keeps it.
func TestStreamedDenseLazyMirrorsExcludeDevicePinned(t *testing.T) {
	const (
		vProj = "model.layers.0.self_attn.v_proj.weight"
		down  = "model.layers.0.mlp.down_proj.weight"
	)
	cfg := model.Config{ModelType: "qwen35", HiddenSize: 256, IntermediateSize: 256}
	cfg.LayerTypes = []string{"linear_attention"}
	if !cfg.IsQwen35Hybrid() {
		t.Fatal("precondition broken: test config is not a qwen35 hybrid")
	}
	o := q4kLoadOptions{streamedDenseQ4K: true, streamedDensePinned: func(canon string) bool { return canon == vProj }}

	if streamedDenseLazy(cfg, o, TensorQ4_K, vProj) {
		t.Fatal("streamedDenseLazy admitted a device-pinned tensor as a range descriptor")
	}
	if !streamedDenseLazy(cfg, o, TensorQ4_K, down) {
		t.Fatal("streamedDenseLazy lost an unpinned eligible tensor")
	}
	st := q4kTensorStorage{canon: vProj, info: TensorInfo{Type: TensorQ4_K}, kind: q4kStoragePacked}
	if mappedResidentFileBacked(cfg, st, o) {
		t.Fatal("mappedResidentFileBacked admitted a device-pinned tensor as a file-backed view")
	}
	st.canon = down
	if !mappedResidentFileBacked(cfg, st, o) {
		t.Fatal("mappedResidentFileBacked lost an unpinned eligible tensor")
	}
	// Nil-safe: no pin declared -> historical behavior byte-for-byte.
	var plain q4kLoadOptions
	plain.streamedDenseQ4K = true
	if plain.streamedDensePins(vProj) {
		t.Fatal("zero-value options must pin nothing (nil predicate)")
	}
	if !streamedDenseLazy(cfg, plain, TensorQ4_K, vProj) {
		t.Fatal("zero-value options must keep the historical lazy route (no pin)")
	}
}
