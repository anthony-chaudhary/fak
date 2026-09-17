package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serve_streamed_offload_test.go — fak#13121 acceptance at the serve-arm seam. The MoE
// --cpu-offload-experts planner charged the whole routed-expert payload host-resident, so a
// DeepSeek-V4.1 Flash Q2_K checkpoint (183.25 GiB of experts against a 62 GiB Halo) refused before
// any load. The streamed policy charges a BOUNDED resident working set instead and faults the rest
// from the staged shards, so the serve arm must SELECT it exactly when the full charge cannot be
// host-resident AND the checkpoint tier can stage the routed slabs.

// serveStreamedSynthWeightSource is a Q2_K routed-expert MoE fixture (the DeepSeek-V4.1 Flash
// encoding) whose fused slabs are 3-D [E, out, in] and therefore stageable by the checkpoint tier.
// Header only: NewWeightSource indexes the directory without reading a payload. GGUF stores dims
// reversed, so Dims {in, out, E} yields shape [E, out, in].
func serveStreamedSynthWeightSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	const (
		experts = 4
		hidden  = 256
	)
	// One Q2_K slab is experts*(hidden/256)*84 bytes; the offsets below are absolute into the blob.
	slabBytes := int64(experts) * int64(hidden/256) * 84
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                         {Type: ggufload.TypeString, Value: "glm_moe_dsa"},
			"glm_moe_dsa.context_length":                   {Type: ggufload.TypeUint64, Value: uint64(16)},
			"glm_moe_dsa.embedding_length":                 {Type: ggufload.TypeUint64, Value: uint64(32)},
			"glm_moe_dsa.block_count":                      {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm_moe_dsa.feed_forward_length":              {Type: ggufload.TypeUint64, Value: uint64(64)},
			"glm_moe_dsa.attention.head_count":             {Type: ggufload.TypeUint64, Value: uint64(4)},
			"glm_moe_dsa.attention.head_count_kv":          {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm_moe_dsa.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
			"glm_moe_dsa.rope.freq_base":                   {Type: ggufload.TypeFloat32, Value: float32(10000)},
			"glm_moe_dsa.expert_count":                     {Type: ggufload.TypeUint64, Value: uint64(experts)},
			"glm_moe_dsa.expert_used_count":                {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm_moe_dsa.expert_feed_forward_length":       {Type: ggufload.TypeUint64, Value: uint64(hidden)},
			"tokenizer.ggml.eos_token_id":                  {Type: ggufload.TypeUint32, Value: uint32(2)},
		},
		Tensors: []ggufload.TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: ggufload.TensorF32, FileOffset: 0},
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: ggufload.TensorQ2_K, FileOffset: 1024},
			{Name: "blk.0.ffn_up_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: ggufload.TensorQ2_K, FileOffset: 1024 + slabBytes},
			{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{hidden, hidden, experts}, Type: ggufload.TensorQ2_K, FileOffset: 1024 + 2*slabBytes},
		},
	}
	// A real in-memory shard so FusedExpertTensors can resolve each slab's owning reader (the
	// checkpoint tier reads through it).
	blob := make([]byte, 1024+3*slabBytes)
	ws, err := ggufload.NewWeightSource(f, bytes.NewReader(blob), int64(len(blob)))
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// The routed slabs of the fixture must be stageable by the checkpoint tier — this is the artifact
// half of the streamed decision, so if it is false the feature is silently inert on the exact
// artifact it exists for.
func TestServeStreamedExpertSlabsAreStageable(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	if !serveStreamedExpertsCapable(ws) {
		t.Fatal("fixture routed slabs are not stageable; the streamed decision has no fault path to arm")
	}
}

// A routed set that fits the host budget keeps the RESIDENT arm byte-for-byte (P4 preserved): the
// streamed policy must not fire when the full charge is host-resident.
func TestServeCPUOffloadStreamedArmNotSelectedWhenResidentFits(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	// A huge host budget: the full routed charge fits, so streaming is unnecessary.
	bigFit := serveFitBudget{Base: 1 << 40, Headroom: 0}
	plan, streamed, err := serveStreamedCPUOffloadPlan(ws, nil, 1, 0, bigFit)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPlan: %v", err)
	}
	if streamed {
		t.Fatalf("streamed policy fired although the full routed charge fits the host budget; plan=%+v", plan)
	}
	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, bigFit)
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if plan.HostTotal() != resident.HostTotal() {
		t.Fatalf("non-streamed plan host total %d, want the resident plan %d (must be byte-identical)", plan.HostTotal(), resident.HostTotal())
	}
}

// The core arm witness: when the full routed charge cannot be host-resident AND the slabs are
// stageable, the streamed policy is selected, the bounded resident bound is charged, and the device
// dense side is UNCHANGED against the resident plan's device side.
func TestServeCPUOffloadStreamedArm(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	if !serveStreamedExpertsCapable(ws) {
		t.Fatal("fixture routed slabs are not stageable; the streamed fault path has no work here")
	}

	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if resident.HostTotal() <= 0 {
		t.Fatal("fixture must host-scope the routed experts (HostTotal>0)")
	}

	// A host budget far below the routed charge forces the streamed decision.
	tinyFit := serveFitBudget{Base: 512, Headroom: 0}
	plan, streamed, err := serveStreamedCPUOffloadPlan(ws, nil, 1, 0, tinyFit)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPlan: %v", err)
	}
	if !streamed {
		t.Fatalf("streamed policy did not fire with a bounded host budget %d against a routed set %d", tinyFit.Base, resident.HostTotal())
	}
	if plan.HostTotal() > tinyFit.avail() {
		t.Fatalf("streamed host total %d exceeds the bounded resident budget %d", plan.HostTotal(), tinyFit.avail())
	}
	if plan.DeviceTotal() != resident.DeviceTotal() {
		t.Fatalf("streamed device total %d, want the unchanged dense side %d", plan.DeviceTotal(), resident.DeviceTotal())
	}
	byDetail := map[string]int64{}
	for _, d := range plan {
		byDetail[d.Detail] += d.Bytes
	}
	if byDetail["gguf-host-expert-offload-streamed"] == 0 {
		t.Fatalf("streamed plan carries no gguf-host-expert-offload-streamed row; plan=%+v", plan)
	}
}

// writeKVUint64ForTest writes a uint64 GGUF metadata value — the type the GLM-MoE-DSA header
// declares for its expert/geometry keys, matching serveStreamedSynthWeightSource's in-memory
// fixture byte-for-byte so the on-disk checkpoint parses to the same config.
func writeKVUint64ForTest(b *bytes.Buffer, key string, value uint64) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeUint64))
	_ = binary.Write(b, binary.LittleEndian, value)
}

// writeServeStreamedSynthGGUF serializes serveStreamedSynthWeightSource's MoE fixture to a real
// GGUF path, so the path-form device arm (fitServeStreamedCPUOffloadPathOnDevice) can be exercised
// over the SAME stageable Q2_K routed-expert checkpoint the in-memory plan tests use. Offsets are
// laid out per slab with the true Q2_K block math (256 values/block, 84 bytes/block) and a
// zero-filled payload of exactly that size, so FusedExpertTensors resolves each slab's reader.
func writeServeStreamedSynthGGUF(t *testing.T, base string) string {
	t.Helper()
	const (
		experts = 4
		hidden  = 256
	)
	slabBytes := uint64(experts) * uint64(hidden) * uint64(hidden) / 256 * 84
	align32 := func(n uint64) uint64 { return (n + 31) &^ 31 }

	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 4, 14)
	writeKVStringForTest(&b, "general.architecture", "glm_moe_dsa")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint64ForTest(&b, "glm_moe_dsa.context_length", 16)
	writeKVUint64ForTest(&b, "glm_moe_dsa.embedding_length", 32)
	writeKVUint64ForTest(&b, "glm_moe_dsa.block_count", 2)
	writeKVUint64ForTest(&b, "glm_moe_dsa.feed_forward_length", 64)
	writeKVUint64ForTest(&b, "glm_moe_dsa.attention.head_count", 4)
	writeKVUint64ForTest(&b, "glm_moe_dsa.attention.head_count_kv", 2)
	writeKVFloat32ForTest(&b, "glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVFloat32ForTest(&b, "glm_moe_dsa.rope.freq_base", 10000)
	writeKVUint64ForTest(&b, "glm_moe_dsa.expert_count", experts)
	writeKVUint64ForTest(&b, "glm_moe_dsa.expert_used_count", 2)
	writeKVUint64ForTest(&b, "glm_moe_dsa.expert_feed_forward_length", hidden)
	writeKVUint32ForTest(&b, "tokenizer.ggml.eos_token_id", 2)

	offset := uint64(0)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{256}, uint32(ggufload.TensorF32), offset)
	offset = align32(offset + 1024)
	for _, name := range []string{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"} {
		writeTensorInfoForTest(&b, name, []uint64{hidden, hidden, experts}, uint32(ggufload.TensorQ2_K), offset)
		offset = align32(offset + slabBytes)
	}
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, int(offset)))

	path := filepath.Join(t.TempDir(), base)
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatalf("writeServeStreamedSynthGGUF: %v", err)
	}
	return path
}

// The device arm regression (fak#13121 "one measurement"): the streamed DECISION and the device
// SIZING plan must be judged against the SAME host-fit snapshot. The device arm previously
// re-probed a fresh host budget inside fitServeStreamedCPUOffloadPathOnDevice, so a decision made
// on the caller's snapshot could be contradicted by a later, different probe. Here the caller's
// hostFit is a TINY snapshot that forces streaming while the real machine's probe is far larger;
// the device arm must honor the threaded snapshot (streamed==decision) rather than disagree.
func TestServeStreamedDeviceArmUsesDecisionHostFitSnapshot(t *testing.T) {
	path := writeServeStreamedSynthGGUF(t, "glm-moe-dsa-streamed.gguf")

	// A device backend with a generous, known device ceiling so device admission never masks the
	// host-fit invariant under test. hostKnown=false keeps host admission fail-open.
	be := serveCapBackend{total: 1 << 40, free: 1 << 40, known: true}

	// The caller's one-measurement snapshot: tiny, so the full routed charge cannot be host-resident.
	tinyFit := serveFitBudget{Base: 512, Headroom: 0}
	decided, _, err := serveStreamedCPUOffloadPathDecision(path, nil, 1, 0, tinyFit, false)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPathDecision: %v", err)
	}
	if !decided {
		t.Fatal("fixture must force the streamed decision under the tiny host-fit snapshot")
	}

	// The same snapshot threaded into the device arm must yield the SAME streamed verdict.
	plan, streamed, err := fitServeStreamedCPUOffloadPathOnDevice(path, be, 1, 0, tinyFit, nil)
	if err != nil {
		t.Fatalf("fitServeStreamedCPUOffloadPathOnDevice: %v", err)
	}
	if streamed != decided {
		t.Fatalf("device sizing disagreed with the streamed decision: decision=%v plan=%v (one-measurement invariant violated)", decided, streamed)
	}
	if plan.HostTotal() > tinyFit.avail() {
		t.Fatalf("streamed device-arm plan host total %d exceeds the bounded resident snapshot %d", plan.HostTotal(), tinyFit.avail())
	}
}

// A fitting artifact keeps the resident arm byte-for-byte on the device path: with the same tiny
// snapshot threaded through but a HOST budget large enough that the full routed charge fits, the
// decision and the device sizing must BOTH report streamed=false and the plan must match the
// resident plan's host total.
func TestServeStreamedDeviceArmFittingArtifactStaysResident(t *testing.T) {
	path := writeServeStreamedSynthGGUF(t, "glm-moe-dsa-resident.gguf")

	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}

	be := serveCapBackend{total: 1 << 40, free: 1 << 40, known: true}
	bigFit := serveFitBudget{Base: 1 << 40, Headroom: 0}
	decided, _, err := serveStreamedCPUOffloadPathDecision(path, nil, 1, 0, bigFit, false)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPathDecision: %v", err)
	}
	if decided {
		t.Fatal("fixture must NOT stream when the full routed charge fits the host snapshot")
	}

	plan, streamed, err := fitServeStreamedCPUOffloadPathOnDevice(path, be, 1, 0, bigFit, nil)
	if err != nil {
		t.Fatalf("fitServeStreamedCPUOffloadPathOnDevice: %v", err)
	}
	if streamed != decided {
		t.Fatalf("device sizing disagreed with the resident decision: decision=%v plan=%v", decided, streamed)
	}
	if plan.HostTotal() != resident.HostTotal() {
		t.Fatalf("resident-path host total %d, want the byte-identical resident plan %d", plan.HostTotal(), resident.HostTotal())
	}
}

// TestServeStreamedResidentBoundMargin is the fak#13140 regression at the bound-arithmetic seam.
// serveCPUOffloadStreamedResidentBound used to return fit.avail() VERBATIM, so on a probeable host
// the bound WAS the budget the streamed plan is judged against (fitServeStreamedCPUOffloadPathOnHost
// -> refuseHostPlanAgainstFit). Any non-expert host row (KV/scratch/activation) or the int64
// truncation in compute.BudgetAfterHeadroom pushed the judged host total a few bytes over the
// budget and failed closed, with no real wall: the physical strix3 reproduction was
// "plan needs 48.01 GiB, host has 47.99 GiB (FitTooBig)" on the exact host class the streamed arm
// exists to serve. The bound must be a DECLARED margin strictly BELOW avail, and a streamed plan
// whose routed set exceeds the raw host budget must then be ADMITTED while still a genuine bounded
// working set (not zero, not the whole budget).
func TestServeStreamedResidentBoundMargin(t *testing.T) {
	// 1) Probeable host: the bound is strictly below the budget it is judged against, and still a
	// genuine nonzero working set the loader can keep resident.
	fits := []int64{1 << 20, 1 << 30, 64 << 30}
	for _, base := range fits {
		fit := serveFitBudget{Base: base, Headroom: 0}
		bound := serveCPUOffloadStreamedResidentBound(fit)
		if bound >= fit.avail() {
			t.Fatalf("base=%d: bound %d is not strictly below avail %d; the plan ties the budget it is judged against", base, bound, fit.avail())
		}
		if bound <= 0 {
			t.Fatalf("base=%d: bound %d collapsed to zero; the streamed arm would bill a stream-through set", base, bound)
		}
	}

	// An unprobeable host keeps the honest zero floor (stream-through) -- preserved behaviour.
	if bound := serveCPUOffloadStreamedResidentBound(serveFitBudget{}); bound != 0 {
		t.Fatalf("unprobeable host bound = %d, want 0 (stream-through floor)", bound)
	}

	// 2) The admission the defect denied: a routed set that CANNOT be host-resident (routed>avail)
	// but is stageable must produce a streamed plan whose host total sits STRICTLY below avail, so
	// the host fit check ADMITS it instead of refusing by a razor-thin arithmetic miss.
	ws := serveStreamedSynthWeightSource(t)
	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if resident.HostTotal() <= 0 {
		t.Fatal("fixture must host-scope the routed experts (HostTotal>0)")
	}

	// A host budget the routed set cannot fully fit, but big enough that the bounded resident set is
	// substantial: it forces the streamed arm and exercises the margin against real KV/scratch rows.
	fit := serveFitBudget{Base: resident.HostTotal() / 2, Headroom: 0}
	if fit.avail() <= 0 {
		t.Fatalf("fixture routed set too small to build a forced-stream budget (%d)", resident.HostTotal())
	}
	plan, streamed, err := serveStreamedCPUOffloadPlan(ws, nil, 1, 0, fit)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPlan: %v", err)
	}
	if !streamed {
		t.Fatalf("streamed policy did not fire with budget %d against routed set %d", fit.avail(), resident.HostTotal())
	}
	if plan.HostTotal() >= fit.avail() {
		t.Fatalf("streamed host total %d is not strictly below avail %d; the plan still ties the budget it is judged against", plan.HostTotal(), fit.avail())
	}
	if err := refuseHostPlanAgainstFit(plan, fit); err != nil {
		t.Fatalf("streamed plan whose routed set exceeds the host budget was NOT admitted by the host fit check: %v (host total %d, avail %d)", err, plan.HostTotal(), fit.avail())
	}
}

// TestServeStreamedHostFitIgnoresDeviceOverride is the fak#13142 regression at the hostFit-selection
// seam. loadServeInKernelModel used to do `hostFit := serveHostFitBudget(); if fit != nil { hostFit =
// *fit }`. On a device serve `fit` is the DEVICE override (serveNativeContextSizingInputs returns
// serveDeviceFitBudget(be) for be != nil), so the bounded-resident streamed bound was sized from the
// device ceiling (0.90 * 71.53 = 64.38 GiB of VRAM) and then judged against real host RAM (~48.89
// GiB): the strix3 "needs 64.38 GiB, host has 48.89 GiB (FitTooBig)" refusal, no load.
//
// The selection must size the streamed decision from the HOST probe whenever a device backend is
// present, accepting the injected override as a host snapshot ONLY on the device-less arm (where it
// IS the host fit the sizing path measured). This test pins that at the real predicate the loader
// calls, then proves the consequence on the plan: a host-scale budget admits the streamed plan even
// though a device-scale override rode in.
func TestServeStreamedHostFitIgnoresDeviceOverride(t *testing.T) {
	// strix3's real host RAM and VRAM ceiling, in bytes. Bound through vars so the float multiply is
	// a runtime expression (a constant float->int64 conversion is rejected at compile time).
	hostBudgetGiB := 48.89
	deviceBudgetGiB := 71.53
	hostBudgetBytes := int64(hostBudgetGiB * float64(int64(1)<<30))
	deviceBudgetBytes := int64(deviceBudgetGiB * float64(int64(1)<<30))
	deviceOverride := serveFitBudget{Base: deviceBudgetBytes, Headroom: 0}

	// Device ARM: the device-scale override must NOT become the streamed host fit. The true host
	// probe decides; whatever this machine reports, it must not equal the injected device override
	// unless the live probe happens to agree byte-for-byte (which the assertion below excludes for
	// the pinned device value on any real host).
	be := serveCapBackend{Backend: compute.Default(), total: deviceBudgetBytes, free: deviceBudgetBytes, known: true}
	deviceSelected := serveStreamedHostFit(be, &deviceOverride)
	if deviceSelected == deviceOverride {
		t.Fatalf("device arm took the DEVICE override %d as the streamed host fit; the streamed bound would be sized from VRAM (fak#13142)", deviceOverride.Base)
	}
	if deviceSelected.Base == deviceBudgetBytes {
		t.Fatalf("device arm host fit base %d equals the device override base; device VRAM leaked into host sizing", deviceSelected.Base)
	}

	// Device-LESS arm: the injected override IS the host snapshot and must win verbatim, so the
	// sizing path's measurement and the load's stay one snapshot (the preserved #13121 behaviour).
	if hostless := serveStreamedHostFit(nil, &deviceOverride); hostless != deviceOverride {
		t.Fatalf("device-less arm override = %+v, want the injected host snapshot %+v", hostless, deviceOverride)
	}
	// A nil override on either arm keeps the live probe (fail-open, unchanged). The live host probe
	// is MemAvailable-driven and drifts between calls, so assert the SHAPE (a positive host-scale
	// budget that is not the injected device override) rather than byte-equality against a second
	// probe.
	live := serveStreamedHostFit(nil, nil)
	if live.Base <= 0 {
		t.Fatalf("nil override device-less fit base %d is not a live host budget (%+v)", live.Base, live)
	}
	if live == deviceOverride {
		t.Fatalf("nil override device-less fit took the injected device override %+v", deviceOverride)
	}
	if liveDevice := serveStreamedHostFit(be, nil); liveDevice.Base <= 0 {
		t.Fatalf("nil override device arm fit base %d is not a live host budget (%+v)", liveDevice.Base, liveDevice)
	}

	// The consequence the defect denied: a HOST-scale budget sizes the streamed resident bound to
	// at most 0.90 * host, and the plan admits host-side instead of producing the 64.38-vs-48.89
	// refusal. The device-scale override riding in must not change any of it.
	ws := serveStreamedSynthWeightSource(t)
	hostFit := serveFitBudget{Base: hostBudgetBytes, Headroom: 0}
	bound := serveCPUOffloadStreamedResidentBound(hostFit)
	maxBound := int64(float64(hostFit.avail()) * (1 - serveCPUOffloadStreamedResidentMargin))
	if bound > maxBound {
		t.Fatalf("streamed resident bound %d exceeds 0.9 * host budget %d; the bound was sized from the device override", bound, maxBound)
	}

	// An artificial routed set bigger than the host budget forces the streamed arm so the margin and
	// the admission are exercised against real KV/scratch rows -- in-memory, no checkpoint needed.
	residentFit := serveFitBudget{Base: hostBudgetBytes * 4, Headroom: 0}
	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, residentFit)
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if resident.HostTotal() <= 0 {
		t.Fatal("fixture must host-scope the routed experts (HostTotal>0)")
	}
	forced := serveFitBudget{Base: resident.HostTotal() / 2, Headroom: 0}
	plan, streamed, err := serveStreamedCPUOffloadPlan(ws, nil, 1, 0, forced)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPlan: %v", err)
	}
	if !streamed {
		t.Fatalf("streamed policy did not fire with host budget %d against routed set %d", forced.avail(), resident.HostTotal())
	}
	if plan.HostTotal() > maxBoundFor(forced) {
		t.Fatalf("streamed host total %d exceeds the bounded resident set %d; a device override could not have admitted this", plan.HostTotal(), maxBoundFor(forced))
	}
	if err := refuseHostPlanAgainstFit(plan, forced); err != nil {
		t.Fatalf("streamed plan sized from the host fit was NOT admitted host-side: %v (the 64.38-vs-48.89 refusal class, fak#13142)", err)
	}
}

// maxBoundFor is the 0.90 margin ceiling for a budget, used to phrase the host admission assertion
// without re-deriving the constant inline.
func maxBoundFor(fit serveFitBudget) int64 {
	return int64(float64(fit.avail()) * (1 - serveCPUOffloadStreamedResidentMargin))
}

// TestServeCPUOffloadSharedPoolStreamsOnGrandTotal is the fak#13172 follow-on regression at the
// integrated-pool arm-selection seam. serveStreamedCPUOffloadPlan judged ONLY plan.HostTotal()
// against host avail -- correct on a DISCRETE device, where the device dense side is independent
// VRAM. On an INTEGRATED/APU tier (ggufload.BackendSharesHostRAM: "integrated:<device>") the device
// dense side and the host-resident expert set draw from ONE physical DRAM pool, so a plan whose HOST
// subset fits avail while its GRAND total (device transit + host resident) exceeds the pool is
// admitted on the host subset and then fails closed on the unified-host-residency bound -- the
// witnessed strix3 refusal: weights=63.09 GiB + offload(host)=43.86 GiB = 107.08 GiB against a
// 62.4 GiB MemTotal, with HostTotal 43.86 <= avail 48.72 so the resident arm was kept.
//
// The shared-pool arm must instead SELECT streaming on the GRAND total and size the bounded resident
// expert set from what REMAINS after the device dense transit, so the streamed plan's simultaneous
// physical footprint fits the pool.
func TestServeCPUOffloadSharedPoolStreamsOnGrandTotal(t *testing.T) {
	ws := serveStreamedSynthWeightSource(t)
	if !serveStreamedExpertsCapable(ws) {
		t.Fatal("fixture routed slabs are not stageable; the shared-pool fault path has no work here")
	}

	// The resident plan over an unbounded budget: its device dense side and its host-scoped routed
	// set are the two terms the shared pool must hold simultaneously.
	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if resident.DeviceTotal() <= 0 || resident.HostTotal() <= 0 {
		t.Fatalf("fixture must charge both a device dense side and a host routed side; plan=%+v", resident)
	}

	// The pool is sized so the HOST subset ALONE fits (the bug's trigger), but the GRAND total does
	// not: avail is at least HostTotal but strictly below DeviceTotal+HostTotal. This is exactly the
	// witnessed shape (host subset <= avail, grand total > pool).
	pool := resident.HostTotal() + resident.DeviceTotal()/2
	fit := serveFitBudget{Base: pool, Headroom: 0}
	if fit.avail() < resident.HostTotal() {
		t.Fatalf("fixture pool %d must admit the host subset %d to reproduce the shared-pool trap", fit.avail(), resident.HostTotal())
	}
	if resident.DeviceTotal()+resident.HostTotal() <= fit.avail() {
		t.Fatalf("fixture pool %d must NOT admit the grand total %d", fit.avail(), resident.DeviceTotal()+resident.HostTotal())
	}

	// 1) The discrete arm keeps the resident plan (P4 preserved): the device dense side is not
	// charged against host RAM there.
	discretePlan, discreteStreamed, err := serveStreamedCPUOffloadPlanForPool(ws, nil, 1, 0, fit, false)
	if err != nil {
		t.Fatalf("discrete plan: %v", err)
	}
	if discreteStreamed {
		t.Fatalf("discrete arm streamed although the host subset %d fits avail %d; the historical trigger must be unchanged", resident.HostTotal(), fit.avail())
	}
	if discretePlan.HostTotal() != resident.HostTotal() {
		t.Fatalf("discrete plan host total %d, want the resident plan %d (byte-identical)", discretePlan.HostTotal(), resident.HostTotal())
	}

	// 2) The shared-pool arm SELECTS streaming on the grand total -- the bug's fix.
	plan, streamed, err := serveStreamedCPUOffloadPlanForPool(ws, nil, 1, 0, fit, true)
	if err != nil {
		t.Fatalf("shared-pool plan: %v", err)
	}
	if !streamed {
		t.Fatalf("shared-pool arm did NOT stream although grand total %d exceeds avail %d and the host subset fits", resident.DeviceTotal()+resident.HostTotal(), fit.avail())
	}
	// The device dense side is charged IDENTICALLY, so the device fit check cannot disagree.
	if plan.DeviceTotal() != resident.DeviceTotal() {
		t.Fatalf("streamed device total %d, want the unchanged dense side %d", plan.DeviceTotal(), resident.DeviceTotal())
	}
	// The simultaneous physical footprint (device transit + bounded resident set) must fit the pool.
	if plan.DeviceTotal()+plan.HostTotal() > fit.avail() {
		t.Fatalf("streamed simultaneous footprint %d (device %d + host %d) exceeds the shared pool %d", plan.DeviceTotal()+plan.HostTotal(), plan.DeviceTotal(), plan.HostTotal(), fit.avail())
	}
	if plan.HostTotal() <= 0 {
		t.Fatalf("bounded resident expert set collapsed to zero; plan=%+v", plan)
	}
	// The device dense side consumed part of the pool, so the resident expert bound must be strictly
	// smaller than the whole-avail bound the discrete case uses (it is sized from the remainder).
	fullBound := serveCPUOffloadStreamedResidentBound(fit)
	sharedBound := serveCPUOffloadSharedPoolResidentBound(fit, resident.DeviceTotal())
	if sharedBound >= fullBound {
		t.Fatalf("shared-pool bound %d is not below the whole-avail bound %d; the device transit was not subtracted", sharedBound, fullBound)
	}
}

// TestServeCPUOffloadStreamedResidentBoundForApertureSplitDoesNotCollapse is the fak#13175
// split-aperture regression at the pure bound seam. On the witnessed strix3 box the device-scoped
// dense side (63.09 GiB) is LARGER than the host window it is judged against (48.57 GiB avail), so
// the shared-pool resident bound computed `avail - deviceTransit` as NEGATIVE and floored to zero
// (stream-through). The arm selection cannot distinguish that from "streaming not selected", so the
// serve fell through to a dense-only FitTooBig and the forward was never entered. But strix3's
// device side has its OWN aperture (an 84.28 GiB stable device-local heap vs a 62.4 GiB MemTotal),
// so the dense transit is a TRANSIENT host staging cost (fak#13171), not a permanent co-resident
// claim: with splitAperture=true the bound is the whole-avail resident set (non-zero) so the
// streamed arm is admissible; with splitAperture=false the historical subtraction is preserved
// byte-for-byte.
func TestServeCPUOffloadStreamedResidentBoundForApertureSplitDoesNotCollapse(t *testing.T) {
	const gib = int64(1 << 30)
	// The witnessed strix3 arithmetic: a dense transit larger than the host window.
	fit := serveFitBudget{Base: 48*gib + gib/2, Headroom: 0} // ~48.5 GiB host avail
	transit := 63*gib + gib/10                               // ~63.1 GiB device-destined dense transit
	if transit <= fit.avail() {
		t.Fatalf("fixture transit %d must exceed avail %d to reproduce the collapse", transit, fit.avail())
	}

	// 1) Whole-transit (no distinct aperture) accounting is UNCHANGED: the bound floors at zero,
	// exactly as before this leaf. This is the rung that produced the dense-only refusal.
	if got := serveCPUOffloadStreamedResidentBoundForPool(fit, transit, true); got != 0 {
		t.Fatalf("whole-transit shared-pool bound = %d, want the 0 stream-through floor (preserved)", got)
	}
	if got := serveCPUOffloadStreamedResidentBoundForAperture(fit, transit, true, false); got != 0 {
		t.Fatalf("non-split aperture bound = %d, want the 0 stream-through floor (preserved)", got)
	}

	// 2) With a SPLIT aperture the transient transit is NOT subtracted: the bound is the whole-avail
	// resident set, non-zero, so the streamed arm has a working set to charge.
	got := serveCPUOffloadStreamedResidentBoundForAperture(fit, transit, true, true)
	want := serveCPUOffloadStreamedResidentBound(fit)
	if got != want {
		t.Fatalf("split-aperture bound = %d, want the whole-avail resident set %d (transit not subtracted)", got, want)
	}
	if got <= 0 || got >= fit.avail() {
		t.Fatalf("split-aperture bound = %d, want a nonzero set strictly below avail %d", got, fit.avail())
	}

	// 3) A DISCRETE sharedPool=false call is untouched regardless of the aperture flag.
	if got := serveCPUOffloadStreamedResidentBoundForAperture(fit, transit, false, true); got != serveCPUOffloadStreamedResidentBound(fit) {
		t.Fatalf("discrete bound changed under splitAperture: %d", got)
	}
}

// TestServeCPUOffloadSharedPoolResidentBoundFloorsAtZero pins the honest floor: a device transit that
// already at or above the pool (or an unprobeable pool) yields a zero bound (stream-through) rather
// than a negative one.
func TestServeCPUOffloadSharedPoolResidentBoundFloorsAtZero(t *testing.T) {
	fit := serveFitBudget{Base: 1 << 30, Headroom: 0}
	if got := serveCPUOffloadSharedPoolResidentBound(fit, fit.avail()); got != 0 {
		t.Fatalf("transit at avail: bound = %d, want 0 (stream-through floor)", got)
	}
	if got := serveCPUOffloadSharedPoolResidentBound(fit, fit.avail()+1); got != 0 {
		t.Fatalf("transit over avail: bound = %d, want 0", got)
	}
	if got := serveCPUOffloadSharedPoolResidentBound(serveFitBudget{}, 1); got != 0 {
		t.Fatalf("unprobeable pool: bound = %d, want 0", got)
	}
	// A genuine remainder keeps the declared margin strictly below it.
	got := serveCPUOffloadSharedPoolResidentBound(fit, 1<<29)
	remaining := fit.avail() - (1 << 29)
	if got <= 0 || got >= remaining {
		t.Fatalf("remainder bound = %d, want a nonzero set strictly below %d", got, remaining)
	}
}

// TestServeStreamedDecisionSharedPoolBoundMatchesPlan is the shared-pool "one measurement" regression:
// the load arm's WithStreamedExperts working set (the bound returned by
// serveStreamedCPUOffloadPathDecision) and the sizing plan's charged host set must be the SAME
// derivation. Before the fix the decision recomputed the bound with the DISCRETE helper even when the
// shared-pool (grand-total) arm had selected streaming, so the bound threaded to the loader was the
// whole-avail bound -- larger than the shared-pool bound the plan charged against the device transit.
func TestServeStreamedDecisionSharedPoolBoundMatchesPlan(t *testing.T) {
	path := writeServeStreamedSynthGGUF(t, "glm-moe-dsa-shared-pool.gguf")

	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	ws.Close()
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if resident.DeviceTotal() <= 0 || resident.HostTotal() <= 0 {
		t.Fatalf("fixture must charge both a device dense side and a host routed side; plan=%+v", resident)
	}

	// The shared-pool trap: the host subset fits, the grand total does not.
	fit := serveFitBudget{Base: resident.HostTotal() + resident.DeviceTotal()/2, Headroom: 0}
	if fit.avail() < resident.HostTotal() || resident.DeviceTotal()+resident.HostTotal() <= fit.avail() {
		t.Fatalf("fixture pool %d does not reproduce the shared-pool trap (host %d, grand %d)", fit.avail(), resident.HostTotal(), resident.DeviceTotal()+resident.HostTotal())
	}

	decided, bound, err := serveStreamedCPUOffloadPathDecision(path, nil, 1, 0, fit, true)
	if err != nil {
		t.Fatalf("serveStreamedCPUOffloadPathDecision(sharedPool=true): %v", err)
	}
	if !decided {
		t.Fatalf("shared-pool decision did NOT stream (host %d fits avail %d, grand %d does not)", resident.HostTotal(), fit.avail(), resident.DeviceTotal()+resident.HostTotal())
	}

	want := serveCPUOffloadSharedPoolResidentBound(fit, resident.DeviceTotal())
	if bound != want {
		t.Fatalf("decision bound %d != the shared-pool bound the plan uses %d; the load arm and sizing path disagree (one-measurement invariant violated)", bound, want)
	}
	wholeAvail := serveCPUOffloadStreamedResidentBound(fit)
	if bound >= wholeAvail {
		t.Fatalf("decision bound %d was not reduced below the whole-avail bound %d despite the device transit", bound, wholeAvail)
	}
}

// TestServeStreamedDenseAppliesBoundedWorkingSet is the fak#13205 acceptance at the serve option
// list: when the env asks for streamed dense (FAK_STREAM_Q4K=1) the option list serveQ4KFitOptions
// builds must enable the streamed DENSE route in its BOUNDED form -- StreamedDenseQ4K &&
// StreamedDenseBounded -- carrying the SAME budget serveStreamedDenseQ4KWorkingSetBound derives from
// the fit snapshot, so the loader retains a bounded host working set instead of the full 63.22 GiB
// device dense side that kernel-OOM-killed staging (the fak#13205 bug). It asserts the EFFECT of the
// list via ApplyQ4KLoadOptions (the loader's own option-application path), not slice length.
func TestServeStreamedDenseAppliesBoundedWorkingSet(t *testing.T) {
	t.Setenv("FAK_STREAM_Q4K", "1")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_GGUF_MMAP", "0")

	path := createTestQ4KGGUF(t)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	t.Cleanup(func() { _ = ws.Close() })

	fit := serveFitBudget{Base: 1 << 40, Headroom: 0.10}
	opts := serveQ4KFitOptions(path, ws, nil, serveLoadArmResidentQ4K, fit)
	eff := ggufload.ApplyQ4KLoadOptions(opts)
	if !eff.StreamedDenseQ4K {
		t.Fatalf("serveQ4KFitOptions did not enable the streamed dense route with FAK_STREAM_Q4K=1; effects=%+v", eff)
	}
	if !eff.StreamedDenseBounded {
		t.Fatalf("streamed dense route was enabled UNBOUNDED; the bounded working set must be selected (fak#13205); effects=%+v", eff)
	}
	want := serveStreamedDenseQ4KWorkingSetBound(serveStreamedHostFit(nil, &fit))
	if eff.StreamedDenseBytes != want {
		t.Fatalf("streamed dense working set=%d, want the single derivation %d (one-measurement invariant)", eff.StreamedDenseBytes, want)
	}
	if eff.StreamedDenseBytes <= 0 || eff.StreamedDenseBytes >= fit.avail() {
		t.Fatalf("streamed dense working set=%d must be nonzero and strictly below avail=%d", eff.StreamedDenseBytes, fit.avail())
	}
}

// TestServeStreamedDenseWorkingSetBoundIsStrictlyBelowAvail pins the bound derivation: an
// unprobeable host (Base<=0) yields zero (stream-through, the honest floor), a probeable budget
// yields a nonzero working set STRICTLY below avail, and no input yields a negative bound.
func TestServeStreamedDenseWorkingSetBoundIsStrictlyBelowAvail(t *testing.T) {
	if got := serveStreamedDenseQ4KWorkingSetBound(serveFitBudget{}); got != 0 {
		t.Fatalf("unprobeable host: bound = %d, want 0 (stream-through)", got)
	}
	if got := serveStreamedDenseQ4KWorkingSetBound(serveFitBudget{Base: -1, Headroom: 0.10}); got != 0 {
		t.Fatalf("negative base: bound = %d, want 0", got)
	}
	for _, fit := range []serveFitBudget{
		{Base: 1 << 40, Headroom: 0},
		{Base: 1 << 40, Headroom: 0.10},
		{Base: 48 << 30, Headroom: 0.10},
	} {
		got := serveStreamedDenseQ4KWorkingSetBound(fit)
		if got <= 0 {
			t.Fatalf("probeable budget %+v: bound = %d, want a nonzero working set", fit, got)
		}
		if got >= fit.avail() {
			t.Fatalf("probeable budget %+v: bound = %d is not strictly below avail = %d", fit, got, fit.avail())
		}
	}
	// The razor-thin budget must still land strictly below avail and never go negative.
	thin := serveStreamedDenseQ4KWorkingSetBound(serveFitBudget{Base: 4, Headroom: 0})
	if thin < 0 || thin >= 4 {
		t.Fatalf("thin budget: bound = %d, want [0,4)", thin)
	}
}

// TestServeSplitApertureDetection pins the split-aperture predicate at the capacity seam: a device
// whose reported window DIFFERS from the process host window is split (strix3: 84.28 GiB heap vs
// 62.4 GiB MemTotal); a nil/unknown backend and a backend reporting the SAME total as the host
// window (a unified heap) report false, so the whole-transit accounting is preserved for them.
// This reuses the established ggufload split test (deviceTotal != hostTotal), not a ratio.
func TestServeSplitApertureDetection(t *testing.T) {
	if serveSplitAperture(nil) {
		t.Fatal("nil backend reported a split aperture; must fail open to false")
	}
	hostTotal, _, known := compute.HostSystemMemoryInfo()
	if !known || hostTotal <= 0 {
		t.Skip("host memory unprobeable on this runner; the split-aperture predicate cannot be exercised")
	}
	if !serveSplitAperture(splitApertureBackend{total: hostTotal + hostTotal/4, free: hostTotal + hostTotal/4, known: true}) {
		t.Fatal("a device window distinct from the host window must report split=true")
	}
	if serveSplitAperture(splitApertureBackend{total: hostTotal, free: hostTotal, known: true}) {
		t.Fatal("a device window equal to the host window (unified heap) must report split=false")
	}
	if serveSplitAperture(splitApertureBackend{known: false}) {
		t.Fatal("a backend reporting no device window must report split=false")
	}
}

// splitApertureBackend is a minimal compute.Backend double that reports a device memory window,
// so TestServeSplitApertureDetection can exercise the predicate without a device.
type splitApertureBackend struct {
	compute.Backend
	total int64
	free  int64
	known bool
}

func (b splitApertureBackend) DeviceMemory() (int64, int64, bool) { return b.total, b.free, b.known }
func (b splitApertureBackend) Caps() compute.Caps                 { return compute.Caps{CapacityProbe: true} }
