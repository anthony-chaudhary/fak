package main

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// serve_model_fit_streamed_context_test.go — context-sizing regression for the cpu-offload-experts
// arm. On a DISCRETE device that arm's KV store is DEVICE-resident while the routed experts stream
// from NVMe into a bounded HOST-resident working set. Two defects broke that split in the
// DeepSeek-V4.1 Flash repro (fak-private ticket aegis-27b-model-capacity):
//
//  1. AUTO path: fitServeStreamedCPUOffloadPathOnDevice sized the context rows against the HOST
//     budget that SELECTS the streamed arm (~22.93 GiB), not the device window the KV lands in
//     (~13.22 GiB) — 86,962 tokens / 19.904 GiB of KV against a 13.22 GiB window, refused by the
//     load-time device check.
//
//  2. EXPLICIT/auto resolver path: the sizing-inputs streamed branch returned a plan that already
//     carried context rows sized against the host pool as "weights"; the resolver subtracted that
//     stale device-scoped KV from the device budget, went negative, and clamped an explicit
//     --native-context-tokens 8192 to the 512-token floor.
//
// The fix makes both paths share one budget math: the context rows are sized against the device
// window (serveStreamedCPUOffloadPlanForAperture's contextFit), and serveNativeContextSizingInputs'
// streamed branch returns the WEIGHTS-only core plan (serveStreamedCPUOffloadCorePlan via
// serveCPUOffloadSizingStreamedPlan). These tests pin both halves at unit scale: a scaled Q2_K
// routed payload whose host budget forces streaming, beside small known device windows.

// writeServeStreamedScaledSynthGGUF serializes the cpu-offload-experts MoE fixture with a SCALED
// routed-expert payload (experts x hidden Q2_K slabs, true block math) and a WIDE declared context
// window (1 Mi tokens), so a test can build a host budget that forces the streamed arm while the
// device window derives a context far above the sizer's 512-token floor — the miniature shape of
// the DeepSeek-V4.1 Flash repro. It mirrors writeServeStreamedSynthGGUF byte-for-byte except the
// two parameters and the window.
func writeServeStreamedScaledSynthGGUF(t *testing.T, base string, experts, hidden uint64) string {
	t.Helper()
	slabBytes := uint64(experts) * uint64(hidden) * uint64(hidden) / 256 * 84
	align32 := func(n uint64) uint64 { return (n + 31) &^ 31 }

	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 4, 14)
	writeKVStringForTest(&b, "general.architecture", "glm_moe_dsa")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint64ForTest(&b, "glm_moe_dsa.context_length", 1<<20)
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
		t.Fatalf("writeServeStreamedScaledSynthGGUF: %v", err)
	}
	return path
}

// servePlanDenseRowTotal is the plan's device-scoped dense weight charge (the
// gguf-device-dense-load rows), excluding any context-sized rows a caller may have appended:
// serveGGUFCPUOffloadMemoryPlan sizes its context rows against the passed budget (fail-open to the
// full declared window when unprobeable), so a plan's raw DeviceTotal is not the dense side alone.
func servePlanDenseRowTotal(plan compute.MemoryPlan) int64 {
	var total int64
	for _, d := range plan {
		if d.Detail == "gguf-device-dense-load" {
			total += d.Bytes
		}
	}
	return total
}

// assertWeightsOnlyPlan pins the serveNativeContextSizingInputs contract: the plan handed to the
// resolver carries NO context-sized rows — no KV store, no per-token HAL transients. A plan that
// already carries context rows sized against ANOTHER pool makes the resolver subtract that stale
// KV from the pool it budgets (defect #2 above).
func assertWeightsOnlyPlan(t *testing.T, plan compute.MemoryPlan, where string) {
	t.Helper()
	for _, d := range plan {
		switch d.Class {
		case compute.MemoryKVCache, compute.MemoryScratchpad, compute.MemoryActivation:
			t.Fatalf("%s: plan carries a context-sized row %+v; the sizing contract is a weights-only plan", where, d)
		}
		if strings.HasPrefix(d.Detail, "hal-") || d.Detail == "compressed-kv-rows" {
			t.Fatalf("%s: plan carries a context detail %q; the sizing contract is a weights-only plan", where, d.Detail)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
// The auto-path half of the repro: on a discrete device the cpu-offload-experts arm's KV store is
// device-resident, so the load path must size it against the DEVICE window — not the host budget
// that selects the streamed arm and bounds its expert working set — and admit the result. Here the
// host budget is below the full routed charge (forcing streaming) and the device window is SMALLER
// than the host-derived bound, so the old host-sized plan would have been refused at load time
// while the device-sized one admits. One budget math: the auto plan and the explicit-at-bound plan
// are byte-identical.
func TestServeCPUOffloadExpertsAutoContextSizesAgainstDeviceNotHost(t *testing.T) {
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	const (
		experts = 32
		hidden  = 512
	)
	path := writeServeStreamedScaledSynthGGUF(t, "glm-moe-dsa-scaled-auto.gguf", experts, hidden)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	csc := cfg.ContextSizeConfigWithPrecision(serveKVPrecision())
	if csc.MaxContext <= 0 {
		t.Fatalf("fixture must declare a window, got %d", csc.MaxContext)
	}
	perToken := csc.KVBytesPerToken()
	if perToken <= 0 {
		t.Fatalf("fixture KV geometry unprobeable (perToken=%d)", perToken)
	}
	scratch := compute.EstimateHALTransientMemoryPlan(csc.Scratch).Total()

	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	routed := resident.HostTotal()
	if routed <= 0 {
		t.Fatal("fixture must host-scope the routed experts (HostTotal>0)")
	}

	// A host budget below the full routed charge forces the streamed arm (the #13121 trigger).
	hostFit := serveFitBudget{Base: routed / 2, Headroom: 0}

	// A device window strictly BELOW the host-derived bound: the host-sized auto context cannot
	// hold, the device-sized one can. Solve for the raw base that leaves exactly devAvail after
	// the device headroom, so sizing and admission read one measurement.
	devAvail := int64(2 << 20)
	devBase := int64(float64(devAvail) / (1 - serveGGUFDeviceHeadroom))
	for compute.BudgetAfterHeadroom(devBase, serveGGUFDeviceHeadroom) < devAvail {
		devBase++
	}
	be := serveCapBackend{Backend: compute.Default(), total: devBase, free: devBase, known: true}
	devFit := serveFitBudget{Base: devBase, Headroom: serveGGUFDeviceHeadroom}
	if !reflect.DeepEqual(serveDeviceFitBudget(be), devFit) {
		t.Fatalf("device fit = %+v, want the derived budget %+v; the test budget must match the derivation", serveDeviceFitBudget(be), devFit)
	}

	plan, streamed, err := fitServeStreamedCPUOffloadPathOnDevice(path, be, 1, 0, hostFit, nil)
	if err != nil {
		t.Fatalf("fitServeStreamedCPUOffloadPathOnDevice: %v; the device-sized plan must ADMIT (the repro's refusal was the host-sized plan)", err)
	}
	if !streamed {
		t.Fatalf("streamed policy did not fire with host budget %d against routed set %d", hostFit.avail(), routed)
	}

	// The clean weights plan the sizing seam shares (one measurement with the load arm).
	clean, sized := serveCPUOffloadSizingStreamedPlan(ws, nil, 1, hostFit)
	if !sized {
		t.Fatal("sizing seam did not select the streamed arm on the same host budget; load and sizing disagree")
	}
	tokensAuto := compute.LargestFittingContextTokens(csc, clean, devFit.avail())
	if tokensAuto <= compute.MinAutoContextTokens || tokensAuto >= csc.MaxContext {
		t.Fatalf("auto context %d must sit strictly inside (%d, %d); the fixture does not exercise the shrink", tokensAuto, compute.MinAutoContextTokens, csc.MaxContext)
	}
	// The repro's divergence, asserted: the host-derived bound exceeds the device-derived bound.
	tokensHost := compute.LargestFittingContextTokens(csc, clean, hostFit.avail())
	if tokensHost <= tokensAuto {
		t.Fatalf("host-derived bound %d does not exceed the device-derived bound %d; the fixture does not reproduce the repro shape", tokensHost, tokensAuto)
	}

	// ONE budget math: the load plan's KV is sized against the DEVICE window, not the host one.
	var kvBytes int64
	for _, d := range plan {
		if d.Class == compute.MemoryKVCache {
			kvBytes += d.Bytes
		}
	}
	if want := csc.KVBytes(tokensAuto); kvBytes != want {
		t.Fatalf("load plan KV = %d, want %d (csc.KVBytes of the device-window-derived %d tokens); the context rows were sized against the wrong pool", kvBytes, want, tokensAuto)
	}

	// The host-streamed expert working set stays OUT of the device charge: it is host-scoped, and
	// the device total is exactly the clean dense side plus the context rows.
	var streamedRow *compute.MemoryDemand
	for i := range plan {
		if plan[i].Detail == "gguf-host-expert-offload-streamed" {
			streamedRow = &plan[i]
		}
	}
	if streamedRow == nil {
		t.Fatalf("load plan carries no gguf-host-expert-offload-streamed row; plan=%+v", plan)
	}
	if streamedRow.Scope != compute.MemoryScopeHost {
		t.Fatalf("streamed expert row scope = %q, want host; the expert working set must not charge the device window", streamedRow.Scope)
	}
	wantDevice := clean.DeviceTotal() + csc.KVBytes(tokensAuto) + scratch
	if plan.DeviceTotal() != wantDevice {
		t.Fatalf("load plan device total %d, want the clean dense side plus the context rows %d; host-streamed expert bytes leaked into the device charge", plan.DeviceTotal(), wantDevice)
	}
	if plan.HostTotal() > hostFit.avail() {
		t.Fatalf("load plan host total %d exceeds the bounded resident snapshot %d", plan.HostTotal(), hostFit.avail())
	}

	// Explicit AT the fitted bound passes through verbatim: the explicit-at-bound load plan is
	// byte-identical to the auto one (same weights, same budget, same token count).
	explicit, explicitStreamed, err := serveStreamedCPUOffloadPlanForAperture(ws, be, 1, tokensAuto, hostFit, devFit, false, serveSplitAperture(be))
	if err != nil {
		t.Fatalf("explicit-at-bound plan: %v", err)
	}
	if !explicitStreamed {
		t.Fatal("explicit-at-bound plan did not select the streamed arm on the same budgets")
	}
	if !reflect.DeepEqual(plan, explicit) {
		t.Fatalf("auto load plan = %+v, want the explicit-at-bound plan %+v; auto and explicit must share one budget math", plan, explicit)
	}
}

// fak-test:runtime fast est=50ms lane=default
// The sizing-contract half of the repro: serveNativeContextSizingInputs must hand the resolver a
// WEIGHTS-only plan and the DEVICE fit budget for the device cpu-offload-experts arm. Before the
// fix the streamed branch returned a plan that already carried context rows sized against the host
// pool (defect #2); any such row breaks the resolver's largest-fitting derivation.
func TestServeCPUOffloadExpertsSizingInputsAreWeightsOnlyWithDeviceFit(t *testing.T) {
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	const (
		experts = 32
		hidden  = 512
	)
	path := writeServeStreamedScaledSynthGGUF(t, "glm-moe-dsa-sizing.gguf", experts, hidden)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	be := serveCapBackend{Backend: compute.Default(), total: 1 << 30, free: 1 << 30, known: true}
	weights, fit, err := serveNativeContextSizingInputs(ws, be, true, false, 1, 0, path)
	if err != nil {
		t.Fatalf("serveNativeContextSizingInputs: %v", err)
	}
	assertWeightsOnlyPlan(t, weights, "serveNativeContextSizingInputs(cpu-offload-experts)")
	if !reflect.DeepEqual(fit, serveDeviceFitBudget(be)) {
		t.Fatalf("sizing fit = %+v, want the DEVICE budget %+v; the resolver sizes the KV against the window it lands in", fit, serveDeviceFitBudget(be))
	}

	// The streamed branch under a forced-stream host budget: the exact seam defect #2 lived in.
	// The returned plan must stay weights-only even though the selected arm is the streamed one.
	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if resident.HostTotal() <= 0 {
		t.Fatal("fixture must host-scope the routed experts (HostTotal>0)")
	}
	hostFit := serveFitBudget{Base: resident.HostTotal() / 2, Headroom: 0}
	clean, streamed := serveCPUOffloadSizingStreamedPlan(ws, nil, 1, hostFit)
	if !streamed {
		t.Fatalf("sizing seam did not select the streamed arm with budget %d against routed set %d", hostFit.avail(), resident.HostTotal())
	}
	assertWeightsOnlyPlan(t, clean, "serveCPUOffloadSizingStreamedPlan")
	if clean.HostTotal() >= hostFit.avail() {
		t.Fatalf("streamed sizing host total %d is not strictly below avail %d; the bounded resident set ties the pool", clean.HostTotal(), hostFit.avail())
	}
	// The device dense side is charged IDENTICALLY by the streamed fold (the same per-tensor
	// classification), so the two arms cannot disagree about what stays on the device. Compare the
	// dense row, not DeviceTotal: the full-charge plan above carries fail-open full-window context
	// rows that the weights-only streamed plan does not.
	if got, want := servePlanDenseRowTotal(clean), servePlanDenseRowTotal(resident); got != want {
		t.Fatalf("streamed sizing dense side %d, want the unchanged dense side %d", got, want)
	}
}

// fak-test:runtime fast est=50ms lane=default
// The explicit-request half of the repro: with the WEIGHTS-only plan the sizing arm now returns,
// an explicit --native-context-tokens 8192 resolves to 8192 when it fits the device window, and a
// request above the fitted bound clamps to THAT bound (tens of thousands of tokens, not the 512
// floor). The second half rebuilds the pre-fix polluted plan (weights + context rows sized against
// the host pool) and shows the same explicit request collapsing under it — the 512-token symptom
// witnessed on the repro box.
func TestServeCPUOffloadExpertsExplicitContextResolvesAgainstCleanWeights(t *testing.T) {
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	const (
		experts = 32
		hidden  = 512
	)
	path := writeServeStreamedScaledSynthGGUF(t, "glm-moe-dsa-explicit.gguf", experts, hidden)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	csc := cfg.ContextSizeConfigWithPrecision(serveKVPrecision())
	if csc.MaxContext <= 0 {
		t.Fatalf("fixture must declare a window, got %d", csc.MaxContext)
	}
	scratch := compute.EstimateHALTransientMemoryPlan(csc.Scratch).Total()

	resident, err := serveGGUFCPUOffloadMemoryPlan(ws, 1, 0, serveFitBudget{})
	if err != nil {
		t.Fatalf("resident plan: %v", err)
	}
	if resident.HostTotal() <= 0 {
		t.Fatal("fixture must host-scope the routed experts (HostTotal>0)")
	}
	// A host budget below the full routed charge forces the streamed arm.
	hostFit := serveFitBudget{Base: resident.HostTotal() / 2, Headroom: 0}
	clean, streamed := serveCPUOffloadSizingStreamedPlan(ws, nil, 1, hostFit)
	if !streamed {
		t.Fatalf("sizing seam did not select the streamed arm with budget %d against routed set %d", hostFit.avail(), resident.HostTotal())
	}

	// The device window: it admits 8192 tokens of KV on top of the clean weights, but NOT the
	// host-sized context rows the pre-fix plan carried.
	devAvail := int64(4_000_000)
	devBase := int64(float64(devAvail) / (1 - serveGGUFDeviceHeadroom))
	for compute.BudgetAfterHeadroom(devBase, serveGGUFDeviceHeadroom) < devAvail {
		devBase++
	}
	devFit := serveFitBudget{Base: devBase, Headroom: serveGGUFDeviceHeadroom}
	be := serveCapBackend{Backend: compute.Default(), total: devBase, free: devBase, known: true}

	fittedClean := compute.LargestFittingContextTokens(csc, clean, devFit.avail())
	if fittedClean < 8192 {
		t.Fatalf("fixture device window %d does not admit the explicit 8192 request (fitted=%d)", devFit.avail(), fittedClean)
	}

	// The pre-fix polluted plan: the SAME weights plus context rows sized against the HOST pool —
	// what appendServeGGUFDevicePlan used to append before the core/wrapper split.
	tokensHost := compute.LargestFittingContextTokens(csc, clean, hostFit.avail())
	dirty := append(append(compute.MemoryPlan(nil), clean...), csc.PerContextMemoryPlan(tokensHost)...)
	if dirty.DeviceTotal()+scratch <= devFit.avail() {
		t.Fatalf("fixture budgets do not reproduce the repro shape: polluted device total %d must exceed the device window %d", dirty.DeviceTotal()+scratch, devFit.avail())
	}

	// Clean weights: the explicit request passes through verbatim.
	resClean, _, err := resolveServeNativeContext(ws, be, clean, devFit, 8192)
	if err != nil {
		t.Fatalf("resolveServeNativeContext(clean): %v", err)
	}
	if resClean.ResolvedTokens != 8192 || resClean.Source != "explicit" {
		t.Fatalf("clean weights: resolved=%d source=%q, want 8192/explicit; the repro's clamp-to-512 was the polluted plan, not the sizer", resClean.ResolvedTokens, resClean.Source)
	}

	// Above the fitted bound: clamps to the bound, never below it.
	resClamped, _, err := resolveServeNativeContext(ws, be, clean, devFit, csc.MaxContext)
	if err != nil {
		t.Fatalf("resolveServeNativeContext(clamp): %v", err)
	}
	if resClamped.ResolvedTokens != fittedClean {
		t.Fatalf("above-bound request: resolved=%d, want the fitted bound %d", resClamped.ResolvedTokens, fittedClean)
	}
	if resClamped.ResolvedTokens <= compute.MinAutoContextTokens {
		t.Fatalf("above-bound request collapsed to the %d floor instead of the fitted bound %d", compute.MinAutoContextTokens, fittedClean)
	}

	// Polluted weights: the same explicit request collapses — the witnessed symptom.
	resDirty, _, err := resolveServeNativeContext(ws, be, dirty, devFit, 8192)
	if err != nil {
		t.Fatalf("resolveServeNativeContext(dirty): %v", err)
	}
	if resDirty.ResolvedTokens >= 8192 {
		t.Fatalf("polluted weights: resolved=%d still honors the explicit request; the fixture did not reproduce the repro", resDirty.ResolvedTokens)
	}
	if want := compute.LargestFittingContextTokens(csc, dirty, devFit.avail()); resDirty.ResolvedTokens != want {
		t.Fatalf("polluted weights: resolved=%d, want the sizer's own derivation %d given the polluted input (the defect was the input, not the sizer)", resDirty.ResolvedTokens, want)
	}
}
