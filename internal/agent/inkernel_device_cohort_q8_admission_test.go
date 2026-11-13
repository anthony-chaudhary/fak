package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_device_cohort_q8_admission_test.go is the [SW-VERIFIED] witness for ticket
// prefill-qwen35-gdn-concurrent-batching-20261108 (gap 2 of the matched-BF16 receipt):
// the inkernel CUDA qwen35-gdn serve path must admit concurrent requests into one
// batched step on the Q8_0-resident (non-Q4_K) load arm instead of serializing every
// request on the whole-turn devMu route. The fixture models that arm on a synthetic
// host: a cpu-ref HAL behind a wrapper that advertises the exact independent-decode
// batch contract without executing one, so no physical GPU witness can be manufactured
// here; the admission gate, the production Complete route and the coordinator mechanics
// are what these tests pin. The N=4/N=16 aggregate h2h cell stays [HW-GATED].

const q8CohortFreeBytes = int64(1) << 30

// q8ResidentDeviceBackend wraps a capacity-probed cpu-ref HAL with the exact CUDA
// independent-decode batch marker plus the stub GDN seam the HAL loop requires. Its
// Qwen35IndependentDecodeBatch must never execute: the concurrency regression stands in
// for the device batch forward with the shared-receipt probe, and any other multi-lane
// dispatch reaching it panics loudly.
type q8ResidentDeviceBackend struct {
	capacityProbeBackend
}

func (*q8ResidentDeviceBackend) Qwen35GDNPath() string { return model.Qwen35GDNCUDAPath }

func (b *q8ResidentDeviceBackend) Qwen35GDNDecode(
	normalizedInput,
	_, _, _, _,
	_, _, _, _, _,
	convState, recurrentState compute.Tensor,
	_, _, _, _, _ int,
	_ float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	// Pipe-model stand-in for the GDN core: pass the normalized input through and keep
	// the in-place state identity the HAL enforces. Both A/B arms run this same stand-in,
	// so token identity between them is unaffected.
	output := compute.NewF32(b.Backend, append([]int(nil), normalizedInput.Shape...), append([]float32(nil), b.Read(normalizedInput)...))
	return output, convState, recurrentState, nil
}

// CloneTensor delegates to the wrapped HAL: the GDN owner privatizes recurrent
// state through it before every forward.
func (b *q8ResidentDeviceBackend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	return b.Backend.(compute.TensorCloner).CloneTensor(t)
}

func (*q8ResidentDeviceBackend) Qwen35IndependentDecodeBatchPath() string {
	return compute.Qwen35IndependentDecodeBatchPath
}

func (*q8ResidentDeviceBackend) Qwen35IndependentDecodeBatch(compute.Qwen35IndependentDecodeBatchRequest) (compute.Qwen35IndependentDecodeBatchResult, error) {
	panic("pipe-model fixture must not execute a batch")
}

// q8ResidentCohortModel builds the served-arm shape: a quantized Qwen3.5-hybrid
// synthetic (three GDN layers plus one gated full-attention layer, the arch family's
// 3-of-4 rhythm), the shape `fak serve --backend cuda` constructs for a BF16
// safetensors checkpoint (Q8_0-resident, inKernelQ4K=false).
func q8ResidentCohortModel(t *testing.T) *model.Model {
	t.Helper()
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	cfg.NumLayers = 4
	cfg.LayerTypes = []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"}
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = 8
	cfg.LinearNumKeyHeads = 2
	cfg.LinearValueHeadDim = 8
	cfg.LinearNumValueHeads = 4
	m := model.NewSynthetic(cfg)
	m.Quantize()
	return m
}

// q8ResidentCohortPlanner builds a device-serialized planner over the shared model:
// q4k=false (the matched-BF16 arm's load-time label), BatchDecode=true (both serve
// arms set it), and either the exact batch capability (capable) or the plain
// capacity-probed HAL (control).
func q8ResidentCohortPlanner(t *testing.T, m *model.Model, capable bool) *InKernelPlanner {
	t.Helper()
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend unavailable")
	}
	probe := capacityProbeBackend{Backend: base, total: q8CohortFreeBytes, free: q8CohortFreeBytes, known: true}
	var backend compute.Backend = probe
	if capable {
		backend = &q8ResidentDeviceBackend{capacityProbeBackend: probe}
	}
	p := NewInKernelPlannerWithConfig(m, loadProbeTok(t), "synthetic-qwen-q8-resident", false, backend, false, InKernelPlannerConfig{BatchDecode: true})
	p.maxNew = 8
	return p
}

// fak-test:runtime fast est=50ms lane=default
func TestInKernelQwenDeviceCohortAdmitsNonQ4KResidentArm(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	m := q8ResidentCohortModel(t)
	p := q8ResidentCohortPlanner(t, m, true)
	if !p.requiresDeviceSerialization() {
		t.Fatal("fixture must be a device-serialized planner")
	}
	base := p.backend.(*q8ResidentDeviceBackend).capacityProbeBackend
	valid := &q8ResidentDeviceBackend{capacityProbeBackend: base}
	var typedNil *q8ResidentDeviceBackend
	for _, tc := range []struct {
		name    string
		q4k     bool
		backend compute.Backend
		metal   bool
		batch   bool
		want    bool
	}{
		{"q8-resident-exact-capability", false, valid, false, true, true},
		{"q8-resident-unsupported", false, base, false, true, false},
		{"q8-resident-wrong-marker", false, &cohortCapabilityBackend{capacityProbeBackend: base, path: "wrong"}, false, true, false},
		{"q8-resident-typed-nil", false, typedNil, false, true, false},
		{"q4k-exact-capability", true, valid, false, true, true},
		{"q8-resident-disabled", false, valid, false, false, false},
		{"metal-host-q4k", true, nil, true, true, true},
		{"metal-host-q8", false, nil, true, true, false},
		{"cpu-host", false, nil, false, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p.backend, p.metal, p.batchDecode, p.q4k = tc.backend, tc.metal, tc.batch, tc.q4k
			if got := p.coalescesQwenDecode(); got != tc.want {
				t.Fatalf("cohort admission = %t, want %t", got, tc.want)
			}
		})
	}
}

// TestInKernelQwenDeviceCohortRouteDivergesByCapability witnesses the production
// Complete route at the changed gate: the capable q8-resident arm charges the
// coalesced in-flight device reservation (the cohort route) while the incapable
// control prices the request at the ordinary serial precheck site. No forward runs:
// both refusals land before any session exists.
// fak-test:runtime fast est=100ms lane=default
func TestInKernelQwenDeviceCohortRouteDivergesByCapability(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	capable := q8ResidentCohortPlanner(t, q8ResidentCohortModel(t), true)
	capable.deviceBudgetMu.Lock()
	capable.deviceReservedBytes = q8CohortFreeBytes
	capable.deviceBudgetMu.Unlock()
	_, err := capable.Complete(context.Background(), []Message{{Role: RoleUser, Content: "alpha"}}, nil)
	var capacityErr *InKernelCapacityError
	if !errors.As(err, &capacityErr) || capacityErr.Scope != compute.MemoryScopeDevice || capacityErr.Site != inKernelCoalescedDeviceReservationSite {
		t.Fatalf("capable q8-resident Complete = %T (%v), want closed coalesced reservation refusal", err, err)
	}
	capable.deviceBudgetMu.Lock()
	held := capable.deviceReservedBytes
	capable.deviceBudgetMu.Unlock()
	if held != q8CohortFreeBytes {
		t.Fatalf("refused admission mutated reservation to %d, want %d", held, q8CohortFreeBytes)
	}

	plain := q8ResidentCohortPlanner(t, q8ResidentCohortModel(t), false)
	probe := plain.backend.(capacityProbeBackend)
	probe.free = 1
	plain.backend = probe
	plain.deviceReservedBytes = 1
	_, err = plain.Complete(context.Background(), []Message{{Role: RoleUser, Content: "alpha"}}, nil)
	var serialErr *InKernelCapacityError
	if !errors.As(err, &serialErr) || serialErr.Scope != compute.MemoryScopeDevice || serialErr.Site != "capacity-precheck" {
		t.Fatalf("incapable q8-resident Complete = %T (%v), want serial capacity-precheck", err, err)
	}
}

// TestInKernelQwenDeviceCohortConcurrencyRegression is the sim/pipe-harness
// concurrency regression the ticket's acceptance gate names: N concurrent requests on
// the Q8_0-resident device arm must be admitted into ONE decode cohort whose batched
// steps stream each projection once per round, instead of serializing on the whole-turn
// devMu route where the aggregate equals single-stream. The probe stands in for the
// device batch forward (real native per-lane Step for token advancement, pipe-modeled
// shared-panel credit for the amortization). Wall-clock figures are diagnostic software
// receipts ([SW-VERIFIED], timing_diagnostic_only); the N=4/N=16 aggregate h2h cell
// stays [HW-GATED] until a GPU host runs it.
// fak-test:runtime fast est=2s lane=default
func TestInKernelQwenDeviceCohortConcurrencyRegression(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	t.Setenv(batchStepHealthEnv, "0")
	const N = 4
	m := q8ResidentCohortModel(t)
	prompts := make([][]Message, N)
	for i := range prompts {
		content := fmt.Sprintf("concurrency regression lane %d ", i) + strings.Repeat("prefill payload ", 24)
		prompts[i] = []Message{{Role: RoleUser, Content: content}}
	}

	// Serial arm: N sequential turns on the same planner shape. This is exactly the
	// aggregate the pre-fix concurrent traffic produced - devMu serialized every
	// request into one-at-a-time single-lane episodes.
	serial := q8ResidentCohortPlanner(t, m, true)
	serialContent := make([]string, N)
	serialStart := time.Now()
	for i := 0; i < N; i++ {
		comp, err := serial.Complete(context.Background(), prompts[i], nil)
		if err != nil {
			t.Fatalf("serial turn %d: %v", i, err)
		}
		if comp.Usage.CompletionTokens != serial.maxNew {
			t.Fatalf("serial turn %d generated %d tokens, want %d", i, comp.Usage.CompletionTokens, serial.maxNew)
		}
		serialContent[i] = comp.Message.Content
	}
	serialTotal := time.Since(serialStart)

	// Concurrent arm: N simultaneous turns through the production admission. The
	// probe stands in for the device batch forward - real native per-lane Step for
	// token advancement, pipe-modeled shared-panel credit for the amortization - and
	// reports the credit the CUDA independent-decode-batch contract computes: one
	// panel per projection per round regardless of lane count, MAC charge linear in
	// the active lane count. The ready hook holds the elected leader until the whole
	// fan-out is queued so the cohort forms deterministically.
	concurrent := q8ResidentCohortPlanner(t, m, true)
	restoreProbe := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
		out := make([][]float32, len(ids))
		activeCount := 0
		for i, on := range active {
			if !on {
				continue
			}
			out[i] = bs.Seqs[i].Step(ids[i])
			activeCount++
		}
		// Pipe model of the device batch forward's accounting: one shared
		// [lanes, hidden] projection panel per layer group plus the output head,
		// streamed ONCE per round regardless of lane count (panel count is
		// lane-invariant), with the MAC charge linear in the active lane count.
		roundPanels := bs.M.Cfg.NumLayers + 1
		return out, roundPanels, int64(roundPanels) * int64(activeCount), true
	})
	defer restoreProbe()

	// Solo-lane reference through the same probe, BEFORE the fan-out hook arms: one
	// lane's episode credits the same per-round panel count as the N-lane cohort
	// (lane-invariance) and one lane's share of the MAC charge. It also anchors the
	// token-identity baseline on this planner.
	soloComp, err := concurrent.Complete(context.Background(), prompts[0], nil)
	if err != nil {
		t.Fatalf("solo reference turn: %v", err)
	}
	if soloComp.InKernelBatch == nil || soloComp.InKernelBatch.CohortSize != 1 {
		t.Fatalf("solo reference carried no single-lane cohort receipt: %+v", soloComp.InKernelBatch)
	}
	soloReceipt := *soloComp.InKernelBatch
	if soloReceipt.SharedPanels <= 0 || soloReceipt.SharedMACs <= 0 {
		t.Fatalf("solo reference credited no shared work: %+v", soloReceipt)
	}
	if soloComp.Message.Content != serialContent[0] {
		t.Fatalf("solo reference diverged from the serial arm:\n got %q\nwant %q", soloComp.Message.Content, serialContent[0])
	}

	concurrent.coalesceReadyHook = func() {
		deadline := time.NewTimer(5 * time.Second)
		defer deadline.Stop()
		for {
			concurrent.coalesceMu.Lock()
			n := len(concurrent.coalesceReady)
			concurrent.coalesceMu.Unlock()
			if n == N {
				return
			}
			select {
			case <-deadline.C:
				t.Error("cohort fan-out never queued")
				return
			default:
				time.Sleep(time.Millisecond)
			}
		}
	}
	type answer struct {
		content string
		receipt InKernelBatchReceipt
		haveRec bool
		err     error
	}
	answers := make([]answer, N)
	var wg sync.WaitGroup
	start := make(chan struct{})
	fireStart := time.Now()
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			comp, err := concurrent.Complete(context.Background(), prompts[i], nil)
			if err != nil {
				answers[i] = answer{err: err}
				return
			}
			answers[i] = answer{content: comp.Message.Content, haveRec: comp.InKernelBatch != nil}
			if comp.InKernelBatch != nil {
				answers[i].receipt = *comp.InKernelBatch
			}
		}(i)
	}
	close(start)
	wg.Wait()
	concurrentTotal := time.Since(fireStart)

	// Amortization law: every lane ran the same number of decode rounds, so the
	// N-lane cohort's panel credit must equal the solo lane's (each projection
	// streams once per round regardless of lane count) while its MAC charge must be
	// exactly N times the solo lane's.
	var cohortID uint64
	for i := range answers {
		a := answers[i]
		if a.err != nil {
			t.Fatalf("concurrent turn %d: %v", i, a.err)
		}
		if !a.haveRec {
			t.Fatalf("concurrent turn %d carried no batch receipt: the cohort did not form", i)
		}
		if a.receipt.CohortSize != N {
			t.Fatalf("concurrent turn %d cohort size=%d, want %d", i, a.receipt.CohortSize, N)
		}
		if cohortID == 0 {
			cohortID = a.receipt.CohortID
		}
		if a.receipt.CohortID != cohortID {
			t.Fatalf("concurrent turn %d cohort id=%d, want the shared cohort %d", i, a.receipt.CohortID, cohortID)
		}
		if a.receipt.SharedPanels != soloReceipt.SharedPanels {
			t.Fatalf("concurrent turn %d shared panels=%d, want %d (panel count is lane-invariant: each projection streams once per round)",
				i, a.receipt.SharedPanels, soloReceipt.SharedPanels)
		}
		if a.receipt.SharedMACs != int64(N)*soloReceipt.SharedMACs {
			t.Fatalf("concurrent turn %d shared macs=%d, want %d (MAC charge is linear in the active lane count)",
				i, a.receipt.SharedMACs, int64(N)*soloReceipt.SharedMACs)
		}
		if a.content != serialContent[i] {
			t.Fatalf("concurrent turn %d diverged from the single-stream baseline:\n got %q\nwant %q", i, a.content, serialContent[i])
		}
	}

	// Pipe-model throughput law: the CUDA independent-decode-batch contract
	// charges one shared projection panel per layer group per round regardless
	// of lane count (lane-invariant, asserted above) plus a MAC charge linear
	// in the active lane count. Costing both episodes in that charge, the
	// N-lane cohort's aggregate throughput must exceed the single-stream
	// episode's whenever panel streaming costs anything:
	//   N*T/(P + N*M) > T/(P + M)  <=>  N*(P+M) > (P+N*M)  (integer form below)
	// where P = solo panel credit, M = solo MAC credit, T = maxNew. That is the
	// load-invariant sim witness for "aggregate > single-stream". The
	// wall-clock A/B stays diagnostic only: the CPU stand-in executes every
	// lane's native Step inside the probe, so its per-round host cost grows
	// linearly in lanes and can invert under host contention - the device
	// weight-streaming amortization this ticket admits is carried by the
	// panel/MAC charge, not by sim wall time.
	soloCost := int64(soloReceipt.SharedPanels) + soloReceipt.SharedMACs
	cohortCost := int64(answers[0].receipt.SharedPanels) + answers[0].receipt.SharedMACs
	soloTokens := int64(concurrent.maxNew)
	cohortTokens := int64(N) * soloTokens
	if int64(N)*(soloCost) <= cohortCost {
		t.Fatalf("sim aggregate did not exceed single-stream: cohort %d tok over %d cost vs solo %d tok over %d cost",
			cohortTokens, cohortCost, soloTokens, soloCost)
	}
	wallRatio := float64(serialTotal) / float64(concurrentTotal)
	t.Logf("[SW-VERIFIED] SOFTWARE RECEIPT q8_resident_cohort_ab n=%d max_new=%d sim_solo=%dtok/%dcost sim_cohort=%dtok/%dcost verdict=AGGREGATE_SCALES "+
		"cohort_size=%d solo_panels=%d solo_macs=%d cohort_panels=%d cohort_macs=%d token_identity=ok "+
		"wall_serial=%s wall_concurrent=%s wall_ratio=%.3f timing_diagnostic_only=true hw_gated=true",
		N, concurrent.maxNew, soloTokens, soloCost, cohortTokens, cohortCost, N,
		soloReceipt.SharedPanels, soloReceipt.SharedMACs,
		answers[0].receipt.SharedPanels, answers[0].receipt.SharedMACs,
		serialTotal, concurrentTotal, wallRatio)
}
