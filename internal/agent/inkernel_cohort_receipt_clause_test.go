package agent

import (
	"bytes"
	"context"
	"log"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInKernelCohortReceiptClause pins the execution summary's cohort clause:
// a coalesced turn renders cohort=<id>/lanes=<peak>/panels=<shared> from its
// batch receipt; a non-coalesced turn (CohortID 0) renders nothing, so every
// pre-existing summary line and its parsers stay byte-identical.
// fak-test:runtime fast est=10ms lane=default
func TestInKernelCohortReceiptClause(t *testing.T) {
	if got := inKernelCohortReceiptClause(InKernelBatchReceipt{}); got != "" {
		t.Fatalf("zero receipt clause = %q, want empty", got)
	}
	want := " cohort=7/lanes=4/panels=96"
	if got := inKernelCohortReceiptClause(InKernelBatchReceipt{CohortID: 7, CohortSize: 4, SharedPanels: 96}); got != want {
		t.Fatalf("cohort clause = %q, want %q", got, want)
	}
}

// TestInKernelExecutionLogAppendsCohortReceipt pins the threading: the summary
// line carries the caller-supplied cohort clause between the decode rate and
// the trailing gemm=/kv= keys, and appends nothing for a non-coalesced turn.
// fak-test:runtime fast est=10ms lane=default
func TestInKernelExecutionLogAppendsCohortReceipt(t *testing.T) {
	m := model.NewSynthetic(model.Config{LayerTypes: []string{"linear_attention"}})
	p := &InKernelPlanner{m: m, modelID: "Qwen3.8-27B-Q4_K_M", metal: true, q4k: true}
	var buf bytes.Buffer
	old := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(old) })

	p.logExecutionSummary("neon", "", 22, 3, 0, 22, 6.41, 3.4, 3, 1.07, 2.8, "")
	if got := strings.TrimSpace(buf.String()); strings.Contains(got, "cohort=") {
		t.Fatalf("non-coalesced execution log %q carries a cohort clause", got)
	}
	buf.Reset()
	p.logExecutionSummary("neon", "", 22, 3, 0, 22, 6.41, 3.4, 3, 1.07, 2.8, " cohort=7/lanes=4/panels=96")
	got := strings.TrimSpace(buf.String())
	for _, want := range []string{"cohort=7/lanes=4/panels=96", " gemm=", " kv=fp32"} {
		if !strings.Contains(got, want) {
			t.Fatalf("execution log %q missing %q", got, want)
		}
	}
	if i, j := strings.Index(got, "cohort=7/lanes=4/panels=96"), strings.Index(got, " gemm="); i < 0 || j < i {
		t.Fatalf("execution log %q reordered the cohort clause ahead of gemm=", got)
	}
}

// TestInKernelCoalescedTurnSummaryLogsCohortReceipt is the end-to-end witness
// the device co-batching receipt needs (anthony-chaudhary/fak#8395): a real
// two-request fan-out through the decode coalescer logs one summary line per
// turn carrying cohort=<id>/lanes=2/panels=<shared>, so a served run can PROVE
// the coalescer engaged (and at what width) from its serve log instead of
// inferring it from aggregate tok/s. The probe-replaced batch forward keeps
// this witness free of any physical device.
// fak-test:runtime fast est=100ms lane=default
func TestInKernelCoalescedTurnSummaryLogsCohortReceipt(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	t.Setenv(batchStepHealthEnv, "0")
	restoreProbe := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
		out := make([][]float32, len(ids))
		lanes := 0
		for i, on := range active {
			if !on {
				continue
			}
			out[i] = bs.Seqs[i].Step(ids[i])
			lanes++
		}
		return out, 1, int64(lanes), true
	})
	t.Cleanup(restoreProbe)

	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	cfg.LayerTypes = []string{"linear_attention"}
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = 8
	cfg.LinearNumKeyHeads = 2
	cfg.LinearValueHeadDim = 8
	cfg.LinearNumValueHeads = 4
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-qwen-cohort-summary", true, nil, false)
	p.maxNew = 3
	p.batchDecode = true
	p.metal = true // synthetic witness for the production host gate

	var buf bytes.Buffer
	old := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(old) })

	const fanOut = 2
	release := make(chan struct{})
	var once sync.Once
	p.coalesceReadyHook = func() {
		// The elected leader blocks until the whole fan-out is queued, so the
		// cohort forms deterministically instead of racing the drainer.
		deadline := time.NewTimer(2 * time.Second)
		defer deadline.Stop()
		for {
			p.coalesceMu.Lock()
			n := len(p.coalesceReady)
			p.coalesceMu.Unlock()
			if n == fanOut {
				break
			}
			select {
			case <-deadline.C:
				t.Error("cohort fan-out never queued")
				return
			default:
				time.Sleep(time.Millisecond)
			}
		}
		once.Do(func() { close(release) })
		<-release
	}

	type answer struct {
		completion *Completion
		err        error
	}
	answers := make([]answer, fanOut)
	var wg sync.WaitGroup
	for i := 0; i < fanOut; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			answers[i].completion, answers[i].err = p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "cohort summary"}}, nil)
		}(i)
	}
	wg.Wait()

	for i, got := range answers {
		if got.err != nil || got.completion == nil || got.completion.InKernelBatch == nil {
			t.Fatalf("completion[%d]=%#v err=%v", i, got.completion, got.err)
		}
		if got.completion.InKernelBatch.CohortSize != fanOut {
			t.Fatalf("completion[%d] cohort size=%d, want %d", i, got.completion.InKernelBatch.CohortSize, fanOut)
		}
	}
	lines := regexp.MustCompile(`cohort=\d+/lanes=2/panels=\d+`).FindAllString(buf.String(), -1)
	if len(lines) < fanOut {
		t.Fatalf("summary log %q carries %d cohort clauses, want >= %d", buf.String(), len(lines), fanOut)
	}
}
