package main

import (
	"context"
	"flag"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serveFlagsForBatchWiring builds the minimal serve flag set needed to resolve a
// native planner config. Only the native-control flags that feed the planner are
// registered, so the test drives the REAL serveNativeControlConfig seam rather
// than a hand-built InKernelPlannerConfig.
func serveFlagsForBatchWiring(t *testing.T) *serveFlags {
	t.Helper()
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse serve flags: %v", err)
	}
	return sf
}

// TestServeNativePlannerEnablesBatchDecode is the #13263 witness: `fak serve`'s
// native chat planner must be wired for the continuous-batch decode coalescer,
// exactly as `fak up` is (#1590), so concurrent same-prefix turns share one
// batched forward on the eligible native path instead of serializing on devMu.
// The coalescer's device gates (q4k/metal/hybrid) still decide at run time, so
// this is a structural opt-in, not a device claim.
func TestServeNativePlannerEnablesBatchDecode(t *testing.T) {
	sf := serveFlagsForBatchWiring(t)
	cfg := serveNativePlannerConfig(sf)

	p := agent.NewInKernelPlannerWithConfig(nil, nil, "serve-batch-wiring-probe", false, nil, false, cfg)
	if p == nil {
		t.Fatal("planner is nil")
	}
	if !p.BatchDecodeEnabled() {
		t.Fatal("fak serve native planner did not enable the continuous-batch decode coalescer (#13263): concurrent turns would serialize on devMu")
	}
}

// TestServePlannerCoalescesConcurrentTurns proves the wiring is REACHED, not just
// declared: given the serve planner config, N concurrent same-prefix requests
// coalesce onto ONE shared cohort through the real coalescer. It reuses the same
// synthetic Qwen3.5-hybrid, device-free seam as the `fak up` witness, and relaxes
// only the device half (metal/q4k) via the exported test seam.
func TestServePlannerCoalescesConcurrentTurns(t *testing.T) {
	tok := testProbeTokenizer(t)
	const fanout = 4
	prompt := "shared-prefix serve fan-out probe"

	mkModel := func() *fakmodel.Model {
		m := fakmodel.NewSynthetic(fakmodel.Config{
			HiddenSize:        32,
			NumLayers:         2,
			NumHeads:          4,
			NumKVHeads:        2,
			HeadDim:           8,
			IntermediateSize:  64,
			VocabSize:         320,
			RMSNormEps:        1e-5,
			RopeTheta:         10000,
			TieWordEmbeddings: true,
			EOSTokenID:        -1,
			LayerTypes:        []string{"linear_attention"},

			LinearConvKernelDim: 3,
			LinearKeyHeadDim:    8,
			LinearNumKeyHeads:   2,
			LinearValueHeadDim:  8,
			LinearNumValueHeads: 4,
		})
		m.Quantize()
		return m
	}

	sf := serveFlagsForBatchWiring(t)
	cfg := serveNativePlannerConfig(sf)

	serial := agent.NewInKernelPlannerWithConfig(mkModel(), tok, "serve-serial", false, nil, false, agent.InKernelPlannerConfig{})
	ref, err := serial.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: prompt}}, nil)
	if err != nil {
		t.Fatalf("serial complete: %v", err)
	}

	p := agent.NewInKernelPlannerWithConfig(mkModel(), tok, "serve-coalesced", false, nil, false, cfg)
	if !p.BatchDecodeEnabled() {
		t.Fatal("serve planner config must enable batch decode")
	}
	restoreSeam := p.AdmitCoalescedDecodeForTest()
	defer restoreSeam()

	released := make(chan struct{})
	restoreHook := p.SetCoalesceReadyHookForTest(func() { <-released })
	defer restoreHook()

	results := make([]*agent.Completion, fanout)
	errs := make([]error, fanout)
	var wg sync.WaitGroup
	for i := 0; i < fanout; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = p.Complete(
				context.Background(),
				[]agent.Message{{Role: agent.RoleUser, Content: prompt}},
				nil,
			)
		}(i)
	}
	waitForCoalesceReady(t, p, fanout, 10*time.Second)
	close(released)
	wg.Wait()

	cohorts := map[uint64]int{}
	for i := 0; i < fanout; i++ {
		if errs[i] != nil {
			t.Fatalf("coalesced[%d]: %v", i, errs[i])
		}
		if results[i].Message.Content != ref.Message.Content {
			t.Fatalf("coalesced[%d] content %q != serial %q", i, results[i].Message.Content, ref.Message.Content)
		}
		r := results[i].InKernelBatch
		if r == nil {
			t.Fatalf("coalesced[%d] carried no batch receipt: serve path bypassed the coalescer", i)
		}
		if r.CohortSize < 2 {
			t.Fatalf("coalesced[%d] cohort size %d, want >= 2", i, r.CohortSize)
		}
		cohorts[r.CohortID]++
	}
	if len(cohorts) != 1 {
		t.Fatalf("serve fan-out split across %d cohorts (%v), want one shared cohort", len(cohorts), cohorts)
	}
}

// TestServePlannerBatchDecodeReceiptSeam is a compile-time guard: the serve flag
// set stays parseable without a live process. Kept tiny so it never masks the
// two behavioral tests above.
func TestServePlannerBatchDecodeReceiptSeam(t *testing.T) {
	fs, _ := newServeFlagSet()
	if fs == nil {
		t.Fatal("newServeFlagSet returned nil")
	}
	if fs.Lookup("native") == nil {
		t.Fatal("serve flag set is missing the --native flag")
	}
	_ = flag.CommandLine
}
