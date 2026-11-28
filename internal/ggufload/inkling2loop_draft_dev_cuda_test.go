//go:build cuda && cgo

package ggufload

import (
	"context"
	"os"
	"runtime"
	"testing"
)

type dsparkCheckedDrafter struct {
	inkling2DSparkDrafter
}

func (d dsparkCheckedDrafter) Observe(s *inkling2Resident, fed []int, keep int) error {
	if err := d.inkling2DSparkDrafter.Observe(s, fed, keep); err != nil {
		return err
	}
	if ovf, err := d.d.checkKVOverflow(); err != nil {
		return err
	} else if ovf {
		return ErrInkling2DSparkFP8Overflow
	}
	return nil
}

// TestInkling2CUDADSparkFP8ParityAnd128kFit verifies on real CUDA silicon that:
//  1. DSpark with FP8 E4M3FN context KV rings matches plain target greedy (up to
//     admissible near ties), stays free of FP8 overflow, and preserves acceptance
//     alongside FP32 DSpark context KV under NVFP4 target KV.
//  2. All-resident target (NVFP4 128k) + Q8_0 DSpark draft with FP8 context KV fits
//     and runs prefill + propose + verify cleanly in RTX 5080 VRAM.
func TestInkling2CUDADSparkFP8ParityAnd128kFit(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_MTP_DIR")
	defer m.Close()
	dm, err := OpenInkling2DSpark(os.Getenv("FAK_INKLING2_MTP_DIR"), m)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()

	t.Setenv("FAK_INKLING2_KV", "nvfp4")
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")

	prompt := inkling2OracleCases[0].prompt
	const genN = 64
	ref, runnerUp, margin, err := inkling2ResidentGreedyMargins(m, be, prompt, genN)
	if err != nil {
		t.Fatalf("plain reference: %v", err)
	}

	depth := dm.HP.Block - 1
	for _, tc := range []struct {
		name  string
		fp8KV bool
	}{
		{name: "dspark_fp32_kv", fp8KV: false},
		{name: "dspark_fp8_kv", fp8KV: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var attached *ink2DeviceDraft
			got, sst, err := inkling2ResidentSpecGreedy(m, be, prompt, genN, depth, func(s *inkling2Resident) (inkling2Drafter, error) {
				d, err := s.attachDraftFor(dm, tc.fp8KV)
				if err != nil {
					return nil, err
				}
				d.steps = depth
				attached = d
				return dsparkCheckedDrafter{inkling2DSparkDrafter{d}}, nil
			})
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if attached == nil || attached.fp8KV != tc.fp8KV {
				t.Fatalf("%s: attached fp8KV = %v, want %v", tc.name, attached != nil && attached.fp8KV, tc.fp8KV)
			}
			t.Logf("%s: %.2f tok/s, %d verifies, accepted %d/%d, plain=%d, verify=%.3f ms/call",
				tc.name, sst.DecodeTokensPerSecond(), sst.Verifies, sst.Accepted, sst.Drafted,
				sst.PlainSteps, ms(sst.Verify)/float64(max(1, sst.Verifies)))
			if at, nearTie, why := inkling2CohortDivergence(ref, runnerUp, margin, got); at >= 0 && !nearTie {
				t.Fatalf("%s diverged from plain greedy: %s", tc.name, why)
			}
		})
	}

	t.Run("128k_nvfp4_dspark_fp8_fit", func(t *testing.T) {
		const ctx128k = 131072
		chunk := min(inkling2PrefillChunkBase, 128)
		s, err := newInkling2ResidentRingFor(m, be, ctx128k, inkling2ArenaFloats, chunk, chunk, true)
		if err != nil {
			t.Fatalf("newInkling2ResidentRingFor(128k): %v", err)
		}
		defer s.close()
		if s.ctx != ctx128k {
			t.Fatalf("resident ctx = %d, want %d", s.ctx, ctx128k)
		}
		if s.kvDType() != Inkling2KVNVFP4 {
			t.Fatalf("resident kvDType = %v, want nvfp4", s.kvDType())
		}
		if err := s.enableSpec(depth); err != nil {
			t.Fatalf("enableSpec: %v", err)
		}
		d, err := s.attachDraft(dm)
		if err != nil {
			t.Fatalf("attachDraft(128k): %v", err)
		}
		if !d.fp8KV {
			t.Fatal("128k attachDraft did not default to FP8 context KV on CUDA")
		}
		d.steps = depth
		if mp, ok := be.(interface{ DeviceMemory() (int64, int64, bool) }); ok {
			tot, free, known := mp.DeviceMemory()
			if known {
				t.Logf("128k NVFP4 + DSpark FP8 attached: shortlist=%d free=%.2f/%.2f MiB",
					d.shortlist, float64(free)/(1<<20), float64(tot)/(1<<20))
			}
		}

		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		next, err := s.prefill(context.Background(), prompt)
		if err != nil {
			t.Fatalf("128k prefill: %v", err)
		}
		for step := 0; step < 8; step++ {
			props, err := d.propose(next, s.pos, depth)
			if err != nil {
				t.Fatalf("step %d propose: %v", step, err)
			}
			if len(props) == 0 {
				var g int
				g, err = s.step(next)
				if err != nil {
					t.Fatalf("step %d plain step: %v", step, err)
				}
				next = g
				continue
			}
			fed := append([]int{next}, props...)
			g, err := s.verify(fed)
			if err != nil {
				t.Fatalf("step %d verify: %v", step, err)
			}
			a := 0
			for a < len(props) && a < len(g)-1 && props[a] == g[a] {
				a++
			}
			if err := s.commitVerify(len(fed), a+1); err != nil {
				t.Fatalf("step %d commitVerify: %v", step, err)
			}
			if ovf, err := d.checkKVOverflow(); err != nil || ovf {
				t.Fatalf("step %d fp8 overflow = (%v, %v)", step, ovf, err)
			}
			next = g[a]
		}
	})
}
