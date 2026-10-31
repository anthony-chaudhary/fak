package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type cudaSequenceLockProfileBackend struct {
	compute.Backend
	available bool
	snapshots int
}

func (b *cudaSequenceLockProfileBackend) CUDAQwen35SequenceLockProfileSnapshot() (uint64, uint64, uint64, bool) {
	b.snapshots++
	return 3, 11, 29, b.available
}

// fak-test:runtime fast est=300ms lane=default
// Exercise the real completion/receipt consumer using a CPU backend double;
// this does not claim that a CUDA sequence or physical device was executed.
func TestNativeCUDASequenceLockProfileReceiptGating(t *testing.T) {
	for _, tc := range []struct {
		name      string
		requested bool
		available bool
	}{
		{"ordinary-completion", false, true},
		{"profile-unavailable", true, false},
		{"profile-available", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tinyConcurrencyConfig()
			cfg.EOSTokenID = -1
			m := model.NewSynthetic(cfg)
			t.Cleanup(func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
			be := &cudaSequenceLockProfileBackend{Backend: compute.Default(), available: tc.available}
			p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-sequence-lock-receipt", false, be, false)
			comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "receipt"}}, nil,
				WithMaxTokens(1), WithNativeInferenceReceipt(tc.requested))
			if err != nil {
				t.Fatal(err)
			}
			if !tc.requested {
				if comp.NativeInference != nil || be.snapshots != 0 {
					t.Fatalf("ordinary completion observed profiling: receipt=%+v calls=%d", comp.NativeInference, be.snapshots)
				}
				return
			}
			if comp.NativeInference == nil || be.snapshots != 1 {
				t.Fatalf("requested receipt/snapshot calls = %+v/%d", comp.NativeInference, be.snapshots)
			}
			raw, err := json.Marshal(comp.NativeInference)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(raw, &fields); err != nil {
				t.Fatal(err)
			}
			block, present := fields["cuda_qwen35_sequence_lock_profile"]
			if present != tc.available {
				t.Fatalf("profile presence=%t want %t: %s", present, tc.available, raw)
			}
			if present {
				var got model.NativeCUDASequenceLockProfile
				if err := json.Unmarshal(block, &got); err != nil {
					t.Fatal(err)
				}
				want := model.NativeCUDASequenceLockProfile{Scope: "process_cuda_qwen35_sequence_prefill",
					Accounting: "cumulative_completed_acquisitions", Gate: "cuda_timestamp_profile",
					CompletedAcquisitions: 3, WaitNS: 11, HoldNS: 29}
				if got != want {
					t.Fatalf("profile = %+v, want %+v", got, want)
				}
			}
		})
	}
}
