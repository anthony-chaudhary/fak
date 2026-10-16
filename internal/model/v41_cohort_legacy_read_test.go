package model

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Exercise the actual public cold-prefill consumer with an ordinary legacy
// reader. This backend cannot mint compute's private independent-storage grant;
// these cases prove defensive-copy lifetime and public failure ordering only.
// fak-test:runtime medium est=20s lane=default
func TestV41ColdPrefillLegacyReadIsolation(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "1")
	for _, name := range []string{
		"healthy_scratch_poison", "transport_dominates", "row_count", "verdict_count",
		"ordered_verdict_before_bad_row", "late_short_row", "late_non_bf16_row",
	} {
		t.Run(name, func(t *testing.T) {
			newSession := func(enabled bool) (*Session, *v41EntryPointCohortWitness) {
				t.Helper()
				m := v41PinnedSlateFixture(t, false)
				b := &v41EntryPointCohortWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{
					v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: enabled,
				}}
				b.ordinary = newV41DeferredPublishRecorder()
				b.ordinary.self = b
				s, err := m.NewBackendSessionChecked(b)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(s.Close)
				return s, b
			}
			s, b := newSession(true)
			healthy := name == "healthy_scratch_poison"
			transport := errors.New("legacy cohort read transport")
			later := errors.New("later legacy cohort verdict")
			type retainedRow struct {
				scratch []float32
				frees   int
			}
			retained := map[compute.Buffer]*retainedRow{}
			targetReads, poisoned := 0, 0
			// These count actual backend contractions, activations, publications
			// and reads, independently of the model's phase ledger and hook count.
			operations := func() [4]int { return [4]int{b.matmuls, b.calls, b.publishes, b.deferredReads} }
			var atRead [4]int
			b.routedRead = func(statuses []compute.V41DeferredStatus, rows []compute.Tensor, verdicts []error, values [][]float32, err error) ([]error, [][]float32, error) {
				if targetReads != 0 || len(rows) < 2 {
					return verdicts, values, err
				}
				targetReads++
				atRead = operations()
				if s.v41RoutedOutputs != nil || err != nil || len(values) != len(rows) || len(statuses) != 2*len(rows) || len(verdicts) != len(statuses) {
					t.Fatal("target is not a completed legacy cold multirow read without retained device outputs")
				}
				batch := b.batches[len(b.batches)-1]
				if len(batch.weights) != 3*len(rows) || batch.readRows != len(rows) {
					t.Fatal("target did not execute every routed gate/up/down contraction")
				}
				for r, row := range rows {
					a, d := statuses[2*r], statuses[2*r+1]
					if a.Kind != compute.V41DeferredRoutedSwiGLU || d.Kind != compute.V41DeferredBF16Publish || d.Site != compute.V41BF16PublishRoutedDown ||
						a.Count != s.M.Cfg.MoEIntermediateSize || d.Count != s.M.Cfg.HiddenSize ||
						row.Buf() == nil || retained[row.Buf()] != nil || len(values[r]) != s.M.Cfg.HiddenSize || !v41BF16Widened(values[r]) {
						t.Fatalf("target row %d lacks real routed status/publication geometry", r)
					}
					retained[row.Buf()] = &retainedRow{scratch: values[r]}
				}
				for i, st := range statuses {
					if _, pending := b.pending[st.Record.Buf()]; pending || verdicts[i] != nil {
						t.Fatalf("target status %d was not consumed successfully before injection", i)
					}
				}
				switch name {
				case "transport_dominates":
					verdicts[0] = &compute.V41RoutedSwiGLUFault{Stage: compute.V41RoutedSwiGLUStageActivation, Index: 0}
					values[len(values)-1] = values[len(values)-1][:1]
					return verdicts, values, transport
				case "row_count":
					values = values[:len(values)-1]
				case "verdict_count":
					verdicts = verdicts[:len(verdicts)-1]
				case "ordered_verdict_before_bad_row":
					verdicts[0] = &compute.V41RoutedSwiGLUFault{Stage: compute.V41RoutedSwiGLUStageActivation, Index: 0}
					verdicts[len(verdicts)-1] = later
					values[len(values)-1] = values[len(values)-1][:1]
				case "late_short_row":
					values[len(values)-1] = values[len(values)-1][:s.M.Cfg.HiddenSize-1]
				case "late_non_bf16_row":
					values[len(values)-1][0] = math.Float32frombits(0x3f800001)
				}
				return verdicts, values, err
			}
			b.afterFree = func(x compute.Tensor) {
				row := retained[x.Buf()]
				if row == nil {
					return
				}
				if s.v41RoutedOutputs != nil {
					t.Fatal("retained device output could bypass the legacy host rows")
				}
				row.frees++
				if row.frees != 1 {
					t.Fatal("target published output freed more than once")
				}
				if healthy {
					// The hook runs after the real Free. Corrupt precisely the reader's
					// returned scratch, after the consumer must have made its copy.
					for i := range row.scratch {
						row.scratch[i] = float32(math.NaN())
					}
					poisoned++
				}
			}
			var got []float32
			var caught any
			returned := false
			func() {
				defer func() { caught = recover() }()
				got = s.Prefill([]int{1, 2, 3})
				returned = true
			}()
			if targetReads != 1 || len(retained) < 2 || len(b.pending) != 0 || len(b.ordinary.deferredPending) != 0 {
				t.Fatalf("target reads=%d retained rows=%d pending=%d/%d", targetReads, len(retained), len(b.pending), len(b.ordinary.deferredPending))
			}
			for _, row := range retained {
				if row.frees != 1 {
					t.Fatal("Prefill exited before a targeted publication was freed")
				}
			}
			if healthy {
				if !returned || caught != nil || len(got) == 0 || poisoned != len(retained) || s.BackendSessionClosed() {
					t.Fatalf("healthy return=%t panic=%v logits=%d poison=%d/%d closed=%t", returned, caught, len(got), poisoned, len(retained), s.BackendSessionClosed())
				}
				for _, row := range retained {
					for _, v := range row.scratch {
						if !math.IsNaN(float64(v)) {
							t.Fatal("legacy scratch was not poisoned before Prefill returned")
						}
					}
				}
				control, serial := newSession(false)
				want := control.Prefill([]int{1, 2, 3})
				if !v41SameBits(got, want) || len(serial.cohortRows) != 0 || serial.calls == 0 {
					t.Fatal("poisoned cold Prefill differs from independent serial session")
				}
				for _, v := range got {
					if !finite32(v) {
						t.Fatal("poisoned backend scratch escaped into public logits")
					}
				}
				return
			}
			failure, ok := caught.(error)
			if returned || got != nil || !ok || operations() != atRead {
				t.Fatalf("failed Prefill returned=%t logits=%v panic=%v operations=%v want %v", returned, got, caught, operations(), atRead)
			}
			if name == "ordered_verdict_before_bad_row" {
				var numeric *V41ExpertOperationError
				if !errors.As(failure, &numeric) || numeric.Layer != 0 || numeric.Stage != "gate/up" || !errors.Is(failure, errV41ExpertResult) || errors.Is(failure, later) {
					t.Fatalf("first numeric verdict lost to a later verdict or malformed row: %v", failure)
				}
				return // Numeric-only refusal need not retire the backend session.
			}
			var selected *BackendForwardOperationError
			if !errors.As(failure, &selected) || selected.Stage != "readback" || !s.BackendSessionClosed() {
				t.Fatalf("legacy readback failure was not retired: %v closed=%t", failure, s.BackendSessionClosed())
			}
			if name == "transport_dominates" {
				if !errors.Is(failure, transport) || errors.Is(failure, errV41ExpertResult) {
					t.Fatalf("transport did not dominate partial verdicts/rows: %v", failure)
				}
			} else if !errors.Is(failure, errV41ExpertResult) {
				t.Fatalf("malformed readback lost the exact result error: %v", failure)
			}
		})
	}
}
