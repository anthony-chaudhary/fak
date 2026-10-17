package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41GroupedPanelChunks(t *testing.T) {
	for _, tc := range []struct {
		name  string
		rows  int
		allow map[int]bool
		want  []int
	}{
		{"qualified_256", 256, map[int]bool{64: true}, []int{64, 64, 64, 64}},
		{"qualified_tail", 127, map[int]bool{64: true, 32: true, 16: true}, []int{64, 32, 16, 8, 4, 2, 1}},
		{"admitted_tail", 127, map[int]bool{64: true, 63: true}, []int{64, 63}},
		{"admitted_511", 511, map[int]bool{64: true, 63: true}, []int{64, 64, 64, 64, 64, 64, 64, 63}},
		{"power_only_511", 511, map[int]bool{64: true, 32: true, 16: true}, []int{64, 64, 64, 64, 64, 64, 64, 32, 16, 8, 4, 2, 1}},
		{"tail_only", 63, map[int]bool{63: true}, []int{63}},
		{"rejected_tail", 63, map[int]bool{32: true, 16: true}, []int{32, 16, 8, 4, 2, 1}},
		{"nonmonotone_tail", 63, map[int]bool{32: true, 31: true}, []int{32, 31}},
		{"lower_bound", 9, map[int]bool{9: true}, []int{9}},
		{"below_bound", 7, map[int]bool{7: true}, []int{4, 2, 1}},
		{"above_bound", 65, map[int]bool{65: true, 64: true}, []int{64, 1}},
		{"limited_32", 80, map[int]bool{32: true, 16: true}, []int{32, 32, 16}},
		{"nonmonotone", 96, map[int]bool{64: true}, []int{64, 8, 8, 8, 8}},
		{"legacy", 31, nil, []int{8, 8, 8, 4, 2, 1}},
		{"scalar_tail", 1, nil, []int{1}},
		{"empty", 0, nil, nil},
		{"negative", -1, nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := map[int]int{}
			got := v41GroupedPanelChunks(tc.rows, func(n int) bool {
				calls[n]++
				if n <= 8 || n > 64 || calls[n] > 1 {
					t.Fatalf("unexpected admission count: rows=%d calls=%d", n, calls[n])
				}
				return tc.allow[n]
			})
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("chunks=%v want=%v", got, tc.want)
			}
		})
	}
}

// Integration witness: requires the model package fixtures.
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputWidenedChunkParity(t *testing.T) {
	for _, tc := range []struct{ rows, chunks int }{{9, 1}, {15, 1}, {16, 1}, {31, 1}, {32, 1}, {63, 1}, {64, 1}, {65, 2}, {127, 2}, {256, 4}, {511, 8}} {
		t.Run(itoa(tc.rows), func(t *testing.T) {
			m, s, b := v41GroupedPanelFixture(t)
			defer s.Close()
			input := v41GroupedPanelInput(tc.rows, m.Cfg.NumHeads*m.Cfg.HeadDim)
			want := v41GroupedPanelCacheOracle(t, m, input, tc.rows)
			got, outcome, err := v41GroupedPanelCacheCall(s, input, tc.rows)
			if err != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) {
				t.Fatalf("outcome=%d err=%v parity=%t", outcome, err, reflect.DeepEqual(got, want))
			}
			if b.rowReads != tc.chunks || b.gathers != tc.chunks || b.matmuls != tc.chunks*(m.Cfg.OGroups+1) {
				t.Fatalf("reads/gathers/matmuls=%d/%d/%d want=%d/%d/%d", b.rowReads, b.gathers, b.matmuls, tc.chunks, tc.chunks, tc.chunks*(m.Cfg.OGroups+1))
			}
		})
	}
}

// CPU-backed selected-operation failure/ownership witness, not hardware timing.
// fak-test:runtime fast est=500ms lane=default
func TestV41GroupedOutputTailSelectedFailure(t *testing.T) {
	for _, fault := range []string{"tail_matmul", "readback", "publication-partial", "aliased-output", "record-input", "record-prior"} {
		t.Run(fault, func(t *testing.T) {
			m, s, b := v41GroupedPanelFixture(t)
			defer s.Close()
			rows := 63
			var cause error = errors.New("selected tail failure")
			switch fault {
			case "tail_matmul":
				rows = 127
				cause = &compute.BackendError{Backend: "cpu-recorder", Site: "BatchedMatMul", Err: cause}
				b.matmulFaultAt, b.fault = m.Cfg.OGroups+2, cause
			case "readback":
				b.deferredReadErr = cause
			default:
				b.malformed, b.fault = fault, cause
			}
			input := v41GroupedPanelInput(rows, m.Cfg.NumHeads*m.Cfg.HeadDim)
			got, outcome, err := v41GroupedPanelCacheCall(s, input, rows)
			var selected *BackendForwardOperationError
			if got != nil || outcome != v41ProjectionError || !errors.As(err, &selected) || selected.Path != "v41-grouped-output-panel" || !s.BackendSessionClosed() {
				t.Fatalf("outcome=%d err=%v output=%v closed=%t", outcome, err, got, s.BackendSessionClosed())
			}
			if (fault == "tail_matmul" || fault == "readback" || fault == "publication-partial") && !errors.Is(err, cause) {
				t.Fatalf("lost selected cause: %v", err)
			}
			if fault == "tail_matmul" && (b.matmuls != m.Cfg.OGroups+2 || b.rowReads != 1 || b.gathers != 1) {
				t.Fatalf("selected tail replayed: matmuls/reads/gathers=%d/%d/%d", b.matmuls, b.rowReads, b.gathers)
			}
			if fault == "readback" && (b.matmuls != m.Cfg.OGroups+1 || b.rowReads != 1 || b.gathers != 1) {
				t.Fatalf("selected readback replayed: matmuls/reads/gathers=%d/%d/%d", b.matmuls, b.rowReads, b.gathers)
			}
			if len(b.deferredPending) != 0 {
				t.Fatalf("pending status records=%d", len(b.deferredPending))
			}
			// Selected failure clears the session memo; the model still owns
			// immutable weights independently of the retired backend session.
			weights := map[compute.Buffer]bool{}
			resident := m.immutableDeviceWeights(b)
			resident.mu.Lock()
			for _, w := range resident.weights {
				weights[w.Buf()] = true
			}
			resident.mu.Unlock()
			if len(weights) != m.Cfg.OGroups+1 {
				t.Fatalf("cached weights=%d want=%d", len(weights), m.Cfg.OGroups+1)
			}
			for buf, created := range b.created {
				wantFreed := 1
				if weights[buf] {
					wantFreed = 0
				}
				if created != 1 || b.freed[buf] != wantFreed {
					t.Fatalf("ownership: created=%d freed=%d want=%d", created, b.freed[buf], wantFreed)
				}
			}
			for buf := range b.freed {
				if b.created[buf] != 1 {
					t.Fatal("freed unowned buffer")
				}
			}
		})
	}
}
