package model

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41GroupedReadbackRecorder struct {
	compute.V41DeferredStatusBackend
	values   [][]float32
	verdicts []error
	cause    error
	calls    int
}

func (b *v41GroupedReadbackRecorder) V41ReadDeferred([]compute.V41DeferredStatus, []compute.Tensor) ([]error, [][]float32, error) {
	b.calls++
	return b.verdicts, b.values, b.cause
}

// fak-test:runtime fast est=100ms lane=default
func TestV41GroupedOutputOwnedReadback(t *testing.T) {
	t.Run("owned-versus-defensive-rows", func(t *testing.T) {
		for _, count := range []int{64, 11, 8, 2, 1} {
			values := make([][]float32, count)
			for i := range values {
				values[i] = []float32{float32(i), float32(i + 1)}
			}
			owned, err := v41GroupedPanelHostRows(values, count, 2, true, true)
			if err != nil || &owned[0][0] != &values[0][0] {
				t.Fatal("owned rows were copied")
			}
			if got := testing.AllocsPerRun(20, func() {
				if _, err := v41GroupedPanelHostRows(values, count, 2, true, true); err != nil {
					panic(err)
				}
			}); got != 0 {
				t.Fatalf("owned handoff allocates: %v", got)
			}
			copied, err := v41GroupedPanelHostRows(values, count, 2, true, false)
			if err != nil || &copied[0][0] == &values[0][0] {
				t.Fatal("unowned rows escaped")
			}
			values[0][0] = 99
			if copied[0][0] != 0 {
				t.Fatal("defensive copy aliases source")
			}
			for _, row := range owned {
				if cap(row) != len(row) {
					t.Fatal("owned fixture has excess capacity")
				}
			}
		}
		if _, err := v41GroupedPanelHostRows([][]float32{{math.Float32frombits(0x3f800001)}}, 1, 1, true, true); err == nil {
			t.Fatal("non-BF16 result admitted")
		}
	})
	t.Run("read-order-and-no-replay", func(t *testing.T) {
		fault := errors.New("numeric")
		meta := []v41GroupedPanelStatus{{width: 1}}
		status := []compute.V41DeferredStatus{{}}
		output := []compute.Tensor{{}}
		legacy := &v41GroupedReadbackRecorder{values: [][]float32{{1, 2}}, verdicts: []error{nil}}
		rows, err := v41GroupedPanelReadRows(legacy, nil, status, meta, output, 2, 8, 1)
		if err != nil || len(rows) != 2 || legacy.calls != 1 || &rows[0][0] == &legacy.values[0][0] {
			t.Fatal("legacy read/copy changed")
		}
		for _, fail := range []string{"numeric", "transport", "count", "success"} {
			backing := &v41GroupedReadbackRecorder{values: [][]float32{{1}, {2}}}
			_, result, err := compute.ReadV41DeferredResult(backing, nil, output)
			if err != nil {
				t.Fatal(err)
			}
			calls := 0
			reader := func(_ []compute.V41DeferredStatus, _ compute.Tensor, rows, width int) ([]error, compute.V41DeferredReadResult, error) {
				calls++
				if rows != 2 || width != 1 {
					t.Fatal("logical layout changed")
				}
				switch fail {
				case "numeric":
					return []error{fault}, result, nil
				case "transport":
					return nil, result, fault
				case "count":
					return nil, result, nil
				default:
					return []error{nil}, result, nil
				}
			}
			rows, err := v41GroupedPanelReadRows(legacy, reader, status, meta, output, 2, 8, 1)
			if calls != 1 || legacy.calls != 1 {
				t.Fatal("read replayed")
			}
			if fail == "success" {
				if err != nil || len(rows) != 2 || &rows[0][0] == &backing.values[0][0] {
					t.Fatal("unsealed result failed defensive copy")
				}
				if remaining, _ := result.TakeRows(); remaining != nil {
					t.Fatal("result was not consumed")
				}
			} else {
				if err == nil || rows != nil {
					t.Fatal("read fault escaped")
				}
				if remaining, _ := result.TakeRows(); len(remaining) != 2 {
					t.Fatal("rows consumed before verdict/shape validation")
				}
			}
		}
		legacy.values = nil
		if rows, err := v41GroupedPanelReadRows(legacy, func([]compute.V41DeferredStatus, compute.Tensor, int, int) ([]error, compute.V41DeferredReadResult, error) {
			t.Fatal("status-only drain used packed reader")
			return nil, compute.V41DeferredReadResult{}, nil
		}, status, meta, nil, 2, 8, 1); err != nil || rows != nil || legacy.calls != 2 {
			t.Fatal("status-only drain changed")
		}
	})
}
