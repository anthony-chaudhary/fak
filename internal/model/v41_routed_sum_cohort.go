package model

import (
	"errors"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Experimental cold-prefill readback grouping. Scalar remains the default;
// fewer API reads are not evidence of a hardware throughput improvement.
const v41SumReadbackRows = 8

var v41PrefillSumReadback atomic.Bool

// SetV41PrefillSumReadback configures the opt-in, eight-row cold-prefill sum
// readback experiment. Configure before serving requests. False (the default)
// preserves scalar reads; model/backend eligibility still gates true.
func SetV41PrefillSumReadback(enabled bool) { v41PrefillSumReadback.Store(enabled) }

// A recovered legacy panic(nil) still fails; Goexit never returns here.
func v41CaptureSumReadback(run func()) *v41RoutedOutputCleanup {
	complete := false
	var value any
	func() {
		defer func() { value = recover() }()
		run()
		complete = true
	}()
	if !complete {
		return &v41RoutedOutputCleanup{value: value, present: true}
	}
	return nil
}

func (o *v41RoutedOutputOwner) sumReadbackEnabled() bool {
	if o == nil || !v41PrefillSumReadback.Load() || !o.live() {
		return false
	}
	b, ok := o.backend.(compute.V41DeferredStatusBackend)
	return ok && b.SupportsV41DeferredStatus(o.width)
}

func (o *v41RoutedOutputOwner) sumReadbackGroups(picks [][]routePick, outputs [][][]float32, out [][]float32) error {
	for first := 0; first < len(picks); first += v41SumReadbackRows {
		last := min(first+v41SumReadbackRows, len(picks))
		rows, err := o.sumReadbackCohort(first, picks[first:last], outputs[first:last])
		if err != nil {
			return err
		}
		copy(out[first:last], rows)
	}
	return nil
}

// Keep every accumulator and uploaded/retained operand live through the one
// read. A later recording failure first resolves the earlier complete rows.
// Temporary and retained cleanup is ordered by token; it cannot hide an earlier
// result failure. No selected failure retries arithmetic or falls back to host.
func (o *v41RoutedOutputOwner) sumReadbackCohort(first int, picks [][]routePick, outputs [][][]float32) (result [][]float32, cause error) {
	s, width := o.session, o.width
	invalid := v41ExpertOperationErr(o.layer, "sum", errV41ExpertResult)
	complete := false
	defer func() {
		if !complete {
			o.release(0, true)
			cause = o.retire(errors.New("model: incomplete V4.1 routed sum readback"))
			result = nil
		}
	}()
	result, cause = v41SelectedDeviceCall(s, o.layer, "v41-routed-sum", func(stage *string) (result [][]float32, cause error) {
		count := len(picks)
		if !o.live() || count < 1 || count > v41SumReadbackRows || len(outputs) != count || first < 0 || first > len(o.picks)-count {
			return nil, invalid
		}
		owned := make([][]compute.Tensor, count)
		accs := make([]compute.Tensor, 0, count)
		faultRow := count
		faultStage := "payload"
		var primary *v41RoutedOutputCleanup
		failed := func(row int, err error, p *v41RoutedOutputCleanup) {
			if row < faultRow {
				faultRow, cause, primary = row, err, p
				faultStage = *stage
			}
		}
		// Schedule all row cleanups before any backend Free. Even Goexit runs
		// every scheduled release; the outer incomplete guard retires afterward.
		defer func() {
			*stage = "cleanup"
			func() {
				for row := count - 1; row >= 0; row-- {
					defer func(row int) {
						defer func() {
							cleanup := o.release(first+row, false)
							if cleanup.present {
								failed(row, nil, &cleanup)
							}
						}()
						for _, tensor := range owned[row] {
							defer func(t compute.Tensor) {
								freed := false
								defer func() {
									p := recover()
									if !freed {
										failed(row, nil, &v41RoutedOutputCleanup{value: p, present: true})
									}
								}()
								o.backend.Free(t)
								freed = true
							}(tensor)
						}
					}(row)
				}
			}()
			if faultRow < count {
				*stage = faultStage
				result = nil
				o.release(0, true)
				if primary != nil {
					if known, ok := v41RoutedCohortPanicCause(primary.value); ok {
						var forward *BackendForwardOperationError
						if errors.As(known, &forward) {
							panic(primary.value)
						}
						cause = known
					} else {
						o.retire(v41RoutedCohortUnknownPanic(primary.value))
						panic(primary.value)
					}
				}
			}
		}()
		for row := range picks {
			p := v41CaptureSumReadback(func() {
				*stage = "payload"
				if len(picks[row]) != len(outputs[row]) {
					failed(row, invalid, nil)
					return
				}
				for _, values := range outputs[row] {
					if len(values) != width {
						failed(row, invalid, nil)
						return
					}
					for _, v := range values {
						if !finite32(v) {
							failed(row, invalid, nil)
							return
						}
					}
				}
				*stage = "accumulator upload"
				acc := s.uploadHostF32([]int{width}, make([]float32, width), compute.MemoryActivation, "V4.1 routed-sum accumulator")
				owned[row] = append(owned[row], acc)
				tensors := make([]compute.Tensor, len(outputs[row]))
				*stage = "expert upload"
				for slot, values := range outputs[row] {
					if index, found := o.index[v41ExpertRow{first + row, slot}]; found {
						tensors[slot] = o.rows[index].tensor
					} else {
						tensors[slot] = s.uploadHostF32([]int{width}, values, compute.MemoryActivation, "V4.1 routed-sum expert output")
						owned[row] = append(owned[row], tensors[slot])
					}
				}
				*stage = "routed-sum"
				v41DeviceAccumulate(o.backend, acc, tensors, v41RoutedSumOrder(picks[row]))
				accs = append(accs, acc)
			})
			if p != nil {
				failed(row, nil, p)
			}
			if faultRow < count {
				break
			}
		}
		if len(accs) > 0 {
			*stage = "readback"
			var verdicts []error
			var values [][]float32
			var err error
			p := v41CaptureSumReadback(func() {
				verdicts, values, err = o.backend.(compute.V41DeferredStatusBackend).V41ReadDeferred(nil, accs)
			})
			if p != nil {
				failed(0, nil, p)
				return nil, cause
			}
			if err != nil {
				failed(0, err, nil)
			} else if len(verdicts) != 0 || len(values) != len(accs) {
				failed(0, invalid, nil)
			} else {
				result = make([][]float32, len(values))
				for row, values := range values {
					if len(values) != width {
						failed(row, invalid, nil)
						break
					}
					for _, v := range values {
						if !finite32(v) {
							failed(row, invalid, nil)
							break
						}
					}
					if faultRow <= row {
						break
					}
					result[row] = append([]float32(nil), values...)
				}
			}
		}
		return result, cause
	})
	complete = true
	return result, cause
}
