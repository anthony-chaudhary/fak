package compute

import "fmt"

// cudaSequenceSplitScratchPlan bounds only compensated FP16 split halves.
// The two maxima may come from different projections; this is not a measured
// peak-VRAM bound and excludes ordinary activations, KV and other GEMM arms.
type cudaSequenceSplitScratchPlan struct {
	weightHalfBytes     int
	activationHalfBytes int
}

func cudaSequenceSplitScratchBytes(weightElements, activationElements uint64) (cudaSequenceSplitScratchPlan, error) {
	maxInt := uint64(^uint(0) >> 1)
	if weightElements == 0 || activationElements == 0 ||
		weightElements > maxInt/2 || activationElements > maxInt/2 {
		return cudaSequenceSplitScratchPlan{}, fmt.Errorf("compensated split half byte size overflows")
	}
	w, x := weightElements*2, activationElements*2
	if w > maxInt/2 || x > (maxInt-2*w)/2 {
		return cudaSequenceSplitScratchPlan{}, fmt.Errorf("compensated split owner total byte size overflows")
	}
	return cudaSequenceSplitScratchPlan{weightHalfBytes: int(w), activationHalfBytes: int(x)}, nil
}

// planCUDASequenceSplitScratch mirrors only the existing compensated dispatch
// predicates. Its caller holds cudaMu from the arm snapshot through scope exit.
func planCUDASequenceSplitScratch(req Qwen35SequencePrefillRequest, arm CUDAQ4KPrefillArm, armErr error) (cudaSequenceSplitScratchPlan, error) {
	var plan cudaSequenceSplitScratchPlan
	tokens := len(req.TokenIDs)
	if tokens < 128 {
		return plan, nil
	}
	var maxW, maxX uint64
	add := func(weight Tensor) error {
		if len(weight.Shape) != 2 || weight.Shape[0] <= 0 || weight.Shape[1] <= 0 {
			return fmt.Errorf("compensated projection needs positive [out,in] dimensions")
		}
		out, in := uint64(weight.Shape[0]), uint64(weight.Shape[1])
		const maxCUDAInt = uint64(1<<31 - 1)
		if out > maxCUDAInt || in > maxCUDAInt || uint64(tokens) > maxCUDAInt ||
			out > maxCUDAInt/in || uint64(tokens) > maxCUDAInt/in {
			return fmt.Errorf("compensated projection element count exceeds CUDA int contract")
		}
		w, x := out*in, uint64(tokens)*in
		if w > maxW {
			maxW = w
		}
		if x > maxX {
			maxX = x
		}
		return nil
	}
	generic := func(weight Tensor) error {
		switch weight.Dtype {
		case Q8_0, Q2_K:
			return add(weight)
		case Q4_K:
			// An invalid Q4 selection is irrelevant unless this request will
			// actually reach a large-P Q4 dispatch.
			if armErr != nil {
				return armErr
			}
			switch arm {
			case CUDAQ4KPrefillCompensated:
				return add(weight)
			case CUDAQ4KPrefillMMQ, CUDAQ4KPrefillFP8LT:
				return nil
			default:
				return fmt.Errorf("unsupported CUDA Q4_K prefill arm %q", arm)
			}
		}
		return nil
	}
	for index, layer := range req.Layers {
		var err error
		for _, weight := range []Tensor{layer.Gate, layer.Up, layer.Down} {
			if err = generic(weight); err != nil {
				break
			}
		}
		if err == nil && layer.Linear {
			quartet := []Tensor{layer.GDNInQKV, layer.GDNInZ, layer.GDNInB, layer.GDNInA}
			allQ8 := true
			for _, weight := range quartet {
				allQ8 = allQ8 && weight.Dtype == Q8_0
			}
			if allQ8 {
				for _, weight := range quartet {
					if err = add(weight); err != nil {
						break
					}
				}
			}
			if err == nil && layer.GDNOut.Dtype == Q8_0 {
				err = add(layer.GDNOut)
			}
		} else if err == nil {
			for _, weight := range []Tensor{layer.Q, layer.K, layer.V, layer.O} {
				if err = generic(weight); err != nil {
					break
				}
			}
		}
		if err != nil {
			return plan, &Qwen35SequenceError{Stage: "split-scratch-plan", Layer: index, Cause: err}
		}
	}
	// Output heads stay outside this layer loan: the last-row head uses P=1,
	// and an all-row head runs only after the loan's checked retirement.
	if maxW == 0 {
		return plan, nil
	}
	return cudaSequenceSplitScratchBytes(maxW, maxX)
}
