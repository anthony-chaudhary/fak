package compute

import (
	"errors"
	"fmt"
)

// ErrUnsupportedLayout identifies storage that an operation cannot interpret.
var ErrUnsupportedLayout = errors.New("compute: unsupported tensor layout")

func checkRowMajor(op string, tensors ...Tensor) error {
	for i := range tensors {
		if tensors[i].Layout != RowMajor {
			return fmt.Errorf("%w: %s operand %d has layout %d, want row-major", ErrUnsupportedLayout, op, i, tensors[i].Layout)
		}
	}
	return nil
}

func requireRowMajor(op string, tensors ...Tensor) {
	if err := checkRowMajor(op, tensors...); err != nil {
		panic(err)
	}
}

func checkCUDAUploadLayout(op string, t Tensor, as Dtype, class MemoryClass) error {
	// This legacy source flag requests CUDA's explicit load-time F16 transpose.
	if t.Layout == ColMajor && t.Dtype == F32 && as == F16 && class == MemoryWeights && len(t.Shape) == 2 {
		return nil
	}
	return checkRowMajor(op, t)
}

func requireCUDAUploadLayout(op string, t Tensor, as Dtype, class MemoryClass, site string) {
	if err := checkCUDAUploadLayout(op, t, as, class); err != nil {
		panic(fmt.Errorf("%w (site %q)", err, site))
	}
}
