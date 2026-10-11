package compute

import (
	"fmt"
	"math"
)

// V41EngramGateBackend is the optional DeepSeek V4.1 Engram gate/add over the
// device-resident mHC stream panel (Ref #13668, #13811). For each row r and
// hyper-connection stream s it transcribes the host Model.v41EngramInjectPrepared
// full-geometry arithmetic:
//
//	key = BF16(projected[r, s*H:(s+1)*H]), value = BF16(projected[r, 4H:5H])
//	dot = serial sum h[i]*(qNorm[sH+i]*kNorm[sH+i])*key[i]
//	dot *= rsqrt(sum(h^2)/H + eps) * rsqrt(sum(key^2)/H + eps) * rsqrt(H)
//	gate = 1/(1 + exp(-copysign(sqrt(max(|dot|, 1e-6)), dot)))
//	next[r, s, i] = BF16(h[i] + gate*value[i])
//
// with h the stream's own row. The three sums are serial binary32 sums of
// separately rounded products, and the reciprocal and square roots are the
// correctly rounded host values, so they are bit-exact. exp is a Cody-Waite
// binary32 evaluation within about one ulp of the host's F64-narrowed exp, so a
// published element can differ from the host by one BF16 ulp when the F32 sum
// lands next to a BF16 rounding boundary.
//
// streams is F32 [rows,4,hidden], projected the raw F32 Engram projection
// [rows,5*hidden] and norms F32 [8*hidden] (qNorm then kNorm, each [4,hidden]).
// next is a fresh [rows,4,hidden] panel; status is a fresh F32 [rows,4] whose
// element is nonzero when that stream's BF16 key or the row's BF16 value holds a
// non-finite element, the host projection's publication refusal. The caller
// must read status and refuse the step before publishing next. Absence permits
// host selection; after dispatch is selected, an error never permits host replay.
type V41EngramGateBackend interface {
	SupportsV41EngramGate() bool
	V41EngramGate(streams, projected, norms Tensor, rows, hidden int, eps float32) (next, status Tensor, err error)
}

// V41EngramGateMaxHidden bounds the serial per-(row, stream) reduction.
const V41EngramGateMaxHidden = 1 << 16

// V41EngramGateStreams is the fixed hyper-connection stream count the gate
// reads from the resident panel; the projection carries one extra value stream.
const V41EngramGateStreams = V41MHCStreams

// V41EngramInvSqrt is the host reciprocal square root float32(1/sqrt(float64(x))).
func V41EngramInvSqrt(x float32) float32 { return float32(1 / math.Sqrt(float64(x))) }

func validateV41EngramGate(streams, projected, norms Tensor, rows, hidden int, eps float32) error {
	if rows <= 0 || hidden <= 0 || hidden > V41EngramGateMaxHidden {
		return fmt.Errorf("compute: V4.1 Engram gate rows=%d hidden=%d outside [1,%d]", rows, hidden, V41EngramGateMaxHidden)
	}
	if math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return fmt.Errorf("compute: V4.1 Engram gate epsilon is non-finite")
	}
	panel := uint64(rows) * V41EngramGateStreams * uint64(hidden)
	proj := uint64(rows) * (V41EngramGateStreams + 1) * uint64(hidden)
	if proj > math.MaxInt32 {
		return fmt.Errorf("compute: V4.1 Engram gate rows=%d hidden=%d exceeds the shader index range", rows, hidden)
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		elems uint64
	}{
		{"streams", streams, panel},
		{"projected", projected, proj},
		{"norms", norms, 2 * V41EngramGateStreams * uint64(hidden)},
	} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || uint64(operand.t.Numel()) != operand.elems {
			return fmt.Errorf("compute: V4.1 Engram gate %s must be an unquantized row-major F32 tensor of %d elements", operand.name, operand.elems)
		}
	}
	return nil
}

func vulkanV41EngramGateStatusError(status int, reason string) *BackendError {
	err := vulkanV41IndexerScoreStatusError(status, reason)
	err.Site = "V41EngramGate"
	err.Message = fmt.Sprintf("dedicated Engram gate failed closed (code %d): %s", status, reason)
	return err
}
