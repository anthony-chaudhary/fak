package compute

import (
	"fmt"
	"slices"
)

// V41RouterTopKMaxK bounds the per-row selection held in shader registers.
const V41RouterTopKMaxK = 8

// V41RouterTopKBackend is the optional DeepSeek-V4.1 MoE router primitive.
// Inputs are immutable, live, unquantized row-major F32 tensors owned by this
// backend: logits[rows,experts], and bias[experts] when hasBias is true;
// otherwise bias is a live [1] dummy the kernel ignores.
//
// Per row, score = sqrt(softplus(z)) per expert, selection is the topK largest
// score+bias in descending order with the lower expert first on ties, and each
// weight is the unbiased selected score divided by the in-order F32 sum
// (sum+1e-20 when full) times routeScale. Validation follows the host router
// order: per expert ascending, logit, score, bias, then selection score; then
// the selected sum (non-finite, negative, or zero when not full), then weights.
//
// Only the topK expert indices and weights per row, plus a four-word status
// record per row, leave the device: the host needs the picks to schedule
// expert paging, and the logits stay resident. Scores use the device F32
// exp/log/sqrt, so weights carry a tolerance against the host F64-backed
// formula, and a selection whose k-th and (k+1)-th choices differ by less than
// that error may order differently. A *V41RouterTopKFault is a numeric verdict
// on the inputs; any other error is a backend failure. Either returns no picks
// and must never replay on the host.
type V41RouterTopKBackend interface {
	Backend
	V41RouterTopKAvailable() bool
	V41RouterTopK(logits, bias Tensor, rows, experts, topK int, routeScale float32, full, hasBias bool) (V41RouterPicks, error)
}

// V41RouterPicks is the row-major [rows,topK] host readback of a device route.
type V41RouterPicks struct {
	Experts []int32
	Weights []float32
}

// V41RouterTopKStage is the producer stage recorded in the device status.
// Keep its values synchronized with v41_router_topk.comp and the native ABI.
type V41RouterTopKStage uint32

const (
	V41RouterTopKStageSuccess       V41RouterTopKStage = 0
	V41RouterTopKStageLogit         V41RouterTopKStage = 1
	V41RouterTopKStageScore         V41RouterTopKStage = 2
	V41RouterTopKStageBias          V41RouterTopKStage = 3
	V41RouterTopKStageSelection     V41RouterTopKStage = 4
	V41RouterTopKStageNormalization V41RouterTopKStage = 5
	V41RouterTopKStageWeight        V41RouterTopKStage = 6
)

func (s V41RouterTopKStage) String() string {
	switch s {
	case V41RouterTopKStageSuccess:
		return "success"
	case V41RouterTopKStageLogit:
		return "logit"
	case V41RouterTopKStageScore:
		return "score"
	case V41RouterTopKStageBias:
		return "correction bias"
	case V41RouterTopKStageSelection:
		return "selection score"
	case V41RouterTopKStageNormalization:
		return "normalization"
	case V41RouterTopKStageWeight:
		return "weight"
	default:
		return fmt.Sprintf("unknown stage %d", uint32(s))
	}
}

// V41RouterTopKFault is one row's first device-reported failure. Expert is -1
// for the row-level normalization stage.
type V41RouterTopKFault struct {
	Row    int
	Stage  V41RouterTopKStage
	Expert int
	Bits   uint32
}

func (f *V41RouterTopKFault) Error() string {
	return fmt.Sprintf("compute: V4.1 router row %d failed at %s (expert %d, bits %#08x)", f.Row, f.Stage, f.Expert, f.Bits)
}

// validateV41RouterTopK rejects malformed geometry before narrowing it for the
// 32-bit shader ABI. It reads no tensor data and allocates nothing.
func validateV41RouterTopK(logits, bias Tensor, rows, experts, topK int, routeScale float32, hasBias bool) (logitBytes, biasBytes, pickBytes, statusBytes int, err error) {
	const maxABIInt = 1<<31 - 1
	if rows <= 0 || experts <= 0 || topK <= 0 || topK > experts || topK > V41RouterTopKMaxK ||
		rows > maxABIInt || experts > maxABIInt || uint64(rows)*uint64(experts) > maxABIInt {
		return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 router invalid geometry rows=%d experts=%d topK=%d", rows, experts, topK)
	}
	if !(routeScale > 0) || routeScale > 3.4028234663852886e38 {
		return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 router scale must be finite and positive, got %g", routeScale)
	}
	biasShape := []int{1}
	if hasBias {
		biasShape = []int{experts}
	}
	for _, operand := range []struct {
		name string
		t    Tensor
		want []int
	}{
		{"logits", logits, []int{rows, experts}},
		{"bias", bias, biasShape},
	} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || !slices.Equal(operand.t.Shape, operand.want) {
			return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 router %s must be unquantized row-major F32 with shape %v", operand.name, operand.want)
		}
	}
	return rows * experts * 4, biasShape[0] * 4, rows * topK * 4, rows * 16, nil
}

// decodeV41RouterTopK checks every row's status record and the picked
// experts, then returns owned picks. words holds rows*4 status words.
func decodeV41RouterTopK(status []uint32, experts []uint32, weights []float32, rows, numExperts, topK int) (V41RouterPicks, error) {
	if len(status) != rows*4 || len(experts) != rows*topK || len(weights) != rows*topK {
		return V41RouterPicks{}, fmt.Errorf("compute: V4.1 router readback has wrong extent")
	}
	for row := 0; row < rows; row++ {
		record := status[row*4 : row*4+4]
		if record[0] == 0 {
			if record[1] != 0 || record[2] != 0 || record[3] != 0 {
				return V41RouterPicks{}, fmt.Errorf("compute: V4.1 router row %d has a malformed success record", row)
			}
			continue
		}
		stage := V41RouterTopKStage(record[0])
		if stage > V41RouterTopKStageWeight || record[1] > uint32(numExperts) || record[3] != 0 {
			return V41RouterPicks{}, fmt.Errorf("compute: V4.1 router row %d has a malformed fault record", row)
		}
		return V41RouterPicks{}, &V41RouterTopKFault{Row: row, Stage: stage, Expert: int(record[1]) - 1, Bits: record[2]}
	}
	out := V41RouterPicks{Experts: make([]int32, len(experts)), Weights: append([]float32(nil), weights...)}
	for row := 0; row < rows; row++ {
		seen := make(map[uint32]bool, topK)
		for slot := 0; slot < topK; slot++ {
			e := experts[row*topK+slot]
			if e >= uint32(numExperts) || seen[e] {
				return V41RouterPicks{}, fmt.Errorf("compute: V4.1 router row %d slot %d picked invalid expert %d", row, slot, e)
			}
			seen[e] = true
			out.Experts[row*topK+slot] = int32(e)
		}
	}
	return out, nil
}
