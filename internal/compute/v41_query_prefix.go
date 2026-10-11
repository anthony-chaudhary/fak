package compute

import (
	"errors"
	"fmt"
	"math"
)

// V41QueryPrefixBackend is a warm-weight, single-row experimental composition.
// It preserves the current generic RMSNorm/F32 matmul kernels and all four
// checked BF16 boundaries. One owner lock covers the initial copy, command,
// statuses, both output reads and cleanup. Ordinary Vulkan does not expose it.
// Availability describes modules only; a busy batch is never capability absence.
type V41QueryPrefixBackend interface {
	V41QueryPrefixAvailable() bool
	V41QueryPrefix(V41QueryPrefixInputs) (V41QueryPrefixResult, error)
}

// V41QueryPrefixInputs borrows canonical immutable resident weights and their
// two corresponding host gain views. No weight is staged by this operation.
// Collapse and both host gains remain immutable for the synchronous call.
type V41QueryPrefixInputs struct {
	Collapse          []float32
	AttentionGainHost []float32
	QueryGainHost     []float32
	AttentionGain     Tensor
	WQA               Tensor
	QueryGain         Tensor
	Epsilon           float32
}

// V41QueryPrefixResult owns two disjoint host rows. Every error withholds both.
// Each result is finite widened BF16, including its signed zero bits.
type V41QueryPrefixResult struct {
	Collapsed []float32
	QLat      []float32
}

// V41QueryPrefixBusyError is a healthy pre-selection refusal. It causes neither
// replay nor owner retirement: retry is possible once this batch/restore ends.
// Sticky or uncertain submissions are fatal errors, never busy errors.
type V41QueryPrefixBusyError struct{ State string }

func (e *V41QueryPrefixBusyError) Error() string {
	return "compute: V4.1 query prefix cannot select while " + e.State + " is active"
}

// V41QueryPrefixStageError preserves operation attribution and its primary cause.
type V41QueryPrefixStageError struct {
	Stage string
	Cause error
}

func (e *V41QueryPrefixStageError) Error() string {
	return fmt.Sprintf("compute: V4.1 query prefix %s: %v", e.Stage, e.Cause)
}
func (e *V41QueryPrefixStageError) Unwrap() error { return e.Cause }

const (
	V41BF16CheckedAttentionNormInput  uint32 = 3
	V41BF16CheckedAttentionNormOutput uint32 = 4
	V41BF16CheckedQueryNormInput      uint32 = 5
	V41BF16CheckedQueryNormOutput     uint32 = 6
)

func validateV41QueryPrefix(in V41QueryPrefixInputs) (h, r int, err error) {
	fail := func(stage, message string) (int, int, error) {
		return 0, 0, &V41QueryPrefixStageError{Stage: stage, Cause: errors.New(message)}
	}
	if in.WQA.Dtype != F32 || in.WQA.Layout != RowMajor || in.WQA.Quant != nil || len(in.WQA.Shape) != 2 {
		return fail("payload", "wq_a must be row-major unquantized F32 [R,H]")
	}
	r, h = in.WQA.Shape[0], in.WQA.Shape[1]
	if h <= 0 || r <= 0 || h > math.MaxInt32 || r > math.MaxInt32 ||
		h > math.MaxInt/4 || r > math.MaxInt/4 || uint64(h)*uint64(r) > math.MaxInt32 ||
		uint64(h)*uint64(r) > uint64(math.MaxInt/4) {
		return fail("payload", "query prefix extent exceeds shader or host byte range")
	}
	if len(in.Collapse) != h || len(in.AttentionGainHost) != h || len(in.QueryGainHost) != r ||
		math.IsNaN(float64(in.Epsilon)) || math.IsInf(float64(in.Epsilon), 0) || in.Epsilon <= 0 {
		return fail("payload", "query prefix host geometry or epsilon is invalid")
	}
	for _, operand := range []struct {
		name string
		t    Tensor
		n    int
		host []float32
	}{{"attention gain", in.AttentionGain, h, in.AttentionGainHost}, {"query gain", in.QueryGain, r, in.QueryGainHost}} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil ||
			len(operand.t.Shape) != 1 || operand.t.Shape[0] != operand.n {
			return fail("payload", operand.name+" must be row-major unquantized F32 of the declared width")
		}
		for _, value := range operand.host {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fail(operand.name, "nonfinite learned norm gain")
			}
		}
	}
	return h, r, nil
}

// decodeV41QueryPrefixStatus has its own 3..6 policy. The final-norm decoder
// remains a 1/2-only wrapper, including its existing matching-site3 refusal.
func decodeV41QueryPrefixStatus(raw []byte, count int, site uint32) error {
	if site < V41BF16CheckedAttentionNormInput || site > V41BF16CheckedQueryNormOutput {
		return &V41BF16CheckedProtocolError{Message: "invalid query-prefix publication site"}
	}
	return decodeV41BF16CheckedEnvelope(raw, count, site)
}

// decodeV41QueryPrefixStatuses is called only AFTER all four checked transport
// reads have succeeded. Any protocol error precedes numerical attribution;
// within either class the fixed logical site order wins, then shader atomicMin.
func decodeV41QueryPrefixStatuses(raw [4][]byte, h, r int) error {
	var numerical error
	counts := [4]int{h, h, r, r}
	for i := range raw {
		err := decodeV41QueryPrefixStatus(raw[i], counts[i], uint32(i)+V41BF16CheckedAttentionNormInput)
		if err == nil {
			continue
		}
		var arithmetic *V41BF16CheckedArithmeticError
		if !errors.As(err, &arithmetic) {
			return err
		}
		if numerical == nil {
			numerical = err
		}
	}
	return numerical
}

func validateV41QueryPrefixResult(out V41QueryPrefixResult, h, r int) error {
	if len(out.Collapsed) != h || len(out.QLat) != r {
		return &V41QueryPrefixStageError{Stage: "output", Cause: errors.New("invalid result geometry")}
	}
	for _, row := range [][]float32{out.Collapsed, out.QLat} {
		for _, value := range row {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Float32bits(value)&0xffff != 0 {
				return &V41QueryPrefixStageError{Stage: "output", Cause: errors.New("result is not finite widened BF16")}
			}
		}
	}
	return nil
}
