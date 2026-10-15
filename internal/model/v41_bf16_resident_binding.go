package model

import (
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"math"
)

// This immutable product carries ordinary behavior and separately typed native
// authority. No serving call site invokes the resident core yet. Wrapping an
// ordinary callback must construct a new product and cannot retain authority.
type v41BF16PublicationBinding struct {
	ordinary v41BF16BoundaryFunc
	core     func(int, string, []float32, *v41BF16ResidentInput) ([]float32, error)
	owner    *Session
	model    *Model
	backend  compute.Backend
}

func (b *v41BF16PublicationBinding) publicationFunc() v41BF16BoundaryFunc {
	if b == nil {
		return nil
	}
	return b.ordinary
}
func (b *v41BF16PublicationBinding) withPublication(f v41BF16BoundaryFunc) *v41BF16PublicationBinding {
	return &v41BF16PublicationBinding{ordinary: f}
}

var errV41BF16ResidentScope = errors.New("model: V4.1 resident BF16 projection scope changed")

// V41BF16GroupedReadError distinguishes a new grouped transport boundary from
// either of the old independent reads. It retains context and the exact cause;
// it does not claim equivalence to the timing of either historical fence.
type V41BF16GroupedReadError struct {
	Layer int
	Site  string
	Cause error
}

func (e *V41BF16GroupedReadError) Error() string {
	return fmt.Sprintf("model: V4.1 grouped projection/BF16 read layer %d site %s: %v", e.Layer, e.Site, e.Cause)
}
func (e *V41BF16GroupedReadError) Unwrap() error { return e.Cause }

// The projection owns this synchronous loan. rawCheck must validate the complete
// raw projection before a BF16 verdict is considered; its contextual selected
// failure stays authoritative. Neither raw nor published storage escapes on a
// fault. Rejection before admission leaves the loan with its projection caller;
// after admission this arm retires it before selected failure closes the session.
// valid pins the caller's immutable binding, epoch and execution scope.
type v41BF16ResidentInput struct {
	tensor   compute.Tensor
	loan     *v41ProjectionLoan
	valid    func() bool
	rawCheck func([]float32) error
}

func (r *v41BF16ResidentInput) admitted(s *Session, m *Model, b compute.Backend) bool {
	if r == nil || s == nil || s.M != m || s.Backend != b || m.lora != nil || m.prism != nil || r.loan == nil || r.loan.backend != b || r.valid == nil || !r.valid() || r.loan.valid == nil || !r.loan.valid() || r.rawCheck == nil {
		return false
	}
	x := r.tensor
	if x.Buf() == nil || x.Backend() != b || x.Dtype != compute.F32 || x.Layout != compute.RowMajor || x.Quant != nil || x.Buf() == r.loan.protected {
		return false
	}
	for _, owned := range r.loan.owned {
		if owned.Buf() == x.Buf() {
			return true
		}
	}
	return false
}

// The ordinary adapter never calls this arm. Native composition must select it
// only after confirming deferred capability; a selected failure never declines.
func v41RunResidentBF16(s *Session, m *Model, b compute.Backend, layer int, site string, siteID uint32, count int, r *v41BF16ResidentInput, stage *string) (result []float32, cause error) {
	defer func() {
		primary := recover()
		cleanup := r.loan.release()
		if primary != nil {
			panic(primary)
		}
		if cause == nil && cleanup != nil {
			panic(cleanup)
		}
	}()
	deferred, ok := v41DeferredBF16Publisher(b, count)
	if !ok || siteID == 0 || count == 0 || count > math.MaxInt/4 || !b.(compute.V41BF16PublishBackend).SupportsV41BF16Publish(count) {
		return nil, errV41BF16BoundaryResult
	}
	var owned []compute.Tensor
	defer func() {
		primary := recover()
		var cleanup any
		for i := len(owned) - 1; i >= 0; i-- {
			func() {
				defer func() {
					if v := recover(); v != nil && cleanup == nil {
						cleanup = v
					}
				}()
				b.Free(owned[i])
			}()
		}
		if primary != nil {
			panic(primary)
		}
		if cause == nil && cleanup != nil {
			panic(cleanup)
		}
	}()
	// Reject every projection-owned alias, not just the immediate input.
	fresh := func(t compute.Tensor) bool {
		if t.Buf() == nil || t.Backend() != b || t.Buf() == r.loan.protected {
			return false
		}
		for _, v := range r.loan.owned {
			if v.Buf() == t.Buf() {
				return false
			}
		}
		for _, v := range owned {
			if v.Buf() == t.Buf() {
				return false
			}
		}
		return true
	}
	input := r.tensor
	input.Shape = []int{count}
	*stage = "checked publication"
	out, status, err := deferred.V41BF16PublishDeferred(input, siteID)
	outputFresh := fresh(out)
	if outputFresh {
		owned = append(owned, out)
	}
	statusFresh := fresh(status.Record)
	if statusFresh {
		owned = append(owned, status.Record)
	}
	if err != nil {
		return nil, err
	}
	if !outputFresh || !statusFresh || out.Dtype != compute.F32 || out.Layout != compute.RowMajor || out.Quant != nil || len(out.Shape) != 1 || out.Shape[0] != count || status.Record.Dtype != compute.F32 || status.Record.Layout != compute.RowMajor || status.Record.Quant != nil || len(status.Record.Shape) != 1 || status.Record.Shape[0] != 4 || status.Kind != compute.V41DeferredBF16Publish || status.Count != count || status.Site != siteID || compute.V41DeferredRecords(status) != 1 {
		return nil, errV41BF16BoundaryResult
	}
	if !r.admitted(s, m, b) {
		return nil, errV41BF16ResidentScope
	}
	if v41HostBF16Witness.Load() {
		v41BF16ResidentCalls.Add(1)
		v41BF16DeferredCalls.Add(1)
	}
	// V41ReadDeferred consumes the status on success or failure. Transfer its
	// ownership before entering that boundary, so cleanup never retries it.
	owned = owned[:len(owned)-1]
	*stage = "grouped projection/publication readback"
	verdicts, rows, err := deferred.V41ReadDeferred([]compute.V41DeferredStatus{status}, []compute.Tensor{input, out})
	if err != nil {
		return nil, &V41BF16GroupedReadError{Layer: layer, Site: site, Cause: err}
	}
	if len(verdicts) != 1 || len(rows) != 2 || len(rows[0]) != count {
		return nil, errV41BF16BoundaryResult
	}
	if !r.admitted(s, m, b) {
		return nil, errV41BF16ResidentScope
	}
	// A later raw nonfinite beats an earlier finite BF16 overflow, matching the
	// original complete projection validation before publication.
	if err := r.rawCheck(rows[0]); err != nil {
		return nil, err
	}
	if !r.admitted(s, m, b) {
		return nil, errV41BF16ResidentScope
	}
	*stage = "checked publication"
	if verdicts[0] != nil {
		return nil, verdicts[0]
	}
	if len(rows[1]) != count || !v41BF16Widened(rows[1]) {
		return nil, errV41BF16BoundaryResult
	}
	return append([]float32(nil), rows[1]...), nil
}
