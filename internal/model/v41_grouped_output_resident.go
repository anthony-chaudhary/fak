package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41GroupedResidentOps struct {
	rows   compute.V41RowLogBackend
	status compute.V41DeferredStatusBackend
}

// Capability admission is read-only. In particular the existing deferred-BF16
// disable switch still chooses the synchronous path before any chain mutation.
func v41GroupedResidentCapabilities(backend compute.Backend, rank, dim int) (v41GroupedResidentOps, bool) {
	rows, ok := backend.(compute.V41RowLogBackend)
	if !ok || !rows.V41RowLogAvailable() {
		return v41GroupedResidentOps{}, false
	}
	publication, ok := backend.(compute.V41BF16PublishBackend)
	if !ok || !publication.SupportsV41BF16Publish(rank) || !publication.SupportsV41BF16Publish(dim) {
		return v41GroupedResidentOps{}, false
	}
	status, ok := v41DeferredBF16Publisher(backend, rank)
	if !ok || !status.SupportsV41DeferredStatus(dim) {
		return v41GroupedResidentOps{}, false
	}
	return v41GroupedResidentOps{rows: rows, status: status}, true
}

type v41GroupedResidentStats struct {
	calls            int
	upload, readback int64
}

// v41GroupedResident executes the selected full-profile wo_a -> concat -> wo_b
// chain. Cached weights are borrowed. All temporaries remain live until their
// last consumer and status fence; no intermediate activation returns to host.
// The first fence reads only wo_a statuses in group order. wo_b never executes
// after a failed group. The second fence validates wo_b and reads its one row.
// Later independent group operations may be recorded before an earlier group's
// arithmetic verdict is known; on a later failure, prior verdicts are drained
// first so an earlier arithmetic failure remains primary.
func (s *Session) v41GroupedResident(ops v41GroupedResidentOps, weights []func() compute.Tensor, outputWeight func() compute.Tensor, input []float32, rank, dim int, protected []compute.Tensor, stage *string, stats *v41GroupedResidentStats) (result []float32, err error) {
	var owned []compute.Tensor
	borrowed := append([]compute.Tensor(nil), protected...)
	var statuses []compute.V41DeferredStatus
	own := func(t compute.Tensor) bool {
		if t.Buf() == nil || t.Backend() != s.Backend {
			return false
		}
		for _, prior := range owned {
			if prior.Buf() == t.Buf() {
				return false
			}
		}
		for _, status := range statuses {
			if status.Record.Buf() == t.Buf() {
				return false
			}
		}
		for _, w := range borrowed {
			if w.Buf() == t.Buf() {
				return false
			}
		}
		owned = append(owned, t)
		return true
	}
	read := func(rows []compute.Tensor) ([][]float32, error) {
		pending := statuses
		statuses = nil // V41ReadDeferred consumes records even on transport error.
		verdicts, values, cause := ops.status.V41ReadDeferred(pending, rows)
		if cause != nil {
			return nil, cause
		}
		if len(verdicts) != len(pending) || len(values) != len(rows) {
			return nil, errV41BF16BoundaryResult
		}
		for i, verdict := range verdicts {
			if verdict != nil {
				*stage = "BF16 publication"
				return nil, fmt.Errorf("grouped publication %d: %w", i, verdict)
			}
		}
		return values, nil
	}
	defer func() {
		p := recover()
		defer func() {
			for i := len(owned) - 1; i >= 0; i-- {
				s.Backend.Free(owned[i])
			}
		}()
		if len(statuses) > 0 {
			_, earlier := read(nil)
			if earlier != nil {
				err = errors.Join(earlier, err)
				if panicErr, ok := p.(error); ok {
					err = errors.Join(err, panicErr)
					p = nil
				}
			}
		}
		if err != nil {
			result = nil
		}
		if p != nil {
			panic(p)
		}
	}()
	valid := func(t compute.Tensor, n int) bool {
		return t.Backend() == s.Backend && t.Buf() != nil && t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil && t.Numel() == n
	}
	publish := func(x compute.Tensor, n int) (compute.Tensor, error) {
		*stage = "BF16 publication"
		x.Shape = []int{n}
		p, status, cause := ops.status.V41BF16PublishDeferred(x, compute.V41BF16PublishGroupedOutputProjection)
		fresh := own(p)
		// A malformed or aliased record cannot be passed to the consuming
		// reader. Claim only distinct storage owned by this backend for cleanup.
		recordFresh := own(status.Record)
		recordValid := recordFresh && status.Kind == compute.V41DeferredBF16Publish && status.Count == n && status.Site == compute.V41BF16PublishGroupedOutputProjection && status.Records <= 1 && valid(status.Record, 4)
		if recordValid {
			owned = owned[:len(owned)-1] // status ownership transfers to read().
			statuses = append(statuses, status)
		}
		if cause != nil {
			return compute.Tensor{}, cause
		}
		if !fresh || !valid(p, n) || !recordValid {
			return compute.Tensor{}, errV41BF16BoundaryResult
		}
		return p, nil
	}
	if len(weights) == 0 || rank <= 0 || dim <= 0 || len(input) == 0 || len(input)%len(weights) != 0 {
		return nil, errV41ProjectionResult
	}
	groupIn := len(input) / len(weights)
	spans := make([]compute.V41RowSpan, len(weights))
	for g, resolve := range weights {
		*stage = "weight upload"
		weight := resolve()
		borrowed = append(borrowed, weight)
		*stage = "activation upload"
		x := s.uploadHostF32([]int{groupIn}, input[g*groupIn:(g+1)*groupIn], compute.MemoryActivation, "V4.1 grouped resident activation")
		if !own(x) || !valid(x, groupIn) {
			return nil, errV41ProjectionResult
		}
		stats.upload += int64(groupIn) * 4
		*stage = "matmul"
		stats.calls++
		y := s.Backend.MatMul(weight, x)
		if !own(y) || !valid(y, rank) {
			return nil, errV41ProjectionResult
		}
		p, cause := publish(y, rank)
		if cause != nil {
			return nil, cause
		}
		p.Shape = []int{1, rank}
		spans[g] = compute.V41RowSpan{Src: p, Rows: 1}
	}
	*stage = "BF16 publication readback"
	if _, err = read(nil); err != nil {
		return nil, err
	}
	*stage = "weight upload"
	output := outputWeight()
	borrowed = append(borrowed, output)
	*stage = "device concat"
	joined, cause := ops.rows.V41RowLogGather(spans, rank)
	fresh := own(joined)
	if cause != nil {
		return nil, cause
	}
	if !fresh || !valid(joined, len(weights)*rank) {
		return nil, errV41ProjectionResult
	}
	joined.Shape = []int{len(weights) * rank}
	*stage = "matmul"
	stats.calls++
	y := s.Backend.MatMul(output, joined)
	if !own(y) || !valid(y, dim) {
		return nil, errV41ProjectionResult
	}
	p, cause := publish(y, dim)
	if cause != nil {
		return nil, cause
	}
	*stage = "readback"
	values, cause := read([]compute.Tensor{p})
	if cause != nil {
		return nil, cause
	}
	stats.readback += int64(len(values[0])) * 4
	if len(values[0]) != dim || !v41BF16Widened(values[0]) {
		return nil, errV41ProjectionResult
	}
	return append([]float32(nil), values[0]...), nil
}
