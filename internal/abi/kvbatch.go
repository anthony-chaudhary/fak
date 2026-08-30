package abi

import (
	"context"
	"fmt"
	"reflect"
)

// KVResidencyRequest identifies one ordered residency transfer.
type KVResidencyRequest struct {
	Digest    string
	From      int
	Positions int
}

// KVBatchStager is an optional ordered batch extension to KVBackend.
type KVBatchStager interface {
	StageSpans(context.Context, []KVResidencyRequest) []KVResidency
}

// KVBatchRestorer is an optional ordered batch extension to KVBackend.
type KVBatchRestorer interface {
	RestoreSpans(context.Context, []KVResidencyRequest) []KVResidency
}

// StageSpans stages requests in order, using a native batch extension when available
// and otherwise adapting the legacy per-span method serially. It always returns one
// validated receipt per request.
func StageSpans(ctx context.Context, backend KVBackend, requests []KVResidencyRequest) []KVResidency {
	if len(requests) == 0 {
		return []KVResidency{}
	}
	if nilContext(ctx) {
		return faultAll(requests, "abi: kv batch canceled: nil context")
	}
	if nilInterface(backend) {
		return faultAll(requests, "abi: kv batch nil backend")
	}
	if err := ctx.Err(); err != nil {
		return faultAll(requests, "abi: kv batch canceled: "+err.Error())
	}
	if batch, ok := backend.(KVBatchStager); ok && !nilInterface(batch) {
		got := batch.StageSpans(ctx, cloneRequests(requests))
		if err := ctx.Err(); err != nil {
			return faultAll(requests, "abi: kv batch canceled: "+err.Error())
		}
		return normalizeBatch(requests, got, false)
	}

	out := make([]KVResidency, len(requests))
	for i, req := range requests {
		if err := ctx.Err(); err != nil {
			fillFaults(out[i:], requests[i:], "abi: kv batch canceled: "+err.Error())
			break
		}
		res, err := backend.StageSpan(ctx, req.Digest, req.From, req.Positions)
		if err != nil {
			out[i] = fault(req, "abi: kv batch stage: "+err.Error())
			continue
		}
		out[i] = normalizeOne(req, res, false)
	}
	return out
}

// RestoreSpans restores requests in order, using a native batch extension when
// available and otherwise adapting the legacy per-span method serially. It always
// returns one validated receipt per request.
func RestoreSpans(ctx context.Context, backend KVBackend, requests []KVResidencyRequest) []KVResidency {
	if len(requests) == 0 {
		return []KVResidency{}
	}
	if nilContext(ctx) {
		return faultAll(requests, "abi: kv batch canceled: nil context")
	}
	if nilInterface(backend) {
		return faultAll(requests, "abi: kv batch nil backend")
	}
	if err := ctx.Err(); err != nil {
		return faultAll(requests, "abi: kv batch canceled: "+err.Error())
	}
	if batch, ok := backend.(KVBatchRestorer); ok && !nilInterface(batch) {
		got := batch.RestoreSpans(ctx, cloneRequests(requests))
		if err := ctx.Err(); err != nil {
			return faultAll(requests, "abi: kv batch canceled: "+err.Error())
		}
		return normalizeBatch(requests, got, true)
	}

	out := make([]KVResidency, len(requests))
	for i, req := range requests {
		if err := ctx.Err(); err != nil {
			fillFaults(out[i:], requests[i:], "abi: kv batch canceled: "+err.Error())
			break
		}
		res, err := backend.RestoreSpan(ctx, req.Digest)
		if err != nil {
			out[i] = fault(req, "abi: kv batch restore: "+err.Error())
			continue
		}
		out[i] = normalizeOne(req, res, true)
	}
	return out
}

func normalizeBatch(requests []KVResidencyRequest, got []KVResidency, restore bool) []KVResidency {
	out := make([]KVResidency, len(requests))
	if len(got) > len(requests) {
		return faultAll(requests, fmt.Sprintf("abi: kv batch cardinality mismatch: got %d receipts for %d requests", len(got), len(requests)))
	}
	for i, req := range requests {
		if i >= len(got) {
			out[i] = fault(req, fmt.Sprintf("abi: kv batch cardinality mismatch: missing receipt %d of %d", i, len(requests)))
			continue
		}
		out[i] = normalizeOne(req, got[i], restore)
	}
	return out
}

func normalizeOne(req KVResidencyRequest, got KVResidency, restore bool) KVResidency {
	if got.Digest != req.Digest {
		return fault(req, fmt.Sprintf("abi: kv batch digest mismatch: got %q", got.Digest))
	}
	if restore && got.Positions == 0 && (got.Outcome == KVResidencyMiss || got.Outcome == KVResidencyFault) {
		got.Positions = req.Positions
	}
	if got.Positions != req.Positions {
		return fault(req, fmt.Sprintf("abi: kv batch position mismatch: got %d", got.Positions))
	}
	switch got.Outcome {
	case KVResidencyOK, KVResidencyMiss, KVResidencyFault:
		return got
	default:
		return fault(req, fmt.Sprintf("abi: kv batch unknown outcome: %d", got.Outcome))
	}
}

func faultAll(requests []KVResidencyRequest, reason string) []KVResidency {
	out := make([]KVResidency, len(requests))
	fillFaults(out, requests, reason)
	return out
}

func fillFaults(out []KVResidency, requests []KVResidencyRequest, reason string) {
	for i, req := range requests {
		out[i] = fault(req, reason)
	}
}

func fault(req KVResidencyRequest, reason string) KVResidency {
	return KVResidency{Outcome: KVResidencyFault, Digest: req.Digest, Positions: req.Positions, Reason: reason}
}

func cloneRequests(requests []KVResidencyRequest) []KVResidencyRequest {
	return append([]KVResidencyRequest(nil), requests...)
}

func nilContext(ctx context.Context) bool { return ctx == nil || nilInterface(ctx) }

func nilInterface(v any) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return rv.IsNil()
	default:
		return false
	}
}
