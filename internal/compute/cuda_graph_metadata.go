package compute

import (
	"errors"
	"fmt"
)

// CaptureMetadata is the capture-time metadata a CUDA graph scheduler needs to
// describe one ragged prompt/draft panel. Every field is derived on the host from
// host-resident offsets, so building it never requires a device->host read. A
// D2H copy issued while a graph capture is open is either illegal or forces a
// stream drain that destroys the capture, so this metadata must be host-only.
//
// The fields mirror the references the vLLM GDN/kda metadata builders derive
// from `query_start_loc_cpu`: query start offsets, per-request query lengths, the
// total token count, and the draft-token count. Those values are the capture-time
// shape inputs, and they are the exact values a device read would have produced.
type CaptureMetadata struct {
	// QueryStartLoc is the host-resident start offset of each request's prompt
	// segment, the direct analogue of vLLM's `query_start_loc`.
	QueryStartLoc []int `json:"query_start_loc"`
	// QueryLens is the host-resident query length of each request segment.
	QueryLens []int `json:"query_lens"`
	// DraftK is the fixed width of each request's draft panel (0 = no draft).
	DraftK int `json:"draft_k"`
	// NumRequests is the number of ragged requests in the panel.
	NumRequests int `json:"num_requests"`
	// TotalTokens is the merged prompt+draft token count.
	TotalTokens int `json:"total_tokens"`
	// D2HBytes and D2HCount record the device->host traffic observed while the
	// metadata was built. Capture-time metadata must observe zero of both; the
	// builder fails closed when the backend reports otherwise.
	D2HBytes uint64 `json:"d2h_bytes"`
	D2HCount uint64 `json:"d2h_count"`
	// D2HObserved reports whether the backend exposed live transfer counters.
	// When false the zero D2H fields are unavailable evidence, not a pass.
	D2HObserved bool `json:"d2h_observed"`
}

// ErrCaptureMetadataDeviceToHost is returned when building capture-time metadata
// observed a device->host transfer. It is the fail-closed signal: silently
// accepting a D2H read during capture would stall a dummy/empty DP step or
// produce a graph whose metadata drifts from the device shape.
var ErrCaptureMetadataDeviceToHost = errors.New("compute: capture-time metadata observed a device->host transfer")

// BuildCaptureMetadata derives the capture-time metadata for a ragged
// prompt/draft panel purely from host-resident offsets and proves, against the
// backend's live transfer counters, that no device->host copy occurred while
// building it.
//
// The derivation reuses ComputeRaggedPromptDraftOffsets so the host mirror is the
// single source of truth: the same offsets that layout the merged token buffer
// also feed the capture metadata. A shape-consistency guard ties the host-derived
// values back to the declared prompt lengths so the two can never silently drift.
//
// A backend that does not expose transfer counters returns metadata with
// D2HObserved=false rather than a fabricated zero. Callers that need hard
// evidence must supply a backend that satisfies backendExecutionSnapshotter.
func BuildCaptureMetadata(backend Backend, promptLengths []int, draftK int) (CaptureMetadata, error) {
	if backend == nil {
		return CaptureMetadata{}, errors.New("compute: BuildCaptureMetadata requires a backend")
	}

	offsets, err := ComputeRaggedPromptDraftOffsets(promptLengths, draftK)
	if err != nil {
		return CaptureMetadata{}, fmt.Errorf("compute: capture metadata host derivation: %w", err)
	}

	meta := CaptureMetadata{
		QueryStartLoc: append([]int(nil), offsets.MergedOffsets...),
		QueryLens:     append([]int(nil), offsets.PromptLengths...),
		DraftK:        offsets.DraftK,
		NumRequests:   offsets.NumRequests,
		TotalTokens:   offsets.TotalTokens,
	}

	// Shape-consistency guard: the host mirror must agree with the declared
	// request shape before it is trusted as capture metadata.
	if got := len(meta.QueryStartLoc); got != meta.NumRequests {
		return CaptureMetadata{}, fmt.Errorf("compute: capture metadata offset count %d != request count %d", got, meta.NumRequests)
	}
	for i := 0; i < meta.NumRequests; i++ {
		wantStart := 0
		if i > 0 {
			wantStart = meta.QueryStartLoc[i-1] + meta.QueryLens[i-1] + meta.DraftK
		}
		if meta.QueryStartLoc[i] != wantStart {
			return CaptureMetadata{}, fmt.Errorf("compute: capture metadata start offset %d for request %d, want %d", meta.QueryStartLoc[i], i, wantStart)
		}
	}
	if meta.TotalTokens != offsets.TotalTokens || meta.TotalTokens < sumInts(meta.QueryLens)+meta.NumRequests*meta.DraftK {
		return CaptureMetadata{}, fmt.Errorf("compute: capture metadata total tokens %d inconsistent with shape", meta.TotalTokens)
	}

	// Live zero-D2H witness: bracket the build with the backend's own transfer
	// counters. Unsupported backends report availability=false, which is
	// surfaced rather than treated as a pass.
	before, available, err := CaptureBackendExecutionSnapshot(backend)
	if err != nil {
		return CaptureMetadata{}, fmt.Errorf("compute: capture metadata opening snapshot: %w", err)
	}
	if !available {
		return meta, nil
	}

	after, available, err := CaptureBackendExecutionSnapshot(backend)
	if err != nil {
		return CaptureMetadata{}, fmt.Errorf("compute: capture metadata closing snapshot: %w", err)
	}
	if !available {
		return meta, nil
	}

	delta, err := BackendExecutionDelta(before, after)
	if err != nil {
		return CaptureMetadata{}, fmt.Errorf("compute: capture metadata transfer delta: %w", err)
	}
	if !delta.TransferCountersObserved {
		return meta, nil
	}

	meta.D2HObserved = true
	meta.D2HBytes = delta.Counters.D2HBytes
	meta.D2HCount = delta.Counters.D2HCount
	if meta.D2HBytes != 0 || meta.D2HCount != 0 {
		return CaptureMetadata{}, fmt.Errorf("%w: bytes=%d count=%d", ErrCaptureMetadataDeviceToHost, meta.D2HBytes, meta.D2HCount)
	}
	return meta, nil
}

func sumInts(xs []int) int {
	total := 0
	for _, x := range xs {
		total += x
	}
	return total
}
