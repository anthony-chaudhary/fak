package compute

import (
	"errors"
	"testing"
)

// metadataCounterBackend wraps a real backend and adds the optional
// backendExecutionSnapshotter extension so BuildCaptureMetadata can read live
// transfer counters. It lets a test drive the D2H counter to prove the
// capture-metadata build observes zero device->host traffic, and, in the
// negative arm, that a nonzero counter fails closed rather than passing.
type metadataCounterBackend struct {
	Backend
	counters BackendCounterSnapshot
	observed bool

	pendingD2HBytes uint64
	pendingD2HCount uint64
	snapshots       int
}

func (b *metadataCounterBackend) BackendExecutionSnapshot() (BackendExecutionSnapshot, error) {
	b.snapshots++
	if b.snapshots > 1 && b.pendingD2HBytes > 0 {
		b.counters.D2HBytes += b.pendingD2HBytes
		b.counters.D2HCount += b.pendingD2HCount
		b.pendingD2HBytes = 0
		b.pendingD2HCount = 0
	}
	return BackendExecutionSnapshot{
		Identity: BackendRuntimeIdentity{
			Backend: b.Backend.Name(),
			Device:  "test-device",
			Driver:  "test-driver",
			Runtime: "test-runtime",
		},
		Counters:                 b.counters,
		TransferCountersObserved: b.observed,
	}, nil
}

// failAfterFirstSnapshot arms a device->host copy that lands between the opening
// and closing snapshots of the next build. A capture-time metadata build must
// never see such a copy; the counter delta is what makes it visible.
func (b *metadataCounterBackend) failAfterFirstSnapshot(bytes, count uint64) {
	b.pendingD2HBytes = bytes
	b.pendingD2HCount = count
	b.snapshots = 0
}

// TestCaptureMetadataNoDeviceToHostSync is the capture-time zero-D2H witness.
// It proves that building the metadata for a ragged prompt/draft panel derives
// every value from host-resident offsets and observes zero device->host copies
// against a backend's live transfer counters, and that a nonzero D2H reading
// fails closed instead of silently passing.
func TestCaptureMetadataNoDeviceToHostSync(t *testing.T) {
	promptLengths := []int{1, 5, 2}
	draftK := 4

	// --- Positive arm: no device->host copy during the build. ---
	be := &metadataCounterBackend{
		Backend:  Pick("cpu-ref"),
		observed: true,
	}

	meta, err := BuildCaptureMetadata(be, promptLengths, draftK)
	if err != nil {
		t.Fatalf("capture metadata build with no D2H must succeed, got: %v", err)
	}
	if !meta.D2HObserved {
		t.Fatal("expected live transfer counters to be observed, got D2HObserved=false")
	}
	if meta.D2HBytes != 0 || meta.D2HCount != 0 {
		t.Fatalf("expected zero D2H at capture time, got bytes=%d count=%d", meta.D2HBytes, meta.D2HCount)
	}

	// Shape-consistency: host-derived values match the declared request shape and
	// the merge offsets the same host mirror would produce.
	if meta.NumRequests != len(promptLengths) {
		t.Fatalf("NumRequests = %d, want %d", meta.NumRequests, len(promptLengths))
	}
	wantTotal := 0
	for _, p := range promptLengths {
		wantTotal += p + draftK
	}
	if meta.TotalTokens != wantTotal {
		t.Fatalf("TotalTokens = %d, want %d", meta.TotalTokens, wantTotal)
	}
	wantStart := 0
	for i, p := range promptLengths {
		if meta.QueryStartLoc[i] != wantStart {
			t.Fatalf("QueryStartLoc[%d] = %d, want %d", i, meta.QueryStartLoc[i], wantStart)
		}
		if meta.QueryLens[i] != p {
			t.Fatalf("QueryLens[%d] = %d, want %d", i, meta.QueryLens[i], p)
		}
		wantStart += p + draftK
	}

	// The host mirror must reproduce the exact device-derived offsets: the
	// offsets that lay out the merged buffer are the capture metadata.
	offsets, err := ComputeRaggedPromptDraftOffsets(promptLengths, draftK)
	if err != nil {
		t.Fatalf("host offset derivation failed: %v", err)
	}
	for i := range offsets.MergedOffsets {
		if meta.QueryStartLoc[i] != offsets.MergedOffsets[i] {
			t.Fatalf("metadata offset %d = %d drifts from host mirror %d", i, meta.QueryStartLoc[i], offsets.MergedOffsets[i])
		}
	}

	// --- Negative arm: a D2H copy during the build fails closed. ---
	be.failAfterFirstSnapshot(4096, 1)
	if _, err := BuildCaptureMetadata(be, promptLengths, draftK); !errors.Is(err, ErrCaptureMetadataDeviceToHost) {
		t.Fatalf("expected ErrCaptureMetadataDeviceToHost when a D2H copy is observed, got: %v", err)
	}

	// --- Unavailable arm: a backend without counters is not a fabricated pass. ---
	plain := &metadataCounterBackend{
		Backend:  Pick("cpu-ref"),
		observed: false,
	}
	metaNoCounters, err := BuildCaptureMetadata(plain, promptLengths, draftK)
	if err != nil {
		t.Fatalf("build without counter support must still derive metadata: %v", err)
	}
	if metaNoCounters.D2HObserved {
		t.Fatal("a backend without live counters must report D2HObserved=false, never a fabricated zero")
	}
}

// TestCaptureMetadataFailClosed pins the shape-validation and argument guards so
// a malformed panel is rejected before any metadata is trusted.
func TestCaptureMetadataFailClosed(t *testing.T) {
	be := &metadataCounterBackend{Backend: Pick("cpu-ref"), observed: true}

	if _, err := BuildCaptureMetadata(nil, []int{1}, 0); err == nil {
		t.Fatal("expected an error when backend is nil")
	}
	if _, err := BuildCaptureMetadata(be, nil, 0); err == nil {
		t.Fatal("expected an error on empty prompt lengths")
	}
	if _, err := BuildCaptureMetadata(be, []int{1, 2}, -1); err == nil {
		t.Fatal("expected an error on negative draft K")
	}
	if _, err := BuildCaptureMetadata(be, []int{1, -3}, 2); err == nil {
		t.Fatal("expected an error on a negative prompt length")
	}
}
