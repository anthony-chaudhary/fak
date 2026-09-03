package compute

import (
	"fmt"
	"testing"
)

// offlineGraphCapturer simulates a device backend supporting stream capture (#10716).
type offlineGraphCapturer struct {
	Backend
	capturing bool
	began     int
	ended     int
	aborted   int
	emissions []string
}

func (c *offlineGraphCapturer) GraphBegin() bool {
	c.capturing = true
	c.began++
	return true
}

func (c *offlineGraphCapturer) GraphEndLaunch() {
	c.capturing = false
	c.ended++
}

func (c *offlineGraphCapturer) GraphAbort() {
	c.capturing = false
	c.aborted++
}

func (c *offlineGraphCapturer) IsCapturing() bool {
	return c.capturing
}

func (c *offlineGraphCapturer) UploadConstantParam(dst Tensor, data []float32, paramKey uint64, lastUploaded *uint64) {
	if !c.capturing && lastUploaded != nil && *lastUploaded == paramKey {
		return
	}
	if hb, ok := dst.buf.(HostBuffer); ok && len(data) > 0 {
		copy(hb.F32(), data)
	}
	if lastUploaded != nil {
		*lastUploaded = paramKey
	}
	c.emissions = append(c.emissions, fmt.Sprintf("key=%d,len=%d,capturing=%v", paramKey, len(data), c.capturing))
}

// TestCUDAGraphOfflineSelfContainedConstantUploadInvariant pins the TurboQuant #10716
// invariant across every platform in pure Go: parameter and constant memory uploads
// must be emitted unconditionally into the stream when graph capture is active,
// bypassing any skip-if-same host caching.
func TestCUDAGraphOfflineSelfContainedConstantUploadInvariant(t *testing.T) {
	capturer := &offlineGraphCapturer{
		Backend: Pick("cpu-ref"),
	}

	// 1. Verify initial state.
	if capturer.IsCapturing() {
		t.Fatal("#10716 offline: IsCapturing() must report false initially")
	}

	// 2. Verify lifecycle tracking with GraphBegin and GraphAbort.
	if !capturer.GraphBegin() {
		t.Fatal("#10716 offline: GraphBegin failed")
	}
	if !capturer.IsCapturing() {
		t.Fatal("#10716 offline: IsCapturing() must report true after GraphBegin")
	}
	capturer.GraphAbort()
	if capturer.IsCapturing() {
		t.Fatal("#10716 offline: IsCapturing() must report false after GraphAbort")
	}

	// 3. Verify lifecycle tracking with GraphBegin and GraphEndLaunch.
	if !capturer.GraphBegin() {
		t.Fatal("#10716 offline: GraphBegin failed")
	}
	if !capturer.IsCapturing() {
		t.Fatal("#10716 offline: IsCapturing() must report true after GraphBegin")
	}
	capturer.GraphEndLaunch()
	if capturer.IsCapturing() {
		t.Fatal("#10716 offline: IsCapturing() must report false after GraphEndLaunch")
	}

	// 4. Outside capture: host caching should skip identical consecutive keys.
	dst := NewF32(capturer, []int{4}, []float32{0, 0, 0, 0})
	var lastKey uint64

	paramA := []float32{1.0, 2.0, 3.0, 4.0}
	paramB := []float32{5.0, 6.0, 7.0, 8.0}

	capturer.emissions = nil
	UploadConstantParam(dst, paramA, 10, &lastKey)
	if len(capturer.emissions) != 1 || lastKey != 10 {
		t.Fatalf("#10716 offline: expected 1 emission for key 10, got %d (lastKey=%d)", len(capturer.emissions), lastKey)
	}

	// Repeat with same key outside capture -> must be skipped.
	UploadConstantParam(dst, paramA, 10, &lastKey)
	if len(capturer.emissions) != 1 {
		t.Fatalf("#10716 offline: expected skip outside capture, but got emission count %d", len(capturer.emissions))
	}

	// 5. Under stream capture: alternating and repeating configs must be emitted unconditionally.
	capturer.emissions = nil
	if !capturer.GraphBegin() {
		t.Fatal("#10716 offline: GraphBegin failed")
	}

	// Simulating alternating layer configs:
	// Layer 0: config A (key 10)
	// Layer 1: config B (key 20)
	// Layer 0: config B (key 20) -> Buggy pattern would skip this because lastKey == 20.
	// Invariant requirement: under capture, must be emitted unconditionally!
	UploadConstantParam(dst, paramA, 10, &lastKey)
	UploadConstantParam(dst, paramB, 20, &lastKey)
	UploadConstantParam(dst, paramB, 20, &lastKey)

	if len(capturer.emissions) != 3 {
		t.Fatalf("#10716 offline: expected 3 unconditional emissions under capture, got %d (%v)",
			len(capturer.emissions), capturer.emissions)
	}

	capturer.GraphEndLaunch()
	if capturer.IsCapturing() {
		t.Fatal("#10716 offline: IsCapturing() must report false after GraphEndLaunch")
	}

	// 6. After capture ends: skip-if-same host caching resumes.
	capturer.emissions = nil
	UploadConstantParam(dst, paramB, 20, &lastKey) // lastKey is 20 from previous call
	if len(capturer.emissions) != 0 {
		t.Fatalf("#10716 offline: expected skip after capture ended, got %d emissions", len(capturer.emissions))
	}

	// Upload new key outside capture -> emitted.
	UploadConstantParam(dst, paramA, 10, &lastKey)
	if len(capturer.emissions) != 1 || lastKey != 10 {
		t.Fatalf("#10716 offline: expected 1 emission for key 10, got %d (lastKey=%d)", len(capturer.emissions), lastKey)
	}
}
