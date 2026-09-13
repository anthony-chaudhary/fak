package compute

import (
	"bytes"
	"errors"
	"testing"
)

// fakeWCDeviceSurface is an in-memory WCDeviceDMASurface. It records the bytes it moved and
// whether the DMA path was taken, and can be configured incapable and/or to fail.
type fakeWCDeviceSurface struct {
	supports  bool
	failWith  error
	shortBy   int
	calls     int
	bytesRead int
	lastDst   []byte
	lastSrc   []byte
}

func (f *fakeWCDeviceSurface) SupportsDeviceDMA() bool { return f.supports }

func (f *fakeWCDeviceSurface) CopyDeviceToHost(dst, src []byte) (int, error) {
	f.calls++
	f.lastDst = dst
	f.lastSrc = src
	if f.failWith != nil {
		return 0, f.failWith
	}
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	if f.shortBy > 0 {
		n -= f.shortBy
		if n < 0 {
			n = 0
		}
	}
	copy(dst[:n], src[:n])
	f.bytesRead += n
	return n, nil
}

func TestWCDeviceDMACopySelectsDeviceDMA(t *testing.T) {
	src := []byte("write-combined payload that must move bit-exactly via dma")

	t.Run("dma capable surface selects device path", func(t *testing.T) {
		surface := &fakeWCDeviceSurface{supports: true}
		adapter := NewWCDeviceCopyAdapter(surface)
		dst := make([]byte, len(src))

		res, err := adapter.Copy(dst, src)
		if err != nil {
			t.Fatalf("Copy: %v", err)
		}
		if res.Path != WCCopyPathDeviceDMA {
			t.Fatalf("path = %q, want %q", res.Path, WCCopyPathDeviceDMA)
		}
		if res.Bytes != len(src) {
			t.Fatalf("result bytes = %d, want %d", res.Bytes, len(src))
		}
		if !bytes.Equal(dst, src) {
			t.Fatalf("dst = %q, want %q", dst, src)
		}
		if surface.calls != 1 || surface.bytesRead != len(src) {
			t.Fatalf("surface calls=%d bytesRead=%d, want 1 and %d", surface.calls, surface.bytesRead, len(src))
		}
		if adapter.Surface() != surface {
			t.Fatalf("adapter surface did not round-trip")
		}
	})

	t.Run("nil surface falls back to cpu memcpy", func(t *testing.T) {
		adapter := NewWCDeviceCopyAdapter(nil)
		dst := make([]byte, len(src))

		res, err := adapter.Copy(dst, src)
		if err != nil {
			t.Fatalf("Copy: %v", err)
		}
		if res.Path != WCCopyPathCPUMemcpy {
			t.Fatalf("path = %q, want %q", res.Path, WCCopyPathCPUMemcpy)
		}
		if res.Bytes != len(src) {
			t.Fatalf("result bytes = %d, want %d", res.Bytes, len(src))
		}
		if !bytes.Equal(dst, src) {
			t.Fatalf("dst = %q, want %q", dst, src)
		}
	})

	t.Run("incapable surface falls back to cpu memcpy", func(t *testing.T) {
		surface := &fakeWCDeviceSurface{supports: false}
		adapter := NewWCDeviceCopyAdapter(surface)
		dst := make([]byte, len(src))

		res, err := adapter.Copy(dst, src)
		if err != nil {
			t.Fatalf("Copy: %v", err)
		}
		if res.Path != WCCopyPathCPUMemcpy {
			t.Fatalf("path = %q, want %q", res.Path, WCCopyPathCPUMemcpy)
		}
		if !bytes.Equal(dst, src) {
			t.Fatalf("dst = %q, want %q", dst, src)
		}
		if surface.calls != 0 {
			t.Fatalf("surface calls = %d, want 0 (incapable surface must not be invoked)", surface.calls)
		}
	})

	t.Run("device error is fail-closed and not flipped to cpu", func(t *testing.T) {
		boom := errors.New("dma engine parked")
		surface := &fakeWCDeviceSurface{supports: true, failWith: boom}
		adapter := NewWCDeviceCopyAdapter(surface)
		dst := make([]byte, len(src))

		res, err := adapter.Copy(dst, src)
		if err == nil {
			t.Fatal("Copy: want device error, got nil")
		}
		if res.Path != WCCopyPathDeviceDMA {
			t.Fatalf("path = %q, want %q (must not silently flip to cpu)", res.Path, WCCopyPathDeviceDMA)
		}
		var devErr *WCCopyDeviceError
		if !errors.As(err, &devErr) {
			t.Fatalf("err type = %T, want *WCCopyDeviceError", err)
		}
		if devErr.Kind != WCDeviceErrorKindDMA {
			t.Fatalf("kind = %q, want %q", devErr.Kind, WCDeviceErrorKindDMA)
		}
		if !errors.Is(err, boom) {
			t.Fatalf("err does not wrap the surface failure: %v", err)
		}
		if res.Err == nil {
			t.Fatal("result Err is nil on device failure")
		}
	})

	t.Run("short dma transfer is a typed error", func(t *testing.T) {
		surface := &fakeWCDeviceSurface{supports: true, shortBy: 4}
		adapter := NewWCDeviceCopyAdapter(surface)
		dst := make([]byte, len(src))

		res, err := adapter.Copy(dst, src)
		if err == nil {
			t.Fatal("Copy: want short-transfer error, got nil")
		}
		var devErr *WCCopyDeviceError
		if !errors.As(err, &devErr) {
			t.Fatalf("err type = %T, want *WCCopyDeviceError", err)
		}
		if devErr.Kind != WCDeviceErrorKindShort {
			t.Fatalf("kind = %q, want %q", devErr.Kind, WCDeviceErrorKindShort)
		}
		if res.Path != WCCopyPathDeviceDMA {
			t.Fatalf("path = %q, want %q", res.Path, WCCopyPathDeviceDMA)
		}
	})

	t.Run("byte accounting is min of extents", func(t *testing.T) {
		cases := []struct {
			name  string
			dstN  int
			srcN  int
			path  WCCopyPath
			bytes int
		}{
			{"equal extents dma", 32, 32, WCCopyPathDeviceDMA, 32},
			{"dst shorter dma", 8, 32, WCCopyPathDeviceDMA, 8},
			{"src shorter dma", 32, 8, WCCopyPathDeviceDMA, 8},
			{"equal extents cpu", 32, 32, WCCopyPathCPUMemcpy, 32},
			{"dst shorter cpu", 8, 32, WCCopyPathCPUMemcpy, 8},
			{"src shorter cpu", 32, 8, WCCopyPathCPUMemcpy, 8},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				srcBytes := bytes.Repeat([]byte{0xAB}, tc.srcN)
				dst := make([]byte, tc.dstN)

				var adapter *WCDeviceCopyAdapter
				if tc.path == WCCopyPathDeviceDMA {
					adapter = NewWCDeviceCopyAdapter(&fakeWCDeviceSurface{supports: true})
				} else {
					adapter = NewWCDeviceCopyAdapter(nil)
				}

				res, err := adapter.Copy(dst, srcBytes)
				if err != nil {
					t.Fatalf("Copy: %v", err)
				}
				if res.Path != tc.path {
					t.Fatalf("path = %q, want %q", res.Path, tc.path)
				}
				want := tc.dstN
				if tc.srcN < want {
					want = tc.srcN
				}
				if res.Bytes != want {
					t.Fatalf("bytes = %d, want %d", res.Bytes, want)
				}
				if !bytes.Equal(dst[:want], srcBytes[:want]) {
					t.Fatalf("dst prefix not filled correctly for %s", tc.name)
				}
			})
		}
	})

	t.Run("empty extents are a no-op", func(t *testing.T) {
		adapter := NewWCDeviceCopyAdapter(&fakeWCDeviceSurface{supports: true})

		res, err := adapter.Copy(nil, nil)
		if err != nil {
			t.Fatalf("Copy(nil,nil): %v", err)
		}
		if res.Bytes != 0 || res.Path != WCCopyPathCPUMemcpy {
			t.Fatalf("empty result = %+v, want zero bytes on cpu path", res)
		}
	})
}
