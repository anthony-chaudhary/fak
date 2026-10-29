//go:build cuda && cgo

package compute

import (
	"errors"
	"reflect"
	"sync/atomic"
	"testing"
	"unsafe"
)

// TestCUDAReadChecked executes the actual Read method, never a copied substitute.
// All native operations are replaced before an owned Go buffer is used as an
// opaque pointer. This test must remain nonparallel; installation/restoration and
// Read itself use cudaMu. It needs no device and must not skip on a CPU host.
// fak-test:runtime fast est=5ms lane=default
// Estimate covers injected calls only, not compilation or CUDA package init.
func TestCUDAReadChecked(t *testing.T) {
	cases := []struct {
		name                                                   string
		dtype                                                  Dtype
		device, status                                         int
		nilLatch, poisoned, invalid, host, empty, convertPanic bool
	}{
		{name: "success-device0", dtype: F32},
		{name: "copy-failure", dtype: F32, status: 1007},
		{name: "selection-failure", dtype: F32, device: 1, status: 1009},
		{name: "nil-latch-typed-failure", dtype: F32, status: 1007, nilLatch: true},
		{name: "poisoned-admission", dtype: F32, poisoned: true},
		{name: "poisoned-narrowed-admission", dtype: F16, poisoned: true},
		{name: "invalid-state", dtype: F32, invalid: true},
		{name: "host-poisoned-fast-path", dtype: F32, host: true, poisoned: true},
		{name: "empty-poisoned-no-fence", dtype: F32, empty: true, poisoned: true},
		{name: "nonzero-device-fence-isolation", dtype: F32, device: 1},
		{name: "f16-success-cleanup", dtype: F16},
		{name: "bf16-success-cleanup", dtype: BF16},
		{name: "f16-copy-failure-cleanup", dtype: F16, status: 1007},
		{name: "narrowed-wrong-owner", dtype: BF16, device: 1},
		{name: "conversion-panic-cleanup", dtype: F16, convertPanic: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := &cudaBackend{fenceGen: 41}
			if !tc.nilLatch {
				c.faultLatch = NewDeviceFaultLatch("cuda", 3)
			}
			if tc.poisoned {
				c.faultLatch.Observe(DeviceFaultExecution, "prior", 1700)
			}
			storage := []float32{3, 4}
			scratch := []float32{0, 0}
			db := &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: 2 * tc.dtype.Bytes(), device: tc.device, be: c, bornGen: 41}
			if tc.invalid {
				db.invalid = 1
			}
			tensor := Tensor{Dtype: tc.dtype, Layout: RowMajor, Shape: []int{2}, buf: db, be: c}
			if tc.host {
				tensor.buf = &hostBuf{f32: storage}
			}
			if tc.empty {
				tensor.Shape = []int{0}
			}
			calls, allocs, converts, frees := 0, 0, 0, 0
			narrowed := tc.dtype == F16 || tc.dtype == BF16
			ops := cudaReadOperations{
				copy: func(device int, dst, src unsafe.Pointer, nbytes uintptr) int {
					calls++
					if device != tc.device || nbytes != 8 {
						t.Errorf("copy domain/size=(%d,%d), want(%d,8)", device, nbytes, tc.device)
					}
					want := unsafe.Pointer(&storage[0])
					if narrowed {
						want = unsafe.Pointer(&scratch[0])
					}
					if src != want {
						t.Error("copy used wrong source/lifetime")
					}
					target := unsafe.Slice((*float32)(dst), 2)
					target[0] = 11 // even a partially written host destination cannot be published
					if tc.status == 0 {
						target[1] = 22
					}
					return tc.status
				},
				allocConversion: func(got *cudaBackend, nbytes int) unsafe.Pointer {
					allocs++
					if got != c || nbytes != 8 {
						t.Error("conversion allocation contract changed")
					}
					return unsafe.Pointer(&scratch[0])
				},
				convert: func(dst, src unsafe.Pointer, n int, dtype Dtype) {
					converts++
					if dst != unsafe.Pointer(&scratch[0]) || src != unsafe.Pointer(&storage[0]) || n != 2 || dtype != tc.dtype {
						t.Error("conversion contract changed")
					}
					if tc.convertPanic {
						panic("injected conversion failure")
					}
				},
				freeConversion: func(ptr unsafe.Pointer) {
					frees++
					if ptr != unsafe.Pointer(&scratch[0]) {
						t.Error("freed wrong conversion allocation")
					}
				},
			}
			cudaMu.Lock()
			prior := cudaReadNative
			cudaReadNative = ops
			cudaMu.Unlock()
			t.Cleanup(func() { cudaMu.Lock(); cudaReadNative = prior; cudaMu.Unlock() })
			read := func() (out []float32, failure any) {
				defer func() { failure = recover() }()
				out = c.Read(tensor)
				return
			}
			out, failure := read()
			noCopy := tc.host || tc.empty || tc.poisoned || tc.invalid || (narrowed && tc.device != 0) || tc.convertPanic
			wantCalls := 1
			if noCopy {
				wantCalls = 0
			}
			if calls != wantCalls {
				t.Errorf("native copy calls=%d, want%d", calls, wantCalls)
			}
			wantConvert := 0
			if narrowed && !tc.host && !tc.empty && !tc.poisoned && !tc.invalid && tc.device == 0 {
				wantConvert = 1
			}
			if allocs != wantConvert || converts != wantConvert || frees != wantConvert {
				t.Errorf("conversion lifetime alloc/convert/free=%d/%d/%d, want%d each", allocs, converts, frees, wantConvert)
			}
			shouldFail := !tc.host && !tc.empty && (tc.poisoned || tc.invalid || (narrowed && tc.device != 0) || tc.convertPanic || tc.status != 0)
			if shouldFail {
				if failure == nil {
					t.Error("failed Read published host result instead of panicking")
				}
				if out != nil {
					t.Errorf("failed Read returned partial host data %v", out)
				}
			} else {
				if failure != nil {
					t.Errorf("successful/host/empty Read panicked: %v", failure)
				}
				if tc.host {
					if len(out) != 2 || &out[0] != &storage[0] {
						t.Error("host fast path lost original view")
					}
				} else if tc.empty {
					if len(out) != 0 {
						t.Errorf("empty result=%v", out)
					}
				} else if !reflect.DeepEqual(out, []float32{11, 22}) {
					t.Errorf("successful result=%v", out)
				}
			}
			wantGen := uint64(41)
			if !shouldFail && !tc.host && !tc.empty && tc.device == 0 {
				wantGen++
			}
			if gen := atomic.LoadUint64(&c.fenceGen); gen != wantGen {
				t.Errorf("fence generation=%d, want%d", gen, wantGen)
			}
			if got := db.Ready(); got != (wantGen > 41 && !tc.invalid) {
				t.Errorf("pending buffer Ready=%t without justified device0 fence", got)
			}
			if tc.poisoned && !tc.host && !tc.empty {
				var fault *DeviceFaultError
				err, _ := failure.(error)
				if !errors.As(err, &fault) {
					t.Errorf("poison refusal lost typed device fault: %#v", failure)
				}
			}
			if tc.invalid {
				var invalid *Qwen35GDNInvalidStateError
				err, _ := failure.(error)
				if !errors.As(err, &invalid) {
					t.Errorf("invalid-state refusal changed: %#v", failure)
				}
			}
			if narrowed && tc.device != 0 && !tc.poisoned && failure != "cuda: narrowed KV read requires its owning device" {
				t.Errorf("wrong-owner refusal changed: %#v", failure)
			}
			if tc.convertPanic && failure != "injected conversion failure" {
				t.Errorf("conversion panic changed: %#v", failure)
			}
			if tc.status != 0 && !noCopy {
				err, ok := failure.(error)
				var op *CUDAOpError
				if !ok || !errors.As(err, &op) || op == nil || op.Op != "Read" || op.Site != "read-d2h" || op.Code != tc.status {
					t.Errorf("failure lost typed Op/Site/Code: %#v", failure)
				}
				if c.faultLatch != nil {
					snap := c.faultLatch.Snapshot()
					if snap.Health != DevicePoisoned || snap.LastClass != DeviceFaultExecution || snap.LastSite != "read-d2h" || snap.LastCode != tc.status {
						t.Errorf("native failure not directly latched: %+v", snap)
					}
					before := calls
					again, refusal := read()
					var fault *DeviceFaultError
					refErr, _ := refusal.(error)
					if again != nil || !errors.As(refErr, &fault) || calls != before {
						t.Errorf("post-fault Read was not refused before copy: result=%v panic=%v calls=%d->%d", again, refusal, before, calls)
					}
				}
			}
		})
	}
}
