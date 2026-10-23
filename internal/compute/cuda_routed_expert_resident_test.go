//go:build cuda && cgo

// Resident routed-expert CUDA GEMV witnesses (RoutedExpertResidentKernel). On a CUDA host,
// after build_cuda.sh build:
//
//	FAK_CUDA_ROUTED_EXPERT_REQUIRED=1 go test -tags cuda ./internal/compute -run CUDARoutedExpertResident

package compute

import (
	"errors"
	"fmt"
	"math/rand"
	"runtime"
	"testing"
)

// residentFormat binds one packed format's upload, scalar oracle, fixture and fault site.
type residentFormat struct {
	name   string
	site   string
	upload func(c *cudaBackend, O, I int, w, s []byte) (RoutedExpertResidentWeight, error)
	oracle func(O, I int, w, s []byte, x []float32) ([]float32, error)
	gen    func(seed int64, c routedExpertCase) (w, s []byte, x []float32)
}

var residentFormats = []residentFormat{
	{
		name: "fp8", site: "routed-expert-resident-fp8-block32",
		upload: (*cudaBackend).UploadExpertFP8Block32E8M0,
		oracle: func(O, I int, w, s []byte, x []float32) ([]float32, error) {
			return GemvFP8Block32E8M0("oracle", O, I, w, s, x)
		},
		gen: routedExpertFP8Case,
	},
	{
		name: "mxfp4", site: "routed-expert-resident-mxfp4",
		upload: (*cudaBackend).UploadExpertMXFP4E8M0,
		oracle: func(O, I int, w, s []byte, x []float32) ([]float32, error) {
			return GemvMXFP4E8M0("oracle", O, I, w, s, x)
		},
		gen: routedExpertMXFP4Case,
	},
}

// residentCounters is one coherent read of every counter the resident contract is held to.
type residentCounters struct {
	uploads, live, launches, h2d uint64
}

func readResidentCounters(cuda *cudaBackend) residentCounters {
	uploads, live := cudaRoutedExpertResidentCounters()
	return residentCounters{uploads: uploads, live: live, launches: cudaRoutedExpertLaunches(), h2d: cuda.H2DXferBytes()}
}

func (a residentCounters) since(b residentCounters) residentCounters {
	return residentCounters{uploads: a.uploads - b.uploads, live: a.live - b.live, launches: a.launches - b.launches, h2d: a.h2d - b.h2d}
}

func mustUploadResident(tb testing.TB, cuda *cudaBackend, f residentFormat, O, I int, w, s []byte) RoutedExpertResidentWeight {
	tb.Helper()
	h, err := f.upload(cuda, O, I, w, s)
	if err != nil || h == nil {
		tb.Fatalf("%s[%dx%d] upload: handle=%v err=%v", f.name, O, I, h, err)
	}
	return h
}

func mustReleaseResident(tb testing.TB, label string, h RoutedExpertResidentWeight) {
	tb.Helper()
	if err := h.Release(); err != nil {
		tb.Fatalf("%s: Release: %v", label, err)
	}
}

// TestCUDARoutedExpertResidentBitExact uploads each ragged case once and holds GEMVs over
// three different x vectors bit-for-bit to the oracle. Each GEMV is one launch whose H2D is
// exactly 4*I bytes (x only), so the packed weight is never re-sent.
//
// fak-test:runtime integration est=3s lane=optin
func TestCUDARoutedExpertResidentBitExact(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	suites := map[string][]routedExpertCase{
		"fp8": {
			rotating(1, 1), rotating(1, 2), rotating(3, 5), rotating(31, 33), rotating(32, 32),
			rotating(33, 127), rotating(40, 96), rotating(65, 129), rotating(97, 300),
			rotating(130, 257), rotating(1024, 4096), rotating(2048, 1000), rotating(4096, 4096),
			subnormalHeavy(33, 257), subnormalHeavy(256, 4096), overflowing(64, 1), overflowing(64, 2),
		},
		"mxfp4": {
			rotating(1, 2), rotating(1, 4), rotating(3, 6), rotating(31, 34), rotating(32, 64),
			rotating(33, 62), rotating(64, 256), rotating(65, 258), rotating(100, 1000),
			rotating(7, 514), rotating(512, 2048), rotating(2048, 4096),
			subnormalHeavy(33, 258), subnormalHeavy(256, 4096), overflowing(64, 2),
		},
	}
	const xPerUpload = 3
	for _, f := range residentFormats {
		var total routedExpertBitStats
		var seen [256]bool
		for idx, c := range suites[f.name] {
			O, I := c.O, c.I
			seed := int64(1000*len(f.name) + idx)
			w, s, x0 := f.gen(seed, c)
			for _, b := range w {
				seen[b] = true
			}
			start := readResidentCounters(cuda)
			h := mustUploadResident(t, cuda, f, O, I, w, s)
			afterUpload := readResidentCounters(cuda)
			if d := afterUpload.since(start); d.uploads != 1 || d.live != 1 || d.launches != 0 || d.h2d != uint64(len(w)+len(s)) {
				t.Fatalf("%s[%dx%d] upload deltas %+v, want 1 upload, 1 live, 0 launches, %d H2D bytes", f.name, O, I, d, len(w)+len(s))
			}
			rng := rand.New(rand.NewSource(seed * 31))
			for k := 0; k < xPerUpload; k++ {
				x := x0
				if k > 0 {
					x = routedExpertX(rng, I, c.xLo, c.xHi)
				}
				want, err := f.oracle(O, I, w, s, x)
				if err != nil {
					t.Fatalf("%s[%dx%d] oracle: %v", f.name, O, I, err)
				}
				before := readResidentCounters(cuda)
				got, err := h.Gemv(x)
				if err != nil {
					t.Fatalf("%s[%dx%d] x#%d resident GEMV: %v", f.name, O, I, k, err)
				}
				if d := readResidentCounters(cuda).since(before); d.uploads != 0 || d.live != 0 || d.launches != 1 || d.h2d != uint64(4*I) {
					t.Fatalf("%s[%dx%d] x#%d GEMV deltas %+v, want 0 uploads, 1 launch, %d H2D bytes", f.name, O, I, k, d, 4*I)
				}
				total.add(requireRoutedExpertBits(t, fmt.Sprintf("resident-%s[%dx%d]#%d", f.name, O, I, k), got, want))
			}
			mustReleaseResident(t, f.name, h)
			if d := readResidentCounters(cuda).since(start); d.live != 0 {
				t.Fatalf("%s[%dx%d] live resident ids after Release changed by %d", f.name, O, I, int64(d.live))
			}
		}
		if f.name == "fp8" {
			for _, b := range routedExpertE4M3Finite {
				if !seen[b] {
					t.Fatalf("E4M3 byte %#02x never exercised", b)
				}
			}
		} else {
			for v := 0; v < 256; v++ {
				if !seen[v] {
					t.Fatalf("MXFP4 byte %#02x (two E2M1 nibbles) never exercised", v)
				}
			}
		}
		requireBoundaryCoverage(t, "resident-"+f.name, total)
	}
}

// TestCUDARoutedExpertResidentE8M0EveryByteFiniteBitExact holds every E8M0 byte 0..254 to the
// oracle bit-for-bit through an upload and a resident GEMV per byte and format, on the
// finite, normal-product fixture e8m0FiniteByteCase.
//
// fak-test:runtime integration est=2s lane=optin
func TestCUDARoutedExpertResidentE8M0EveryByteFiniteBitExact(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = e8m0FiniteByteO, e8m0FiniteByteI
	for v := 0; v <= 254; v++ {
		b := byte(v)
		for _, f := range residentFormats {
			w, s, x := e8m0FiniteByteCase(t, b, f.name == "mxfp4")
			label := fmt.Sprintf("resident-%s-e8m0[%#02x]", f.name, b)
			want, err := f.oracle(O, I, w, s, x)
			if err != nil {
				t.Fatalf("%s oracle: %v", label, err)
			}
			h := mustUploadResident(t, cuda, f, O, I, w, s)
			before := cudaRoutedExpertLaunches()
			got, err := h.Gemv(x)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			if n := cudaRoutedExpertLaunches() - before; n != 1 {
				t.Fatalf("%s: device launches = %d, want 1", label, n)
			}
			mustReleaseResident(t, label, h)
			st := requireRoutedExpertBits(t, label, got, want)
			if st.inf != 0 || st.nan != 0 || st.normal == 0 {
				t.Fatalf("%s: outputs left the finite bit-exact regime: normal=%d inf=%d nan=%d", label, st.normal, st.inf, st.nan)
			}
		}
	}
}

// TestCUDARoutedExpertResidentReuseWithoutReupload uploads one V4.1-sized projection per
// format and runs many distinct x through it: exactly one upload, one launch per GEMV, and
// H2D of exactly 4*I bytes per GEMV, each output bit-exact.
//
// fak-test:runtime integration est=3s lane=optin
func TestCUDARoutedExpertResidentReuseWithoutReupload(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const calls = 16
	for _, f := range residentFormats {
		O, I := 1024, 2048
		if f.name == "mxfp4" {
			O, I = 2048, 4096
		}
		c := rotating(O, I)
		w, s, _ := f.gen(71, c)
		start := readResidentCounters(cuda)
		h := mustUploadResident(t, cuda, f, O, I, w, s)
		afterUpload := readResidentCounters(cuda)
		rng := rand.New(rand.NewSource(72))
		for k := 0; k < calls; k++ {
			x := routedExpertX(rng, I, c.xLo, c.xHi)
			got, err := h.Gemv(x)
			if err != nil {
				t.Fatalf("%s call %d: %v", f.name, k, err)
			}
			want, err := f.oracle(O, I, w, s, x)
			if err != nil {
				t.Fatal(err)
			}
			requireRoutedExpertBits(t, fmt.Sprintf("reuse-%s#%d", f.name, k), got, want)
		}
		d := readResidentCounters(cuda).since(afterUpload)
		if d.uploads != 0 || d.launches != calls || d.h2d != uint64(calls*4*I) {
			t.Fatalf("%s: %d GEMVs moved %+v, want 0 uploads, %d launches, %d H2D bytes", f.name, calls, d, calls, calls*4*I)
		}
		mustReleaseResident(t, f.name, h)
		if total := readResidentCounters(cuda).since(start); total.uploads != 1 || total.live != 0 {
			t.Fatalf("%s: lifetime deltas %+v, want exactly 1 upload and 0 live after Release", f.name, total)
		}
	}
}

// TestCUDARoutedExpertResidentRefusalParity holds upload plus GEMV to the oracle's acceptance
// set: a weight/scale input the oracle refuses is refused at upload with a nil handle, no
// upload, no live id and no launch; an x the oracle refuses is refused by the GEMV with a nil
// y and no launch; neither poisons the session.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertResidentRefusalParity(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = 40, 96
	with := func(b []byte, k int, v byte) []byte {
		c := append([]byte(nil), b...)
		c[k] = v
		return c
	}
	grow := func(b []byte, n int) []byte { return append(append([]byte(nil), b...), make([]byte, n)...) }
	type tc struct {
		name   string
		O, I   int
		w, s   []byte
		accept bool
	}
	for _, f := range residentFormats {
		w, s, x := f.gen(81, rotating(O, I))
		cases := []tc{
			{"ok", O, I, w, s, true},
			{"zero O", 0, I, w, s, false},
			{"negative I", O, -I, w, s, false},
			{"weight short", O, I, w[:len(w)-1], s, false},
			{"weight long", O, I, grow(w, 1), s, false},
			{"scales short", O, I, w, s[:len(s)-1], false},
			{"scales long", O, I, w, grow(s, 1), false},
			{"e8m0 nan first", O, I, w, with(s, 0, 0xff), false},
			{"e8m0 nan last", O, I, w, with(s, len(s)-1, 0xff), false},
			{"nil weight", O, I, nil, s, false},
		}
		if f.name == "fp8" {
			cases = append(cases,
				tc{"e4m3 nan 0x7f", O, I, with(w, 17, 0x7f), s, false},
				tc{"e4m3 nan 0xff last", O, I, with(w, len(w)-1, 0xff), s, false})
		} else {
			cases = append(cases,
				tc{"0x7f/0xff weight bytes are finite E2M1", O, I, with(with(w, 3, 0x7f), 5, 0xff), s, true},
				tc{"odd I", O, I + 1, grow(w, O/2), s, false})
		}
		for _, c := range cases {
			label := f.name + "/" + c.name
			xc := x
			if c.I > 0 && c.I != I {
				xc = grow32(x, c.I-I)
			}
			_, oerr := f.oracle(c.O, c.I, c.w, c.s, xc)
			before := readResidentCounters(cuda)
			h, uerr := f.upload(cuda, c.O, c.I, c.w, c.s)
			d := readResidentCounters(cuda).since(before)
			if (oerr == nil) != c.accept || (uerr == nil) != c.accept {
				t.Fatalf("%s: oracle err=%v upload err=%v, want accept=%v", label, oerr, uerr, c.accept)
			}
			if !c.accept {
				if h != nil || d.uploads != 0 || d.live != 0 || d.launches != 0 || d.h2d != 0 {
					t.Fatalf("%s: refused upload returned handle=%v with deltas %+v", label, h, d)
				}
				continue
			}
			y, err := h.Gemv(xc)
			if err != nil || len(y) != c.O {
				t.Fatalf("%s: accepted upload GEMV len(y)=%d err=%v", label, len(y), err)
			}
			mustReleaseResident(t, label, h)
		}

		h := mustUploadResident(t, cuda, f, O, I, w, s)
		for _, xc := range []struct {
			name string
			x    []float32
		}{
			{"x short", x[:I-1]},
			{"x long", grow32(x, 1)},
			{"x nil", nil},
		} {
			label := f.name + "/" + xc.name
			if _, oerr := f.oracle(O, I, w, s, xc.x); oerr == nil {
				t.Fatalf("%s: oracle accepted", label)
			}
			before := readResidentCounters(cuda)
			y, err := h.Gemv(xc.x)
			if err == nil || y != nil {
				t.Fatalf("%s: resident GEMV len(y)=%d err=%v, want refusal", label, len(y), err)
			}
			if d := readResidentCounters(cuda).since(before); d != (residentCounters{}) {
				t.Fatalf("%s: refused GEMV moved %+v", label, d)
			}
		}
		want, _ := f.oracle(O, I, w, s, x)
		got, err := h.Gemv(x)
		if err != nil {
			t.Fatalf("%s: GEMV after refused x: %v", f.name, err)
		}
		requireRoutedExpertBits(t, f.name+"/after-refusals", got, want)
		mustReleaseResident(t, f.name, h)
	}
	if hl := cuda.faultLatch.Health(); hl != DeviceHealthy {
		t.Fatalf("refusals poisoned the device session: health=%v", hl)
	}
}

func grow32(x []float32, n int) []float32 {
	return append(append([]float32(nil), x...), make([]float32, n)...)
}

// TestCUDARoutedExpertResidentReleaseLifetime pins the handle lifetime: Release frees the id
// and its pooled allocations; a GEMV after Release, and a second Release, return
// ErrRoutedExpertResidentReleased with no device work and no session poisoning.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertResidentReleaseLifetime(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = 96, 320
	for _, f := range residentFormats {
		w, s, x := f.gen(91, rotating(O, I))
		_, liveAllocsBefore := cuda.cudaAllocationCountsForTest()
		start := readResidentCounters(cuda)
		h := mustUploadResident(t, cuda, f, O, I, w, s)
		if _, live := cuda.cudaAllocationCountsForTest(); live != liveAllocsBefore+2 {
			t.Fatalf("%s: resident upload holds %d pooled allocations, want 2", f.name, live-liveAllocsBefore)
		}
		if _, err := h.Gemv(x); err != nil {
			t.Fatalf("%s: GEMV before Release: %v", f.name, err)
		}
		mustReleaseResident(t, f.name, h)
		if _, live := cuda.cudaAllocationCountsForTest(); live != liveAllocsBefore {
			t.Fatalf("%s: Release left %d pooled allocations checked out", f.name, live-liveAllocsBefore)
		}
		before := readResidentCounters(cuda)
		if d := before.since(start); d.live != 0 {
			t.Fatalf("%s: live resident ids after Release changed by %d", f.name, int64(d.live))
		}
		y, err := h.Gemv(x)
		if y != nil || !errors.Is(err, ErrRoutedExpertResidentReleased) {
			t.Fatalf("%s: GEMV after Release len(y)=%d err=%v, want ErrRoutedExpertResidentReleased", f.name, len(y), err)
		}
		var opErr *CUDAOpError
		if !errors.As(err, &opErr) || opErr.Site != f.site {
			t.Fatalf("%s: use-after-release err = %T %v, want *CUDAOpError at %s", f.name, err, err, f.site)
		}
		if err := h.Release(); !errors.Is(err, ErrRoutedExpertResidentReleased) {
			t.Fatalf("%s: second Release err=%v, want ErrRoutedExpertResidentReleased", f.name, err)
		}
		if d := readResidentCounters(cuda).since(before); d != (residentCounters{}) {
			t.Fatalf("%s: released-handle calls moved %+v", f.name, d)
		}
	}
	if hl := cuda.faultLatch.Health(); hl != DeviceHealthy {
		t.Fatalf("use-after-release poisoned the device session: health=%v", hl)
	}
}

// TestCUDARoutedExpertResidentRefusedUnderGraphCapture opens a real CUDA graph capture: upload,
// GEMV and Release are each refused with a Go-side *CUDAOpError at the resident site and no
// device work, the refused Release leaves the handle resident, and after the capture is
// aborted the same handle serves bit-exact and releases.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertResidentRefusedUnderGraphCapture(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = 40, 96
	type prepared struct {
		f       residentFormat
		h       RoutedExpertResidentWeight
		w, s    []byte
		x, want []float32
	}
	var ps []prepared
	for _, f := range residentFormats {
		w, s, x := f.gen(101, rotating(O, I))
		want, err := f.oracle(O, I, w, s, x)
		if err != nil {
			t.Fatal(err)
		}
		ps = append(ps, prepared{f, mustUploadResident(t, cuda, f, O, I, w, s), w, s, x, want})
	}
	defer withGraphEnabled(true)()
	cuda.GraphReset()
	t.Cleanup(func() { cuda.TrimLarge(1 << 20) })
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if !cuda.GraphBegin() {
		t.Fatal("GraphBegin did not open a CUDA graph capture")
	}
	open := true
	defer func() {
		if open {
			cuda.GraphAbort()
		}
	}()
	refusedAt := func(label, site string, err error) {
		t.Helper()
		var opErr *CUDAOpError
		var launchErr *CUDALaunchError
		if !errors.As(err, &opErr) || opErr.Site != site || opErr.Code != 0 || errors.As(err, &launchErr) {
			t.Fatalf("%s under capture: err = %T %v, want a Go-side *CUDAOpError refusal at %s", label, err, err, site)
		}
	}
	for _, p := range ps {
		before := readResidentCounters(cuda)
		h, err := p.f.upload(cuda, O, I, p.w, p.s)
		refusedAt(p.f.name+" upload", p.f.site, err)
		if h != nil {
			t.Fatalf("%s: refused upload returned a handle", p.f.name)
		}
		y, err := p.h.Gemv(p.x)
		refusedAt(p.f.name+" GEMV", p.f.site, err)
		if y != nil {
			t.Fatalf("%s: refused GEMV returned len(y)=%d", p.f.name, len(y))
		}
		refusedAt(p.f.name+" Release", p.f.site, p.h.Release())
		if d := readResidentCounters(cuda).since(before); d.uploads != 0 || d.live != 0 || d.launches != 0 {
			t.Fatalf("%s: capture refusals moved %+v", p.f.name, d)
		}
	}
	cuda.GraphAbort()
	open = false
	if hl := cuda.faultLatch.Health(); hl != DeviceHealthy {
		t.Fatalf("capture refusal poisoned the device session: health=%v", hl)
	}
	for _, p := range ps {
		got, err := p.h.Gemv(p.x)
		if err != nil {
			t.Fatalf("%s after GraphAbort: %v", p.f.name, err)
		}
		requireRoutedExpertBits(t, p.f.name+"/after-capture", got, p.want)
		mustReleaseResident(t, p.f.name, p.h)
	}
}

// requireResidentPoisonedOnce asserts the latch moved healthy->poisoned exactly once by class at site.
func requireResidentPoisonedOnce(t *testing.T, label string, latch *DeviceFaultLatch, class DeviceFaultClass, site string, code int) {
	t.Helper()
	snap := latch.Snapshot()
	if snap.Health != DevicePoisoned || snap.Faults != 1 || snap.LastClass != class || snap.LastSite != site || snap.LastCode != code {
		t.Fatalf("%s: latch health=%v faults=%d class=%v site=%q code=%d, want poisoned once by %v at %s code %d",
			label, snap.Health, snap.Faults, snap.LastClass, snap.LastSite, snap.LastCode, class, site, code)
	}
	if len(snap.Transitions) != 1 || snap.Transitions[0].From != DeviceHealthy || snap.Transitions[0].To != DevicePoisoned {
		t.Fatalf("%s: latch transitions = %d, want exactly healthy->poisoned", label, len(snap.Transitions))
	}
}

// TestCUDARoutedExpertResidentDeviceFaultsFailClosed drives the one-shot C fault seam through
// the resident path. An upload allocation fault returns *DeviceAllocError (weights class), no
// handle and no retained allocation. A GEMV allocation or launch fault returns its typed error
// with a nil y. Each poisons the session once at the resident site; the next call is refused
// before device work; after a validated reconstruction the pre-fault handle is refused as
// ErrRoutedExpertResidentStale yet still releases, and a fresh upload serves bit-exact.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertResidentDeviceFaultsFailClosed(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = 40, 96
	t.Cleanup(func() { routedExpertInjectFaultForTest(cudaRoutedExpertFaultNone) })
	prodLatch := cuda.faultLatch
	t.Cleanup(func() { cuda.faultLatch = prodLatch })
	for _, f := range residentFormats {
		w, s, x := f.gen(111, rotating(O, I))
		want, err := f.oracle(O, I, w, s, x)
		if err != nil {
			t.Fatal(err)
		}

		label := f.name + "/upload-allocation"
		cuda.faultLatch = NewDeviceFaultLatch("cuda", cudaFaultReconstructBudget)
		_, liveAllocsBefore := cuda.cudaAllocationCountsForTest()
		before := readResidentCounters(cuda)
		routedExpertInjectFaultForTest(cudaRoutedExpertFaultAlloc)
		h, err := f.upload(cuda, O, I, w, s)
		var allocErr *DeviceAllocError
		if h != nil || !errors.As(err, &allocErr) || allocErr.Site != f.site || allocErr.Class != MemoryWeights {
			t.Fatalf("%s: handle=%v err = %T %v, want *DeviceAllocError (weights) at %s", label, h, err, err, f.site)
		}
		if _, live := cuda.cudaAllocationCountsForTest(); live != liveAllocsBefore {
			t.Fatalf("%s: failed upload retained %d pooled allocations", label, live-liveAllocsBefore)
		}
		requireResidentPoisonedOnce(t, label, cuda.faultLatch, DeviceFaultAllocation, f.site, 0)
		var faultErr *DeviceFaultError
		if h, err = f.upload(cuda, O, I, w, s); h != nil || !errors.As(err, &faultErr) || faultErr.Site != f.site {
			t.Fatalf("%s: post-fault upload err = %T %v, want *DeviceFaultError at %s", label, err, err, f.site)
		}
		if d := readResidentCounters(cuda).since(before); d != (residentCounters{}) {
			t.Fatalf("%s: faulted and refused uploads moved %+v", label, d)
		}
		if err := cuda.faultLatch.Reconstruct(nil, nil); err != nil {
			t.Fatalf("%s: latch reconstruct: %v", label, err)
		}
		h = mustUploadResident(t, cuda, f, O, I, w, s)
		mustReleaseResident(t, label, h)

		for _, fault := range []struct {
			name  string
			kind  int
			class DeviceFaultClass
		}{
			{"gemv-allocation", cudaRoutedExpertFaultAlloc, DeviceFaultAllocation},
			{"gemv-launch", cudaRoutedExpertFaultLaunch, DeviceFaultExecution},
		} {
			label := f.name + "/" + fault.name
			cuda.faultLatch = NewDeviceFaultLatch("cuda", cudaFaultReconstructBudget)
			h := mustUploadResident(t, cuda, f, O, I, w, s)
			before := readResidentCounters(cuda)
			routedExpertInjectFaultForTest(fault.kind)
			y, err := h.Gemv(x)
			if y != nil {
				t.Fatalf("%s: faulted GEMV returned len(y)=%d", label, len(y))
			}
			wantCode := 0
			if fault.kind == cudaRoutedExpertFaultAlloc {
				var allocErr *DeviceAllocError
				if !errors.As(err, &allocErr) || allocErr.Site != f.site || allocErr.Class != MemoryScratchpad {
					t.Fatalf("%s: err = %T %v, want *DeviceAllocError (scratchpad) at %s", label, err, err, f.site)
				}
			} else {
				var launchErr *CUDALaunchError
				if !errors.As(err, &launchErr) || launchErr.Site != f.site || launchErr.Code <= 0 {
					t.Fatalf("%s: err = %T %v, want *CUDALaunchError with a positive cudaError at %s", label, err, err, f.site)
				}
				wantCode = launchErr.Code
			}
			requireResidentPoisonedOnce(t, label, cuda.faultLatch, fault.class, f.site, wantCode)
			var faultErr *DeviceFaultError
			if y, err = h.Gemv(x); y != nil || !errors.As(err, &faultErr) || faultErr.Site != f.site {
				t.Fatalf("%s: post-fault GEMV err = %T %v, want *DeviceFaultError at %s", label, err, err, f.site)
			}
			if err := cuda.faultLatch.Reconstruct(nil, nil); err != nil {
				t.Fatalf("%s: latch reconstruct: %v", label, err)
			}
			if y, err = h.Gemv(x); y != nil || !errors.Is(err, ErrRoutedExpertResidentStale) {
				t.Fatalf("%s: pre-fault handle after reconstruction len(y)=%d err=%v, want ErrRoutedExpertResidentStale", label, len(y), err)
			}
			if d := readResidentCounters(cuda).since(before); d.launches != 0 || d.uploads != 0 {
				t.Fatalf("%s: faulted, refused and stale GEMVs moved %+v, want no launch or upload", label, d)
			}
			mustReleaseResident(t, label+"/stale", h)
			fresh := mustUploadResident(t, cuda, f, O, I, w, s)
			got, err := fresh.Gemv(x)
			if err != nil {
				t.Fatalf("%s: fresh upload after reconstruction: %v", label, err)
			}
			requireRoutedExpertBits(t, label+"/recovered", got, want)
			mustReleaseResident(t, label+"/fresh", fresh)
		}
	}
}

// BenchmarkCUDARoutedExpertResidentGemv is BenchmarkCUDARoutedExpertGemv over a resident
// weight uploaded once outside the timer: H2D carries only x, so ns/op approaches the kernel.
//
//	go test -tags cuda ./internal/compute -run '^$' -bench 'CUDARoutedExpert(Resident)?Gemv' -benchtime 50x
func BenchmarkCUDARoutedExpertResidentGemv(b *testing.B) {
	cuda := routedExpertCUDAOrSkip(b)
	for _, bc := range []struct {
		name   string
		format int
		O, I   int
	}{
		{"mxfp4_O2048_I4096", cudaRoutedExpertMXFP4, 2048, 4096},
		{"fp8_O4096_I4096", cudaRoutedExpertFP8Block32, 4096, 4096},
	} {
		b.Run(bc.name, func(b *testing.B) {
			var w, s []byte
			var x []float32
			if bc.format == cudaRoutedExpertMXFP4 {
				w, s, x = routedExpertMXFP4Case(41, rotating(bc.O, bc.I))
			} else {
				w, s, x = routedExpertFP8Case(42, rotating(bc.O, bc.I))
			}
			h, err := cuda.uploadRoutedExpertResident(bc.format, bc.O, bc.I, w, s)
			if err != nil {
				b.Fatal(err)
			}
			defer h.Release()
			var tm cudaRoutedExpertTiming
			if _, err := h.gemv(x, &tm); err != nil {
				b.Fatal(err) // warm the pooled x/y buffers outside the timed loop
			}
			var h2d, kern, d2h float64
			b.SetBytes(int64(len(w) + len(s)))
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				if _, err := h.gemv(x, &tm); err != nil {
					b.Fatal(err)
				}
				h2d += tm.H2DMillis
				kern += tm.KernelMillis
				d2h += tm.D2HMillis
			}
			n := float64(b.N)
			b.ReportMetric(1e3*h2d/n, "h2d_us/call")
			b.ReportMetric(1e3*kern/n, "kernel_us/call")
			b.ReportMetric(1e3*d2h/n, "d2h_us/call")
		})
	}
}
