//go:build cuda && cgo

// Routed-expert CUDA GEMV witnesses. On a CUDA host, after build_cuda.sh build:
//
//	FAK_CUDA_ROUTED_EXPERT_REQUIRED=1 go test -tags cuda ./internal/compute -run CUDARoutedExpert

package compute

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"
)

const cudaRoutedExpertRequiredEnv = "FAK_CUDA_ROUTED_EXPERT_REQUIRED"

func routedExpertCUDAOrSkip(tb testing.TB) *cudaBackend {
	tb.Helper()
	be, ok := Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		if os.Getenv(cudaRoutedExpertRequiredEnv) == "1" {
			tb.Fatalf("%s=1: real CUDA backend required", cudaRoutedExpertRequiredEnv)
		}
		tb.Skip("CUDA backend is unavailable")
	}
	cuda, ok := be.(*cudaBackend)
	if !ok {
		tb.Fatalf("registered CUDA backend has type %T", be)
	}
	return cuda
}

// routedExpertBitStats classifies the oracle outputs a case compared, so the suite can
// assert it actually crossed the subnormal and overflow boundaries it claims to cover.
type routedExpertBitStats struct {
	normal, subnormal, zero, inf, nan int
}

func (s *routedExpertBitStats) add(o routedExpertBitStats) {
	s.normal += o.normal
	s.subnormal += o.subnormal
	s.zero += o.zero
	s.inf += o.inf
	s.nan += o.nan
}

// requireRoutedExpertBits asserts got == want bit-for-bit, except that a NaN must meet a
// NaN: IEEE 754 leaves the payload of a generated NaN unspecified (x86 SSE yields
// 0xffc00000, CUDA 0x7fffffff), so NaN is compared as a class.
func requireRoutedExpertBits(tb testing.TB, label string, got, want []float32) routedExpertBitStats {
	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("%s: len(got)=%d want %d", label, len(got), len(want))
	}
	var st routedExpertBitStats
	for k := range want {
		g, w := math.Float32bits(got[k]), math.Float32bits(want[k])
		wantNaN := want[k] != want[k]
		if wantNaN {
			if got[k] == got[k] {
				tb.Fatalf("%s: y[%d] bits %#08x, want NaN (%#08x)", label, k, g, w)
			}
			st.nan++
			continue
		}
		if g != w {
			tb.Fatalf("%s: y[%d] bits %#08x != oracle %#08x (%g vs %g)", label, k, g, w, got[k], want[k])
		}
		switch exp := (w >> 23) & 0xff; {
		case exp == 0xff:
			st.inf++
		case w&0x7fffffff == 0:
			st.zero++
		case exp == 0:
			st.subnormal++
		default:
			st.normal++
		}
	}
	return st
}

// Scale bands. A row group draws its E8M0 bytes from one band: ordinary (117..137), tiny
// (0..4: the subnormal 2^-127 through 2^-123, driving subnormal products and sums), huge
// (250..254: overflow to Inf and Inf-Inf NaN), or wide (64..190, finite). bandRotate
// cycles the four across row groups.
const (
	bandOrdinary = iota
	bandTiny
	bandHuge
	bandWide
	bandRotate = -1
)

func routedExpertScale(rng *rand.Rand, band int) byte {
	switch band & 3 {
	case bandOrdinary:
		return byte(117 + rng.Intn(21))
	case bandTiny:
		return byte(rng.Intn(5))
	case bandHuge:
		return byte(250 + rng.Intn(5))
	default:
		return byte(64 + rng.Intn(127))
	}
}

// routedExpertCase is one deterministic GEMV fixture: shape, scale band (bandRotate or one
// fixed band) and the binary-exponent range of x.
type routedExpertCase struct {
	O, I     int
	band     int
	xLo, xHi int
}

func (c routedExpertCase) bandFor(group int, seed int64) int {
	if c.band == bandRotate {
		return group + int(seed)
	}
	return c.band
}

// rotating is the default fixture: rotating scale bands, x over 2^-30..2^12.
func rotating(O, I int) routedExpertCase {
	return routedExpertCase{O: O, I: I, band: bandRotate, xLo: -30, xHi: 12}
}

// subnormalHeavy keeps every scale tiny and x small, so most products, partial sums and
// outputs are subnormal: a flush-to-zero kernel would fail the bit comparison.
func subnormalHeavy(O, I int) routedExpertCase {
	return routedExpertCase{O: O, I: I, band: bandTiny, xLo: -30, xHi: -16}
}

// overflowing puts huge scales on one- and two-term rows, so outputs reach +-Inf without
// always cancelling into Inf-Inf NaN.
func overflowing(O, I int) routedExpertCase {
	return routedExpertCase{O: O, I: I, band: bandHuge, xLo: 0, xHi: 12}
}

// routedExpertX mixes signs, magnitudes 2^lo..2^hi, exact +-0 and subnormal entries.
func routedExpertX(rng *rand.Rand, n, lo, hi int) []float32 {
	x := make([]float32, n)
	for i := range x {
		switch r := rng.Intn(40); {
		case r == 0:
			x[i] = 0
		case r == 1:
			x[i] = float32(math.Copysign(0, -1))
		case r == 2:
			x[i] = math.Float32frombits(uint32(1 + rng.Intn(1<<22)))
		default:
			v := float32(math.Ldexp(1+rng.Float64(), lo+rng.Intn(hi-lo+1)))
			if rng.Intn(2) == 0 {
				v = -v
			}
			x[i] = v
		}
	}
	return x
}

var routedExpertE4M3Finite = func() []byte {
	var out []byte
	for v := 0; v < 256; v++ {
		if !e4m3NaNByte(byte(v)) {
			out = append(out, byte(v))
		}
	}
	return out
}()

func routedExpertFP8Case(seed int64, c routedExpertCase) (weight, scales []byte, x []float32) {
	O, I := c.O, c.I
	rng := rand.New(rand.NewSource(seed))
	weight = make([]byte, O*I)
	for k := range weight {
		weight[k] = routedExpertE4M3Finite[rng.Intn(len(routedExpertE4M3Finite))]
	}
	sO, sI := fp8Block32ScaleShape(O, I)
	scales = make([]byte, sO*sI)
	for rt := 0; rt < sO; rt++ {
		for j := 0; j < sI; j++ {
			scales[rt*sI+j] = routedExpertScale(rng, c.bandFor(rt, seed))
		}
	}
	return weight, scales, routedExpertX(rng, I, c.xLo, c.xHi)
}

func routedExpertMXFP4Case(seed int64, c routedExpertCase) (weight, scales []byte, x []float32) {
	O, I := c.O, c.I
	rng := rand.New(rand.NewSource(seed))
	weight = make([]byte, O*I/2)
	for k := range weight {
		weight[k] = byte(rng.Intn(256))
	}
	scaleCols := (I + FP8Block32Dim - 1) / FP8Block32Dim
	scales = make([]byte, O*scaleCols)
	for o := 0; o < O; o++ {
		for j := 0; j < scaleCols; j++ {
			scales[o*scaleCols+j] = routedExpertScale(rng, c.bandFor(o/FP8Block32Dim, seed))
		}
	}
	return weight, scales, routedExpertX(rng, I, c.xLo, c.xHi)
}

// requireBoundaryCoverage fails a suite whose cases never reached a boundary class, so the
// bit-exact claim cannot pass vacuously on all-normal data.
func requireBoundaryCoverage(t *testing.T, label string, st routedExpertBitStats) {
	t.Helper()
	t.Logf("%s output classes: normal=%d subnormal=%d zero=%d inf=%d nan=%d", label, st.normal, st.subnormal, st.zero, st.inf, st.nan)
	if st.normal == 0 || st.subnormal == 0 || st.inf == 0 || st.nan == 0 {
		t.Fatalf("%s: boundary coverage incomplete: %+v", label, st)
	}
}

// fak-test:runtime integration est=4s lane=optin
func TestCUDARoutedExpertFP8Block32BitExact(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	cases := []routedExpertCase{
		rotating(1, 1), rotating(1, 2), rotating(3, 5), rotating(31, 33), rotating(32, 32),
		rotating(33, 127), rotating(40, 96), rotating(65, 129), rotating(97, 300),
		rotating(130, 257), rotating(1024, 4096), rotating(2048, 1000), rotating(4096, 4096),
		subnormalHeavy(33, 257), subnormalHeavy(256, 4096), overflowing(64, 1), overflowing(64, 2),
	}
	var total routedExpertBitStats
	seen := make([]bool, 256)
	for idx, c := range cases {
		O, I := c.O, c.I
		weight, scales, x := routedExpertFP8Case(int64(idx+1), c)
		for _, b := range weight {
			seen[b] = true
		}
		want, err := GemvFP8Block32E8M0("oracle", O, I, weight, scales, x)
		if err != nil {
			t.Fatalf("[%dx%d] oracle: %v", O, I, err)
		}
		before := cudaRoutedExpertLaunches()
		got, err := cuda.GemvExpertFP8Block32E8M0(O, I, weight, scales, x)
		if err != nil {
			t.Fatalf("[%dx%d] device: %v", O, I, err)
		}
		if n := cudaRoutedExpertLaunches() - before; n != 1 {
			t.Fatalf("[%dx%d] device launches = %d, want 1", O, I, n)
		}
		total.add(requireRoutedExpertBits(t, fmt.Sprintf("fp8[%dx%d]", O, I), got, want))
	}
	for _, b := range routedExpertE4M3Finite {
		if !seen[b] {
			t.Fatalf("E4M3 byte %#02x never exercised", b)
		}
	}
	requireBoundaryCoverage(t, "fp8", total)
}

// fak-test:runtime integration est=3s lane=optin
func TestCUDARoutedExpertMXFP4BitExact(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	cases := []routedExpertCase{
		rotating(1, 2), rotating(1, 4), rotating(3, 6), rotating(31, 34), rotating(32, 64),
		rotating(33, 62), rotating(64, 256), rotating(65, 258), rotating(100, 1000),
		rotating(7, 514), rotating(512, 2048), rotating(2048, 4096),
		subnormalHeavy(33, 258), subnormalHeavy(256, 4096), overflowing(64, 2),
	}
	var total routedExpertBitStats
	var nibbles [16]bool
	for idx, c := range cases {
		O, I := c.O, c.I
		weight, scales, x := routedExpertMXFP4Case(int64(100+idx), c)
		for _, b := range weight {
			nibbles[b&0x0f], nibbles[b>>4] = true, true
		}
		want, err := GemvMXFP4E8M0("oracle", O, I, weight, scales, x)
		if err != nil {
			t.Fatalf("[%dx%d] oracle: %v", O, I, err)
		}
		before := cudaRoutedExpertLaunches()
		got, err := cuda.GemvExpertMXFP4E8M0(O, I, weight, scales, x)
		if err != nil {
			t.Fatalf("[%dx%d] device: %v", O, I, err)
		}
		if n := cudaRoutedExpertLaunches() - before; n != 1 {
			t.Fatalf("[%dx%d] device launches = %d, want 1", O, I, n)
		}
		total.add(requireRoutedExpertBits(t, fmt.Sprintf("mxfp4[%dx%d]", O, I), got, want))
	}
	for nib, ok := range nibbles {
		if !ok {
			t.Fatalf("E2M1 nibble %#x never exercised", nib)
		}
	}
	requireBoundaryCoverage(t, "mxfp4", total)
}

// TestCUDARoutedExpertE8M0EveryScaleByte drives all 255 non-NaN E8M0 bytes through
// distinct FP8 tiles and MXFP4 groups of ONE launch, witnessing per-tile scale indexing.
// Random weights overflow near byte 254 (Inf/NaN compare as a class there), so
// TestCUDARoutedExpertE8M0EveryByteFiniteBitExact holds each byte bit-for-bit.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertE8M0EveryScaleByte(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	rng := rand.New(rand.NewSource(255))
	const O, I = 255 * FP8Block32Dim, FP8Block32Dim // one row tile per scale byte
	fp8 := make([]byte, O*I)
	for k := range fp8 {
		fp8[k] = routedExpertE4M3Finite[rng.Intn(len(routedExpertE4M3Finite))]
	}
	fp8Scales := make([]byte, 255)
	mxScales := make([]byte, O)
	for b := range fp8Scales {
		fp8Scales[b] = byte(b)
	}
	for o := range mxScales {
		mxScales[o] = byte(o / FP8Block32Dim)
	}
	mx := make([]byte, O*I/2)
	for k := range mx {
		mx[k] = byte(rng.Intn(256))
	}
	x := routedExpertX(rng, I, -30, 12)

	wantFP8, err := GemvFP8Block32E8M0("oracle", O, I, fp8, fp8Scales, x)
	if err != nil {
		t.Fatal(err)
	}
	gotFP8, err := cuda.GemvExpertFP8Block32E8M0(O, I, fp8, fp8Scales, x)
	if err != nil {
		t.Fatal(err)
	}
	requireRoutedExpertBits(t, "fp8-every-scale", gotFP8, wantFP8)
	wantMX, err := GemvMXFP4E8M0("oracle", O, I, mx, mxScales, x)
	if err != nil {
		t.Fatal(err)
	}
	gotMX, err := cuda.GemvExpertMXFP4E8M0(O, I, mx, mxScales, x)
	if err != nil {
		t.Fatal(err)
	}
	requireRoutedExpertBits(t, "mxfp4-every-scale", gotMX, wantMX)
}

// TestCUDARoutedExpertE8M0EveryByteFiniteBitExact holds every E8M0 byte 0..254 to the
// oracle bit-for-bit on finite, normal products (e8m0FiniteByteCase), one call per byte
// and format, each with a nonzero output: a decode that clamps 254 to 253, reads it as
// 2^126 or Inf, or flushes the byte-0 subnormal fails here.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertE8M0EveryByteFiniteBitExact(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = e8m0FiniteByteO, e8m0FiniteByteI
	for v := 0; v <= 254; v++ {
		b := byte(v)
		for _, mxfp4 := range []bool{false, true} {
			weight, scales, x := e8m0FiniteByteCase(t, b, mxfp4)
			label := fmt.Sprintf("fp8-e8m0[%#02x]", b)
			var want, got []float32
			var oerr, derr error
			before := cudaRoutedExpertLaunches()
			if mxfp4 {
				label = fmt.Sprintf("mxfp4-e8m0[%#02x]", b)
				want, oerr = GemvMXFP4E8M0("oracle", O, I, weight, scales, x)
				got, derr = cuda.GemvExpertMXFP4E8M0(O, I, weight, scales, x)
			} else {
				want, oerr = GemvFP8Block32E8M0("oracle", O, I, weight, scales, x)
				got, derr = cuda.GemvExpertFP8Block32E8M0(O, I, weight, scales, x)
			}
			if oerr != nil || derr != nil {
				t.Fatalf("%s: oracle err=%v device err=%v", label, oerr, derr)
			}
			if n := cudaRoutedExpertLaunches() - before; n != 1 {
				t.Fatalf("%s: device launches = %d, want 1", label, n)
			}
			st := requireRoutedExpertBits(t, label, got, want)
			if st.inf != 0 || st.nan != 0 || st.normal == 0 {
				t.Fatalf("%s: outputs left the finite bit-exact regime: normal=%d inf=%d nan=%d", label, st.normal, st.inf, st.nan)
			}
		}
	}
}

// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertDispatchReachesDevice(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = 96, 320
	fw, fs, x := routedExpertFP8Case(11, rotating(O, I))
	before := cudaRoutedExpertLaunches()
	got, err := GemvFP8Block32ExpertDevice("v41.w1", cuda, O, I, fw, fs, x)
	if errors.Is(err, ErrExpertDeviceKernelAbsent) || err != nil {
		t.Fatalf("FP8 dispatch on cuda: %v", err)
	}
	if n := cudaRoutedExpertLaunches() - before; n != 1 {
		t.Fatalf("FP8 dispatch ran %d device launches, want 1", n)
	}
	want, _ := GemvFP8Block32E8M0("oracle", O, I, fw, fs, x)
	requireRoutedExpertBits(t, "fp8-dispatch", got, want)

	mw, ms, mx := routedExpertMXFP4Case(12, rotating(O, I))
	before = cudaRoutedExpertLaunches()
	got, err = GemvMXFP4ExpertDevice("v41.w2", cuda, O, I, mw, ms, mx)
	if errors.Is(err, ErrExpertDeviceKernelAbsent) || err != nil {
		t.Fatalf("MXFP4 dispatch on cuda: %v", err)
	}
	if n := cudaRoutedExpertLaunches() - before; n != 1 {
		t.Fatalf("MXFP4 dispatch ran %d device launches, want 1", n)
	}
	want, _ = GemvMXFP4E8M0("oracle", O, I, mw, ms, mx)
	requireRoutedExpertBits(t, "mxfp4-dispatch", got, want)
}

// routedExpertCall is one well-formed routed-expert call of one format with its oracle
// output, for the fail-closed tests.
type routedExpertCall struct {
	site string
	call func() ([]float32, error)
	want []float32
}

func routedExpertCalls(t *testing.T, cuda *cudaBackend, seed int64) []routedExpertCall {
	t.Helper()
	const O, I = 40, 96
	fw, fs, x := routedExpertFP8Case(seed, rotating(O, I))
	mw, ms, mx := routedExpertMXFP4Case(seed+1, rotating(O, I))
	fwant, err := GemvFP8Block32E8M0("oracle", O, I, fw, fs, x)
	if err != nil {
		t.Fatal(err)
	}
	mwant, err := GemvMXFP4E8M0("oracle", O, I, mw, ms, mx)
	if err != nil {
		t.Fatal(err)
	}
	return []routedExpertCall{
		{"routed-expert-fp8-block32", func() ([]float32, error) { return cuda.GemvExpertFP8Block32E8M0(O, I, fw, fs, x) }, fwant},
		{"routed-expert-mxfp4", func() ([]float32, error) { return cuda.GemvExpertMXFP4E8M0(O, I, mw, ms, mx) }, mwant},
	}
}

// TestCUDARoutedExpertRefusedUnderGraphCapture opens a real CUDA graph capture and holds
// both formats to a typed CUDAOpError refusal with a nil y and no launch: the per-call
// upload and read-back are not capturable. The refusal is fak refusing on purpose, so the
// session stays healthy and serves bit-exact again once the capture is aborted.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertRefusedUnderGraphCapture(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	calls := routedExpertCalls(t, cuda, 51)
	defer withGraphEnabled(true)()
	cuda.GraphReset()
	// GraphBegin prewarms every pooled bucket, including the large ones earlier tests left.
	t.Cleanup(func() { cuda.TrimLarge(1 << 20) })
	// A global-mode capture must end on the OS thread that began it.
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
	if !cuda.IsCapturing() {
		t.Fatal("GraphBegin returned true but the backend is not capturing")
	}
	for _, c := range calls {
		before := cudaRoutedExpertLaunches()
		y, err := c.call()
		launched := cudaRoutedExpertLaunches() - before
		var opErr *CUDAOpError
		var launchErr *CUDALaunchError
		// Code 0 pins the Go guard; the C ABI's own capture refusal would carry status -2.
		if !errors.As(err, &opErr) || opErr.Site != c.site || opErr.Code != 0 || errors.As(err, &launchErr) {
			t.Fatalf("%s under capture: err = %T %v, want a Go-side *CUDAOpError refusal at %s", c.site, err, err, c.site)
		}
		if y != nil || launched != 0 {
			t.Fatalf("%s under capture: returned len(y)=%d and launched %d kernels", c.site, len(y), launched)
		}
	}
	cuda.GraphAbort()
	open = false
	if cuda.IsCapturing() {
		t.Fatal("GraphAbort left the capture open")
	}
	if h := cuda.faultLatch.Health(); h != DeviceHealthy {
		t.Fatalf("capture refusal poisoned the device session: health=%v", h)
	}
	for _, c := range calls {
		before := cudaRoutedExpertLaunches()
		y, err := c.call()
		if err != nil {
			t.Fatalf("%s after GraphAbort: %v", c.site, err)
		}
		if n := cudaRoutedExpertLaunches() - before; n != 1 {
			t.Fatalf("%s after GraphAbort: device launches = %d, want 1", c.site, n)
		}
		requireRoutedExpertBits(t, c.site+"/after-capture", y, c.want)
	}
}

// TestCUDARoutedExpertDeviceFaultsFailClosed injects a one-shot allocation failure and a
// rejected launch at the C ABI and holds both formats to the binding's fail-closed
// contract: the typed error, a nil y, no completed launch, the session latch moved
// healthy->poisoned at the routed-expert site, the next call refused before any device
// work, and service restored only by a validated reconstruction.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertDeviceFaultsFailClosed(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	calls := routedExpertCalls(t, cuda, 61)
	t.Cleanup(func() { routedExpertInjectFaultForTest(cudaRoutedExpertFaultNone) })
	// Isolate the shared backend's session latch: each case poisons and reconstructs it,
	// and the production latch must not be left poisoned or spent for later tests.
	prodLatch := cuda.faultLatch
	t.Cleanup(func() { cuda.faultLatch = prodLatch })
	for _, c := range calls {
		for _, fault := range []struct {
			name  string
			kind  int
			class DeviceFaultClass
		}{
			{"allocation", cudaRoutedExpertFaultAlloc, DeviceFaultAllocation},
			{"launch", cudaRoutedExpertFaultLaunch, DeviceFaultExecution},
		} {
			label := c.site + "/" + fault.name
			cuda.faultLatch = NewDeviceFaultLatch("cuda", cudaFaultReconstructBudget)
			before := cudaRoutedExpertLaunches()
			routedExpertInjectFaultForTest(fault.kind)
			y, err := c.call()
			if y != nil {
				t.Fatalf("%s: faulted call returned len(y)=%d", label, len(y))
			}
			wantCode := 0
			if fault.kind == cudaRoutedExpertFaultAlloc {
				var allocErr *DeviceAllocError
				if !errors.As(err, &allocErr) || allocErr.Site != c.site {
					t.Fatalf("%s: err = %T %v, want *DeviceAllocError at %s", label, err, err, c.site)
				}
			} else {
				var launchErr *CUDALaunchError
				if !errors.As(err, &launchErr) || launchErr.Site != c.site || launchErr.Code <= 0 {
					t.Fatalf("%s: err = %T %v, want *CUDALaunchError with a positive cudaError at %s", label, err, err, c.site)
				}
				wantCode = launchErr.Code
			}
			snap := cuda.faultLatch.Snapshot()
			if snap.Health != DevicePoisoned || snap.Faults != 1 || snap.LastClass != fault.class ||
				snap.LastSite != c.site || snap.LastCode != wantCode {
				t.Fatalf("%s: latch health=%v faults=%d class=%v site=%q code=%d, want poisoned once by %v at %s code %d",
					label, snap.Health, snap.Faults, snap.LastClass, snap.LastSite, snap.LastCode, fault.class, c.site, wantCode)
			}
			if len(snap.Transitions) != 1 || snap.Transitions[0].From != DeviceHealthy || snap.Transitions[0].To != DevicePoisoned {
				t.Fatalf("%s: latch transitions = %d, want exactly healthy->poisoned", label, len(snap.Transitions))
			}
			y, err = c.call()
			var faultErr *DeviceFaultError
			if y != nil || !errors.As(err, &faultErr) || faultErr.Site != c.site {
				t.Fatalf("%s: post-fault call err = %T %v len(y)=%d, want *DeviceFaultError at %s", label, err, err, len(y), c.site)
			}
			if n := cudaRoutedExpertLaunches() - before; n != 0 {
				t.Fatalf("%s: faulted and refused calls completed %d launches, want 0", label, n)
			}
			if err := cuda.faultLatch.Reconstruct(nil, nil); err != nil {
				t.Fatalf("%s: latch reconstruct: %v", label, err)
			}
			y, err = c.call()
			if err != nil {
				t.Fatalf("%s: call after reconstruction: %v", label, err)
			}
			if n := cudaRoutedExpertLaunches() - before; n != 1 {
				t.Fatalf("%s: launches after reconstruction = %d, want 1 (fault seam must be one-shot)", label, n)
			}
			requireRoutedExpertBits(t, label+"/recovered", y, c.want)
		}
	}
}

// TestCUDARoutedExpertRefusalParity holds the device to the oracle's acceptance set: every
// input the oracle refuses is refused by the device with a nil y, no kernel launch, and no
// session poisoning; the one well-formed control is accepted by both.
//
// fak-test:runtime integration est=1s lane=optin
func TestCUDARoutedExpertRefusalParity(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = 40, 96
	fw, fs, x := routedExpertFP8Case(21, rotating(O, I))
	mw, ms, _ := routedExpertMXFP4Case(22, rotating(O, I))
	with := func(b []byte, k int, v byte) []byte {
		c := append([]byte(nil), b...)
		c[k] = v
		return c
	}
	type gemv func(O, I int, w, s []byte, x []float32) ([]float32, error)
	type tc struct {
		name   string
		O, I   int
		w, s   []byte
		x      []float32
		accept bool
	}
	run := func(t *testing.T, label string, oracle, device gemv, cases []tc) {
		for _, c := range cases {
			_, oerr := oracle(c.O, c.I, c.w, c.s, c.x)
			before := cudaRoutedExpertLaunches()
			y, derr := device(c.O, c.I, c.w, c.s, c.x)
			launched := cudaRoutedExpertLaunches() - before
			if (oerr == nil) != c.accept || (derr == nil) != c.accept {
				t.Fatalf("%s/%s: oracle err=%v device err=%v, want accept=%v", label, c.name, oerr, derr, c.accept)
			}
			if !c.accept && (y != nil || launched != 0) {
				t.Fatalf("%s/%s: refused call returned len(y)=%d and launched %d kernels", label, c.name, len(y), launched)
			}
			if c.accept && (len(y) != c.O || launched != 1) {
				t.Fatalf("%s/%s: accepted call returned len(y)=%d, launched %d", label, c.name, len(y), launched)
			}
		}
	}
	fp8Oracle := func(O, I int, w, s []byte, x []float32) ([]float32, error) {
		return GemvFP8Block32E8M0("oracle", O, I, w, s, x)
	}
	run(t, "fp8", fp8Oracle, cuda.GemvExpertFP8Block32E8M0, []tc{
		{"ok", O, I, fw, fs, x, true},
		{"zero O", 0, I, fw, fs, x, false},
		{"negative I", O, -I, fw, fs, x, false},
		{"weight short", O, I, fw[:len(fw)-1], fs, x, false},
		{"weight long", O, I, append(append([]byte(nil), fw...), 0), fs, x, false},
		{"scales short", O, I, fw, fs[:len(fs)-1], x, false},
		{"x short", O, I, fw, fs, x[:I-1], false},
		{"x long", O, I, fw, fs, append(append([]float32(nil), x...), 1), false},
		{"e8m0 nan first", O, I, fw, with(fs, 0, 0xff), x, false},
		{"e8m0 nan last", O, I, fw, with(fs, len(fs)-1, 0xff), x, false},
		{"e4m3 nan 0x7f", O, I, with(fw, 17, 0x7f), fs, x, false},
		{"e4m3 nan 0xff last", O, I, with(fw, len(fw)-1, 0xff), fs, x, false},
	})
	mxOracle := func(O, I int, w, s []byte, x []float32) ([]float32, error) {
		return GemvMXFP4E8M0("oracle", O, I, w, s, x)
	}
	run(t, "mxfp4", mxOracle, cuda.GemvExpertMXFP4E8M0, []tc{
		{"ok", O, I, mw, ms, x, true},
		{"ok with 0x7f/0xff weight bytes", O, I, with(with(mw, 3, 0x7f), 5, 0xff), ms, x, true},
		{"zero O", 0, I, mw, ms, x, false},
		{"zero I", O, 0, mw, ms, x, false},
		{"odd I", O, I + 1, append(append([]byte(nil), mw...), make([]byte, O/2)...), ms, append(append([]float32(nil), x...), 1), false},
		{"weight short", O, I, mw[:len(mw)-1], ms, x, false},
		{"scales long", O, I, mw, append(append([]byte(nil), ms...), 127), x, false},
		{"x short", O, I, mw, ms, x[:I-1], false},
		{"e8m0 nan", O, I, mw, with(ms, 7, 0xff), x, false},
	})
	if h := cuda.faultLatch.Health(); h != DeviceHealthy {
		t.Fatalf("refusals poisoned the device session: health=%v", h)
	}
}

// fak-test:runtime integration est=2s lane=optin
func TestCUDARoutedExpertRunToRunIdentical(t *testing.T) {
	cuda := routedExpertCUDAOrSkip(t)
	const O, I = 1024, 2048
	fw, fs, x := routedExpertFP8Case(31, rotating(O, I))
	mw, ms, mx := routedExpertMXFP4Case(32, rotating(O, I))
	for _, tc := range []struct {
		name string
		call func() ([]float32, error)
	}{
		{"fp8", func() ([]float32, error) { return cuda.GemvExpertFP8Block32E8M0(O, I, fw, fs, x) }},
		{"mxfp4", func() ([]float32, error) { return cuda.GemvExpertMXFP4E8M0(O, I, mw, ms, mx) }},
	} {
		a, err := tc.call()
		if err != nil {
			t.Fatal(err)
		}
		b, err := tc.call()
		if err != nil {
			t.Fatal(err)
		}
		for k := range a {
			if math.Float32bits(a[k]) != math.Float32bits(b[k]) {
				t.Fatalf("%s: run-to-run y[%d] %#08x != %#08x", tc.name, k, math.Float32bits(a[k]), math.Float32bits(b[k]))
			}
		}
	}
}

// BenchmarkCUDARoutedExpertGemv reports the per-call phase split at the V4.1-like shapes:
// H2D of the packed bytes dominates until a resident-weight variant exists.
//
//	go test -tags cuda ./internal/compute -run '^$' -bench CUDARoutedExpertGemv -benchtime 50x
func BenchmarkCUDARoutedExpertGemv(b *testing.B) {
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
			var tm cudaRoutedExpertTiming
			if _, err := cuda.routedExpertGemv(bc.format, bc.O, bc.I, w, s, x, &tm); err != nil {
				b.Fatal(err) // warm the pooled buffers outside the timed loop
			}
			var h2d, kern, d2h float64
			b.SetBytes(int64(len(w) + len(s)))
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				if _, err := cuda.routedExpertGemv(bc.format, bc.O, bc.I, w, s, x, &tm); err != nil {
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
