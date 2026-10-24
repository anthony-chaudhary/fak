package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestNVFP4BackendProductDomain(t *testing.T) {
	const rows, cols = 16 * 127, 16
	be, ok := Default().(*cpuBackend)
	if !ok {
		t.Fatal("registered default is not the CPU reference backend")
	}
	packed := make([]byte, rows*cols/2)
	scales := make([]byte, rows)
	for row := range scales {
		code, scale := byte(row%16), byte(row/16)
		packed[row*8] = code
		scales[row] = scale
	}
	w, err := NewNVFP4(be, []int{rows, cols}, packed, scales, 1)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, cols)
	x[0] = 1
	got := be.Read(be.MatMul(be.Upload(w, NVFP4), NewF32(be, []int{cols}, x)))
	if !be.SupportsDeviceWeightDtype(NVFP4) || len(got) != rows {
		t.Fatal("CPU backend did not expose the complete NVFP4 operation")
	}
	for row, value := range got {
		want := nvfp4TestLocal(byte(row%16), byte(row/16))
		if value != want {
			t.Fatalf("code=%x scale=%x got=%v want=%v", row%16, row/16, value, want)
		}
		if bits := math.Float32bits(want); bits&0xffff != 0 {
			t.Fatalf("finite local product code=%x scale=%x is not exact BF16: %08x", row%16, row/16, bits)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4BackendMatrixAndActivationPrecision(t *testing.T) {
	const rows, cols, batch = 3, 32, 3
	be := &cpuBackend{}
	packed := make([]byte, rows*cols/2)
	for i := range packed {
		packed[i] = byte(i*17 + 0x32)
	}
	scales := []byte{0x01, 0x39, 0x7e, 0x28, 0, 0x40}
	x := make([]float32, batch*cols)
	for i := range x {
		x[i] = float32(i%13-6)*0.137 + 0.003
	}
	x[0], x[1] = math.Float32frombits(0x3f808000), math.Float32frombits(0x3f818000)
	original := append([]float32(nil), x...)
	w, err := NewNVFP4(be, []int{rows, cols}, packed, scales, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	got := be.Read(be.BatchedMatMul(w, NewF32(be, []int{batch, cols}, x), batch))
	if len(got) != batch*rows {
		t.Fatalf("output length=%d want=%d", len(got), batch*rows)
	}
	for token := 0; token < batch; token++ {
		one := be.Read(be.MatMul(w, NewF32(be, []int{cols}, x[token*cols:(token+1)*cols])))
		for row := 0; row < rows; row++ {
			want := nvfp4TestDot(row, cols, packed, scales, 0.1, x[token*cols:(token+1)*cols])
			if got[token*rows+row] != want || one[row] != want {
				t.Fatalf("token=%d row=%d batch=%v single=%v want=%v", token, row, got[token*rows+row], one[row], want)
			}
		}
	}

	for i, value := range x {
		if math.Float32bits(value) != math.Float32bits(original[i]) {
			t.Fatalf("activation input mutated at %d", i)
		}
	}

	packed = make([]byte, 8)
	packed[0] = 2
	w, err = NewNVFP4(be, []int{1, 16}, packed, []byte{0x38}, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ input, want float32 }{
		{1.375, 1.375},
		{math.Float32frombits(0x3f808000), 1},
		{math.Float32frombits(0x3f818000), 1.015625},
		{math.Float32frombits(0xbf818000), -1.015625},
	} {
		values := make([]float32, 16)
		values[0] = tc.input
		value := be.Read(be.MatMul(w, NewF32(be, []int{16}, values)))[0]
		if value != tc.want {
			t.Fatalf("activation=%v got=%v want BF16 RN-even %v", tc.input, value, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4BackendLateTensorScale(t *testing.T) {
	be := &cpuBackend{}
	codes := []byte{10, 7, 14, 5, 6, 5, 3, 14, 13, 13, 10, 12, 5, 2, 1, 4}
	packed := make([]byte, 8)
	for i, code := range codes {
		packed[i/2] |= code << uint(4*(i%2))
	}
	x := []float32{0.125, 0.125, -1, -2, 0.125, 1.375, 1, 1.375, 123, 1.375, 123, -2, 0.75, 0.75, -1, 0.125}
	w, err := NewNVFP4(be, []int{1, 16}, packed, []byte{0x38}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	got := be.Read(be.MatMul(w, NewF32(be, []int{16}, x)))[0]
	if math.Float32bits(got) != 0xc2440ccd {
		t.Fatalf("late tensor scaling got=%v (%08x), want -49.01250076293945", got, math.Float32bits(got))
	}
	early, err := GemvNVFP4Reference(1, 16, packed, []byte{0x38}, 0.1, x)
	if err != nil {
		t.Fatal(err)
	}
	if got == early[0] {
		t.Fatal("fixture failed to distinguish late tensor scaling from scaling every weight")
	}

	packed = make([]byte, 8)
	packed[0] = 0xf7
	w, err = NewNVFP4(be, []int{1, 16}, packed, []byte{0x38}, math.MaxFloat32)
	if err != nil {
		t.Fatalf("finite late-scale cancellation was rejected: %v", err)
	}
	x = make([]float32, 16)
	x[0], x[1] = 1, 1
	if got := be.Read(be.MatMul(w, NewF32(be, []int{16}, x)))[0]; got != 0 {
		t.Fatalf("cancelled sum got=%v want=0", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4BackendConstructorAndOwnership(t *testing.T) {
	be := &cpuBackend{}
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name           string
		shape          []int
		packed, scales []byte
		global         float32
		want           error
	}{
		{"rank", []int{16}, make([]byte, 8), []byte{0x38}, 1, ErrNVFP4Shape},
		{"zero", []int{0, 16}, nil, nil, 1, ErrNVFP4Shape},
		{"block", []int{1, 17}, make([]byte, 8), []byte{0x38}, 1, ErrNVFP4Shape},
		{"overflow", []int{maxInt, 16}, nil, nil, 1, ErrNVFP4Shape},
		{"packed", []int{1, 16}, make([]byte, 7), []byte{0x38}, 1, ErrNVFP4Shape},
		{"scales", []int{1, 16}, make([]byte, 8), nil, 1, ErrNVFP4Scale},
		{"negative block", []int{1, 16}, make([]byte, 8), []byte{0xb8}, 1, ErrNVFP4Scale},
		{"NaN block", []int{1, 16}, make([]byte, 8), []byte{0x7f}, 1, ErrNVFP4Scale},
		{"negative global", []int{1, 16}, make([]byte, 8), []byte{0x38}, -1, ErrNVFP4Scale},
		{"NaN global", []int{1, 16}, make([]byte, 8), []byte{0x38}, float32(math.NaN()), ErrNVFP4Scale},
		{"infinite global", []int{1, 16}, make([]byte, 8), []byte{0x38}, float32(math.Inf(1)), ErrNVFP4Scale},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewNVFP4(be, tc.shape, tc.packed, tc.scales, tc.global)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error=%v want sentinel=%v", err, tc.want)
			}
		})
	}
	shape, packed, scales := []int{1, 16}, make([]byte, 8), []byte{0x38}
	packed[0] = 2
	w, err := NewNVFP4(be, shape, packed, scales, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	shape[0], packed[0], scales[0] = 9, 7, 0x7e
	x := make([]float32, 16)
	x[0] = 1
	if got := be.Read(be.MatMul(w, NewF32(be, []int{16}, x)))[0]; got != 0.5 || w.Shape[0] != 1 {
		t.Fatalf("constructor did not own shape and payload: output=%v shape=%v", got, w.Shape)
	}
	cloned, err := be.CloneTensor(w)
	if err != nil {
		t.Fatal(err)
	}
	w.buf.(HostBuffer).I8()[0], w.Quant.Scale[0], w.Shape[0] = 7, 2, 3
	if got := be.Read(be.MatMul(cloned, NewF32(be, []int{16}, x)))[0]; got != 0.5 || cloned.Shape[0] != 1 {
		t.Fatalf("clone shares NVFP4 data or descriptors: output=%v shape=%v", got, cloned.Shape)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4BackendRejectsCorruptWeights(t *testing.T) {
	be := &cpuBackend{}
	for _, tc := range []struct {
		name   string
		mutate func(*Tensor)
		want   error
	}{
		{"dtype", func(w *Tensor) { w.Dtype = F32 }, ErrNVFP4Shape},
		{"layout", func(w *Tensor) { w.Layout = ColMajor }, ErrNVFP4Shape},
		{"rank", func(w *Tensor) { w.Shape = []int{16} }, ErrNVFP4Shape},
		{"storage", func(w *Tensor) { w.buf.(*hostBuf).i8 = w.buf.(*hostBuf).i8[:8] }, ErrNVFP4Shape},
		{"quant missing", func(w *Tensor) { w.Quant = nil }, ErrNVFP4Scale},
		{"block", func(w *Tensor) { w.Quant.Block = 32 }, ErrNVFP4Scale},
		{"axis", func(w *Tensor) { w.Quant.Axis = 0 }, ErrNVFP4Scale},
		{"bits", func(w *Tensor) { w.Quant.Bits = 8 }, ErrNVFP4Scale},
		{"asymmetric", func(w *Tensor) { w.Quant.Symmetric = false }, ErrNVFP4Scale},
		{"zero point", func(w *Tensor) { w.Quant.ZeroPoint = []int32{0} }, ErrNVFP4Scale},
		{"static activation", func(w *Tensor) { w.Quant.StaticAct = true }, ErrNVFP4Scale},
		{"global missing", func(w *Tensor) { w.Quant.Scale = nil }, ErrNVFP4Scale},
		{"global extra", func(w *Tensor) { w.Quant.Scale = []float32{1, 2} }, ErrNVFP4Scale},
		{"global NaN", func(w *Tensor) { w.Quant.Scale[0] = float32(math.NaN()) }, ErrNVFP4Scale},
		{"raw scale NaN", func(w *Tensor) { w.buf.(HostBuffer).I8()[8] = 0x7f }, ErrNVFP4Scale},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, batched := range []bool{false, true} {
				w, err := NewNVFP4(be, []int{1, 16}, make([]byte, 8), []byte{0x38}, 1)
				if err != nil {
					t.Fatal(err)
				}
				tc.mutate(&w)
				nvfp4TestPanic(t, tc.want, func() {
					if batched {
						be.BatchedMatMul(w, NewF32(be, []int{1, 16}, make([]float32, 16)), 1)
					} else {
						be.MatMul(w, NewF32(be, []int{16}, make([]float32, 16)))
					}
				})
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4BackendRejectsInvalidActivations(t *testing.T) {
	be := &cpuBackend{}
	packed := make([]byte, 8)
	packed[0] = 7
	w, err := NewNVFP4(be, []int{1, 16}, packed, []byte{0x38}, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Tensor)
	}{
		{"shape", func(x *Tensor) { x.Shape = []int{15} }},
		{"storage", func(x *Tensor) { x.buf.(*hostBuf).f32 = make([]float32, 15) }},
		{"dtype", func(x *Tensor) { x.Dtype = BF16 }},
		{"layout", func(x *Tensor) { x.Layout = ColMajor }},
		{"NaN", func(x *Tensor) { x.buf.(*hostBuf).f32[0] = float32(math.NaN()) }},
		{"infinity", func(x *Tensor) { x.buf.(*hostBuf).f32[0] = float32(math.Inf(1)) }},
		{"BF16 overflow", func(x *Tensor) { x.buf.(*hostBuf).f32[0] = math.MaxFloat32 }},
		{"dot overflow", func(x *Tensor) { x.buf.(*hostBuf).f32[0] = math.Float32frombits(0x7e800000) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := NewF32(be, []int{16}, make([]float32, 16))
			tc.mutate(&x)
			nvfp4TestPanic(t, ErrNVFP4Input, func() { be.MatMul(w, x) })
		})
	}
	for _, p := range []int{0, -1, 2} {
		nvfp4TestPanic(t, ErrNVFP4Input, func() { be.BatchedMatMul(w, NewF32(be, []int{1, 16}, make([]float32, 16)), p) })
	}
	w.Quant.Scale[0] = math.MaxFloat32
	x := make([]float32, 16)
	x[0] = 1
	nvfp4TestPanic(t, ErrNVFP4Input, func() { be.MatMul(w, NewF32(be, []int{16}, x)) })
}

// fak-test:runtime fast est=3s lane=default
func TestNVFP4BackendDoesNotMaterializeWeight(t *testing.T) {
	const rows, cols, batch = 64, 512, 2
	be := &cpuBackend{}
	packed, scales := make([]byte, rows*cols/2), make([]byte, rows*cols/16)
	for i := range packed {
		packed[i] = 0x32
	}
	for i := range scales {
		scales[i] = 0x38
	}
	w, err := NewNVFP4(be, []int{rows, cols}, packed, scales, 1)
	if err != nil {
		t.Fatal(err)
	}
	x := NewF32(be, []int{cols}, make([]float32, cols))
	xs := NewF32(be, []int{batch, cols}, make([]float32, batch*cols))
	for _, tc := range []struct {
		name string
		run  func()
	}{
		{"MatMul", func() { be.MatMul(w, x) }},
		{"BatchedMatMul", func() { be.BatchedMatMul(w, xs, batch) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result := testing.Benchmark(func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					tc.run()
				}
			})
			t.Logf("%s allocation=%d B/op packed weight=%d bytes", tc.name, result.AllocedBytesPerOp(), len(packed)+len(scales))
			if got, ceiling := result.AllocedBytesPerOp(), int64(len(packed)+len(scales)); got >= ceiling {
				t.Fatalf("allocates %d B/op >= packed weight %d bytes; weight copied or materialized", got, ceiling)
			}
		})
	}
}

func nvfp4TestLocal(code, scale byte) float32 {
	var integer int
	exponent := int(scale>>3) - 10
	if scale>>3 == 0 {
		integer, exponent = int(scale&7), -9
	} else {
		integer = 8 + int(scale&7)
	}
	q := int(code & 7)
	var numerator int
	switch q {
	case 0:
		numerator = 0
	case 1:
		numerator = 1
	case 2:
		numerator = 2
	case 3:
		numerator = 3
	case 4:
		numerator = 4
	case 5:
		numerator = 6
	case 6:
		numerator = 8
	case 7:
		numerator = 12
	}
	if code&8 != 0 {
		numerator = -numerator
	}
	return float32(math.Ldexp(float64(integer*numerator), exponent-1))
}

func nvfp4TestBF16(value float32) float32 {
	bits := math.Float32bits(value)
	hi, lo := bits>>16, bits&0xffff
	if lo > 0x8000 || lo == 0x8000 && hi&1 != 0 {
		hi++
	}
	return math.Float32frombits(hi << 16)
}

func nvfp4TestDot(row, cols int, packed, scales []byte, global float32, x []float32) float32 {
	var sum float32
	for k, value := range x {
		index := row*cols + k
		code := packed[index/2] >> uint(4*(k%2)) & 15
		product := float32(nvfp4TestLocal(code, scales[row*(cols/16)+k/16]) * nvfp4TestBF16(value))
		sum = float32(sum + product)
	}
	return sum * global
}

func nvfp4TestPanic(t *testing.T, want error, run func()) {
	t.Helper()
	defer func() {
		got := recover()
		err, ok := got.(error)
		if !ok || !errors.Is(err, want) {
			t.Errorf("panic=%v want sentinel=%v", got, want)
		}
	}()
	run()
}
