//go:build cuda && cgo && fak_cuda_v41_router

package compute

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

func cudaV41RouterOrSkip(t *testing.T) *cudaBackend {
	t.Helper()
	cb := cudaOrSkip(t)
	if !cb.V41RouterTopKAvailable() {
		t.Skip("CUDA V4.1 router admission refused on this device")
	}
	return cb
}

// v41RouterFixture draws distinct 1/64-quantized logits and 1/8-quantized
// biases so no two corrected values tie by accident.
func v41RouterFixture(rows, experts int, seed uint32) (logits, bias []float32) {
	logits, bias = make([]float32, rows*experts), make([]float32, experts)
	for i := range logits {
		seed = seed*1664525 + 1013904223
		logits[i] = float32(int32(seed>>8)%1201-600) / 64
	}
	for e := range bias {
		seed = seed*1664525 + 1013904223
		bias[e] = float32(int32(seed>>8)%17-8) / 8
	}
	return logits, bias
}

func requireV41RouterParity(t *testing.T, label string, got V41RouterPicks, wantIDs []int32, wantW []float32) {
	t.Helper()
	if len(got.Experts) != len(wantIDs) || len(got.Weights) != len(wantW) {
		t.Fatalf("%s: extent %d/%d, want %d", label, len(got.Experts), len(got.Weights), len(wantIDs))
	}
	for i := range wantIDs {
		if got.Experts[i] != wantIDs[i] {
			t.Fatalf("%s: slot %d expert %d, want %d (all %v vs %v)", label, i, got.Experts[i], wantIDs[i], got.Experts, wantIDs)
		}
		if d := math.Abs(float64(got.Weights[i] - wantW[i])); d > 4e-6*math.Max(1, math.Abs(float64(wantW[i]))) {
			t.Fatalf("%s: slot %d weight %g, want %g (|d|=%g)", label, i, got.Weights[i], wantW[i], d)
		}
	}
}

// fak-test:runtime slow est=3s lane=cuda
func TestCUDAV41RouterTopKParity(t *testing.T) {
	cb := cudaV41RouterOrSkip(t)
	cases := 0
	for _, experts := range []int{7, 256, 384} {
		for _, topK := range []int{1, 6, 8} {
			if topK > experts {
				continue
			}
			for _, rows := range []int{1, 5, 33} {
				logits, bias := v41RouterFixture(rows, experts, uint32(experts*131+topK*17+rows))
				dl := mkResident(cb, []int{rows, experts}, logits)
				db := mkResident(cb, []int{experts}, bias)
				dummy := mkResident(cb, []int{1}, []float32{0})
				for _, full := range []bool{false, true} {
					for _, hasBias := range []bool{false, true} {
						label := fmt.Sprintf("E%d k%d r%d full=%v bias=%v", experts, topK, rows, full, hasBias)
						hb, devBias := []float32(nil), dummy
						if hasBias {
							hb, devBias = bias, db
						}
						ids, w, stage := v41RouterReferenceForTest(logits, hb, nil, rows, experts, topK, 1.5, full)
						if stage != 0 {
							t.Fatalf("%s: reference faulted at %v", label, stage)
						}
						before := cb.HostXferBytes()
						got, err := cb.V41RouterTopK(dl, devBias, rows, experts, topK, 1.5, full, hasBias)
						if err != nil {
							t.Fatalf("%s: %v", label, err)
						}
						// Only status words and picks cross host-ward, never the logits.
						if moved, want := cb.HostXferBytes()-before, uint64(rows*16+2*rows*topK*4); moved != want {
							t.Fatalf("%s: D2H %d bytes, want %d", label, moved, want)
						}
						requireV41RouterParity(t, label, got, ids, w)
						cases++
					}
				}
				cb.Free(dl)
				cb.Free(db)
				cb.Free(dummy)
			}
		}
	}
	t.Logf("router top-k parity: %d cases bit-exact expert ids vs host reference", cases)
}

// fak-test:runtime slow est=1s lane=cuda
func TestCUDAV41RouterTopKTiesAndFaults(t *testing.T) {
	cb := cudaV41RouterOrSkip(t)
	// 300 equal logits and biases: the picks are the lowest experts in order.
	equal := make([]float32, 300)
	dl := mkResident(cb, []int{1, 300}, equal)
	dummy := mkResident(cb, []int{1}, []float32{0})
	got, err := cb.V41RouterTopK(dl, dummy, 1, 300, 6, 1.5, true, false)
	if err != nil {
		t.Fatal(err)
	}
	for slot, e := range got.Experts {
		if e != int32(slot) || got.Weights[slot] != got.Weights[0] {
			t.Fatalf("tie order = %v %v", got.Experts, got.Weights)
		}
	}
	cb.Free(dl)

	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	for _, tc := range []struct {
		name    string
		logits  []float32
		bias    []float32
		k       int
		full    bool
		stage   V41RouterTopKStage
		expert  int
		rowWant int
	}{
		{"lowest-logit-fault", []float32{1, 2, 3, inf, 4, 5, 6, nan}, nil, 2, true, V41RouterTopKStageLogit, 3, 0},
		{"bias-fault", []float32{1, 2, 3, 4}, []float32{0, nan, 0, 0}, 2, true, V41RouterTopKStageBias, 1, 0},
		{"zero-reduced-sum", []float32{-1000, -1000, -1000, -1000}, nil, 2, false, V41RouterTopKStageNormalization, -1, 0},
	} {
		rows := len(tc.logits) / 4
		experts := len(tc.logits) / rows
		dl := mkResident(cb, []int{rows, experts}, tc.logits)
		db, hasBias := dummy, tc.bias != nil
		if hasBias {
			db = mkResident(cb, []int{experts}, tc.bias)
		}
		_, err := cb.V41RouterTopK(dl, db, rows, experts, tc.k, 1.5, tc.full, hasBias)
		var fault *V41RouterTopKFault
		if !errors.As(err, &fault) || fault.Stage != tc.stage || (tc.expert >= 0 && fault.Expert != tc.expert) || fault.Row != tc.rowWant {
			t.Fatalf("%s: err=%v, want stage %v expert %d", tc.name, err, tc.stage, tc.expert)
		}
		cb.Free(dl)
		if hasBias {
			cb.Free(db)
		}
	}
	cb.Free(dummy)
}

// fak-test:runtime slow est=1s lane=cuda
func TestCUDAV41RouterHashParity(t *testing.T) {
	cb := cudaV41RouterOrSkip(t)
	for _, experts := range []int{256, 384} {
		for _, rows := range []int{1, 3, 17} {
			const topK = 6
			logits, _ := v41RouterFixture(rows, experts, uint32(experts+rows))
			ids := make([]int32, rows*topK)
			seed := uint32(rows*977 + experts)
			for row := 0; row < rows; row++ {
				for slot := 0; slot < topK; {
					seed = seed*1664525 + 1013904223
					id := int32(seed>>8) % int32(experts)
					dup := false
					for _, prev := range ids[row*topK : row*topK+slot] {
						dup = dup || prev == id
					}
					if !dup {
						ids[row*topK+slot] = id
						slot++
					}
				}
			}
			dl := mkResident(cb, []int{rows, experts}, logits)
			for _, full := range []bool{false, true} {
				label := fmt.Sprintf("hash E%d r%d full=%v", experts, rows, full)
				wantIDs, wantW, stage := v41RouterReferenceForTest(logits, nil, ids, rows, experts, topK, 1.5, full)
				if stage != 0 {
					t.Fatalf("%s: reference faulted at %v", label, stage)
				}
				got, err := cb.V41RouterHash(dl, ids, rows, experts, topK, 1.5, full)
				if err != nil {
					t.Fatalf("%s: %v", label, err)
				}
				requireV41RouterParity(t, label, got, wantIDs, wantW)
			}
			cb.Free(dl)
		}
	}
	// A non-finite logit outside the table still faults, as on the host.
	logits := make([]float32, 8)
	logits[5] = float32(math.NaN())
	dl := mkResident(cb, []int{1, 8}, logits)
	_, err := cb.V41RouterHash(dl, []int32{0, 1}, 1, 8, 2, 1.5, false)
	var fault *V41RouterTopKFault
	if !errors.As(err, &fault) || fault.Stage != V41RouterTopKStageLogit || fault.Expert != 5 {
		t.Fatalf("hash logit fault = %v", err)
	}
	if _, err := cb.V41RouterHash(dl, []int32{0, 0}, 1, 8, 2, 1.5, false); err == nil || errors.As(err, &fault) {
		t.Fatalf("repeated hash id = %v, want a geometry refusal", err)
	}
	cb.Free(dl)
}

// fak-test:runtime slow est=1s lane=cuda
func TestCUDAV41RouterGateTopKMatchesMatMulThenRoute(t *testing.T) {
	cb := cudaV41RouterOrSkip(t)
	const experts, hidden, topK = 256, 4096, 6
	w, x, bias := v41RouterGateProbeFixture(experts, hidden)
	dw := mkResident(cb, []int{experts, hidden}, w)
	dx := mkResident(cb, []int{hidden}, x)
	db := mkResident(cb, []int{experts}, bias)
	defer func() { cb.Free(dw); cb.Free(dx); cb.Free(db) }()
	for _, full := range []bool{false, true} {
		got, err := cb.V41RouterGateTopK(dw, dx, db, experts, hidden, topK, 1.5, full, true)
		if err != nil {
			t.Fatal(err)
		}
		logits := cb.MatMul(dw, dx)
		host := cb.Read(logits)
		row := makeTensor(cb, F32, RowMajor, []int{1, experts}, nil, logits.buf)
		want, err := cb.V41RouterTopK(row, db, 1, experts, topK, 1.5, full, true)
		cb.Free(logits)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.Experts {
			if got.Experts[i] != want.Experts[i] || math.Float32bits(got.Weights[i]) != math.Float32bits(want.Weights[i]) {
				t.Fatalf("full=%v gate %v %v != MatMul+route %v %v", full, got.Experts, got.Weights, want.Experts, want.Weights)
			}
		}
		ids, wts, stage := v41RouterReferenceForTest(host, bias, nil, 1, experts, topK, 1.5, full)
		if stage != 0 {
			t.Fatalf("reference faulted at %v", stage)
		}
		requireV41RouterParity(t, fmt.Sprintf("gate full=%v", full), got, ids, wts)
	}
}

// fak-test:runtime slow est=1s lane=cuda
func TestCUDAV41EmbeddingGatherBitExact(t *testing.T) {
	cb := cudaV41RouterOrSkip(t)
	const vocab, hidden = 97, 4096
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = math.Float32frombits(uint32(i)*2654435761 | 1) // includes NaN/Inf/denormal patterns
	}
	dt := mkResident(cb, []int{vocab, hidden}, table)
	defer cb.Free(dt)
	ids := []int{5, 0, 96, 5, 42}
	out, err := cb.V41EmbeddingGather(dt, ids, hidden)
	if err != nil {
		t.Fatal(err)
	}
	got := cb.Read(out)
	cb.Free(out)
	for r, id := range ids {
		for c := 0; c < hidden; c++ {
			if math.Float32bits(got[r*hidden+c]) != math.Float32bits(table[id*hidden+c]) {
				t.Fatalf("row %d (id %d) col %d bits %#x, want %#x", r, id, c, math.Float32bits(got[r*hidden+c]), math.Float32bits(table[id*hidden+c]))
			}
		}
	}
	if _, err := cb.V41EmbeddingGather(dt, []int{vocab}, hidden); err == nil {
		t.Fatal("out-of-range id was admitted")
	}
}

// fak-test:runtime slow est=1s lane=default
func TestCUDAV41RouterFullSoftplusPublication(t *testing.T) {
	cb := cudaV41RouterOrSkip(t)
	const rows, experts, topK = 2, 384, 3
	values := make([]float32, rows*experts)
	for i := range values {
		values[i] = -110
	}
	logits := mkResident(cb, []int{rows, experts}, values)
	defer cb.Free(logits)
	bias := mkResident(cb, []int{1}, []float32{0})
	defer cb.Free(bias)
	hash := []int32{7, 67, 131, 13, 87, 191}
	for _, hashed := range []bool{false, true} {
		var got V41RouterPicks
		var err error
		if hashed {
			got, err = cb.V41RouterHash(logits, hash, rows, experts, topK, 1.5, true)
		} else {
			got, err = cb.V41RouterTopK(logits, bias, rows, experts, topK, 1.5, true, false)
		}
		if err != nil {
			t.Fatal(err)
		}
		if len(got.Experts) != rows*topK || len(got.Weights) != rows*topK {
			t.Fatal("publication extent")
		}
		for i := range got.Experts {
			want := int32(i % topK)
			if hashed {
				want = hash[i]
			}
			if got.Experts[i] != want || math.Float32bits(got.Weights[i]) != 0 {
				t.Fatalf("full=%v hash=%v slot%d: expert=%d weight=%g", true, hashed, i, got.Experts[i], got.Weights[i])
			}
		}
	}
	// Reduced geometry intentionally retains sqrt(softplus) across the F32
	// publication boundary; the one-pick score/itself normalization is exact.
	got, err := cb.V41RouterTopK(logits, bias, rows, experts, 1, 1.5, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Experts) != rows || len(got.Weights) != rows {
		t.Fatal("reduced publication extent")
	}
	for i := range got.Experts {
		if got.Experts[i] != 0 || got.Weights[i] != 1.5 {
			t.Fatalf("reduced control: %+v", got)
		}
	}
}
