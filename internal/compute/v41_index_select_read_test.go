package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

type indexSelectOwnedRecorder struct {
	Backend
	calls int
	fault error
}

func (b *indexSelectOwnedRecorder) v41IndexSelectReadOwned(original Backend, _ Tensor, _, _, _, _ int) ([]float32, bool, error) {
	if original != b {
		return nil, false, nil
	}
	b.calls++
	if b.fault != nil {
		return nil, true, b.fault
	}
	return []float32{1}, true, nil
}

type indexSelectOwnedWrapper struct{ *indexSelectOwnedRecorder }

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexSelectOwnedReadIdentityAndFailure(t *testing.T) {
	b := &indexSelectOwnedRecorder{}
	r, selected, err := TryV41IndexSelectReadOwned(b, Tensor{}, 4, 1, 0, 0)
	if !selected || err != nil || len(r) != 1 || b.calls != 1 {
		t.Fatal("selection", r, selected, err)
	}
	w := &indexSelectOwnedWrapper{b}
	r, selected, err = TryV41IndexSelectReadOwned(w, Tensor{}, 4, 1, 0, 0)
	if selected || err != nil || r != nil || b.calls != 1 {
		t.Fatal("wrapper promotion selected")
	}
	failure := errors.New("selected")
	b.fault = failure
	r, selected, err = TryV41IndexSelectReadOwned(b, Tensor{}, 4, 1, 0, 0)
	if !selected || err != failure || r != nil || b.calls != 2 {
		t.Fatal("failure replay or loss")
	}
	r, selected, err = TryV41IndexSelectReadOwned(nil, Tensor{}, 4, 1, 0, 0)
	if selected || err != nil || r != nil {
		t.Fatal("missing capability")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexSelectOwnedReadVerdictAndStorage(t *testing.T) {
	for _, little := range []bool{false, true} {
		for _, status := range []uint32{0, 1, 2, 0xffffffff} {
			const n = 8
			image := newV41HostReadbackImage([]int{4, 4 * n}, 1, little)
			binary.LittleEndian.PutUint32(image.region(0, 4), status)
			for i := 4; i < 16; i++ {
				image.host[i] = 0xff
			}
			if image.offsets[1] != 16 || len(image.host) != 16+4*n {
				t.Fatal("aligned image extent")
			}
			payload := image.region(1, 4*n)
			for i := 0; i < n; i++ {
				binary.LittleEndian.PutUint32(payload[4*i:], math.Float32bits(float32(i-1)))
			}
			rows, err := v41IndexSelectOwnedRows(image, n)
			if status != 0 {
				if err == nil || rows != nil {
					t.Fatal("published refused rows", status, rows, err)
				}
				continue
			}
			if err != nil || len(rows) != n || cap(rows) != n || rows[0] != -1 || rows[n-1] != n-2 {
				t.Fatal("row ownership or bits", rows, err)
			}
			if little && (len(image.floats) != n+4 || &rows[0] != &image.floats[4]) {
				t.Fatal("unbounded retained status prefix")
			}
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41IndexSelectOwnedStatusLayoutBounds(t *testing.T) {
	for _, tc := range []struct {
		sizes      []int
		rows       int
		little, ok bool
	}{
		{[]int{4, 4}, 1, true, false}, {[]int{4, 12}, 1, true, false}, {[]int{4, 16}, 1, true, true}, {[]int{4, 65520}, 1, true, true},
		{[]int{4, 4}, 1, false, false}, {[]int{4, 4}, 2, true, false},
		{[]int{4, 4, 16}, 1, true, false}, {[]int{16, 4, 32}, 1, true, false},
		{[]int{8, 32}, 1, true, false}, {[]int{12, 32}, 1, true, false},
		{[]int{4, 0}, 1, true, false}, {[]int{4, 3}, 1, true, false},
		{[]int{4, math.MaxInt &^ 3}, 1, true, false},
	} {
		prefix, total, ok := v41TypedReadbackLayout(tc.sizes, tc.rows, tc.little)
		if ok != tc.ok {
			t.Fatal("layout admission", tc, ok)
		}
		if ok && (prefix != 16 || total != 16+tc.sizes[1]) {
			t.Fatal("status backing", prefix, total)
		}
	}
}
