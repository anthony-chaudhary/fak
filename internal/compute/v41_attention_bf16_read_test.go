package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

type attentionOwnedRecorder struct {
	Backend
	calls int
	fault error
}

func (b *attentionOwnedRecorder) v41AttentionBF16ReadOwned(original Backend, _ Tensor, _ uint32) ([]float32, bool, error) {
	if original != b {
		return nil, false, nil
	}
	b.calls++
	if b.fault != nil {
		return nil, true, b.fault
	}
	return []float32{1}, true, nil
}

type attentionOwnedWrapper struct{ *attentionOwnedRecorder }

// fak-test:runtime fast est=5ms lane=default
func TestV41AttentionOwnedReadIdentityAndFailure(t *testing.T) {
	b := &attentionOwnedRecorder{}
	r, selected, err := TryV41AttentionBF16ReadOwned(b, Tensor{}, 3)
	if !selected || err != nil || len(r) != 1 || b.calls != 1 {
		t.Fatal("selection", r, selected, err)
	}
	w := &attentionOwnedWrapper{b}
	r, selected, err = TryV41AttentionBF16ReadOwned(w, Tensor{}, 3)
	if selected || err != nil || r != nil || b.calls != 1 {
		t.Fatal("wrapper promotion selected")
	}
	failure := errors.New("selected")
	b.fault = failure
	r, selected, err = TryV41AttentionBF16ReadOwned(b, Tensor{}, 4)
	if !selected || err != failure || r != nil || b.calls != 2 {
		t.Fatal("failure replay or loss")
	}
	r, selected, err = TryV41AttentionBF16ReadOwned(nil, Tensor{}, 3)
	if selected || err != nil || r != nil {
		t.Fatal("missing capability")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41AttentionOwnedReadVerdictAndStorage(t *testing.T) {
	for _, little := range []bool{false, true} {
		for _, site := range []uint32{1, 2, 3, 4, 5, 6} {
			for _, bad := range []bool{false, true} {
				const n = 8
				image := newV41HostReadbackImage([]int{16, 4 * n}, 1, little)
				raw := image.region(0, 16)
				binary.LittleEndian.PutUint32(raw, 0x42463143)
				binary.LittleEndian.PutUint32(raw[4:], site)
				idx := uint32(math.MaxUint32)
				if bad {
					idx = 2
				}
				binary.LittleEndian.PutUint32(raw[8:], idx)
				binary.LittleEndian.PutUint32(raw[12:], n)
				payload := image.region(1, 4*n)
				for j := 0; j < n; j++ {
					bits := math.Float32bits(float32(j))
					if j == 0 {
						bits = 0x80000000
					}
					binary.LittleEndian.PutUint32(payload[4*j:], bits)
				}
				row, err := v41AttentionBF16OwnedRow(image, n, site)
				valid := (site == 3 || site == 4) && !bad
				if (err == nil) != valid {
					t.Fatal("site/index verdict", site, bad, err)
				}
				if !valid {
					if row != nil {
						t.Fatal("published invalid row")
					}
					continue
				}
				if len(row) != n || cap(row) != n || math.Float32bits(row[0]) != 0x80000000 || row[7] != 7 {
					t.Fatal("owned row bits or bounds", row)
				}
				if little && (len(image.floats) != n+4 || &row[0] != &image.floats[4]) {
					t.Fatal("unbounded retained prefix")
				}
			}
		}
	}
}
