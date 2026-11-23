package compute

import (
	"strings"
	"testing"
)

func TestCheckPackedMatVecRefusesBadGeometry(t *testing.T) {
	w := Tensor{Shape: []int{8, 100}}
	x := make([]float32, 128)
	if err := CheckPackedMatVec([]Tensor{w}, x, []int{0}, []int{0}, 64); err != nil {
		t.Fatalf("valid batch refused: %v", err)
	}
	for _, c := range []struct {
		name       string
		ws         []Tensor
		xOff, yOff []int
		yLen       int
		want       string
	}{
		{"empty", nil, nil, nil, 64, "one x and y offset"},
		{"offset count", []Tensor{w}, []int{0, 64}, []int{0}, 64, "one x and y offset"},
		{"unaligned", []Tensor{w}, []int{0}, []int{8}, 64, "not 64-aligned"},
		{"x overrun", []Tensor{w}, []int{64}, []int{0}, 64, "out of range"},
		{"y overrun", []Tensor{w}, []int{0}, []int{0}, 4, "out of range"},
		{"rank", []Tensor{{Shape: []int{8}}}, []int{0}, []int{0}, 64, "shape"},
	} {
		err := CheckPackedMatVec(c.ws, x, c.xOff, c.yOff, c.yLen)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want %q", c.name, err, c.want)
		}
	}
}

func TestCheckSituMoERefusesMismatchedGeometry(t *testing.T) {
	gu, dn := Tensor{Shape: []int{8, 32}}, Tensor{Shape: []int{16, 4}}
	one := []float32{1}
	if err := CheckSituMoE([]Tensor{gu}, []Tensor{dn}, make([]float32, 30), one, one, one); err != nil {
		t.Fatalf("valid call refused: %v", err)
	}
	for _, c := range []struct {
		name   string
		gu, dn []Tensor
		x      int
		want   string
	}{
		{"count", []Tensor{gu, gu}, []Tensor{dn}, 32, "equal"},
		{"chain", []Tensor{gu}, []Tensor{{Shape: []int{16, 3}}}, 32, "do not chain"},
		{"x too long", []Tensor{gu}, []Tensor{dn}, 33, "do not chain"},
		{"odd gate-up", []Tensor{{Shape: []int{7, 32}}}, []Tensor{dn}, 32, "do not chain"},
	} {
		w := make([]float32, len(c.gu))
		err := CheckSituMoE(c.gu, c.dn, make([]float32, c.x), w, w, w)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want %q", c.name, err, c.want)
		}
	}
	err := CheckSituMoE([]Tensor{gu, gu}, []Tensor{dn, {Shape: []int{16, 8}}}, make([]float32, 32), []float32{1, 1}, []float32{1, 1}, []float32{1, 1})
	if err == nil || !strings.Contains(err.Error(), "differ") {
		t.Errorf("mixed shapes: err %v", err)
	}
}
