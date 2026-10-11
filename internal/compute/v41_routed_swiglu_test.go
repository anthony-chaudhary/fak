package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41RoutedSwiGLUValidation(t *testing.T) {
	vec := func(n int) Tensor { return Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{n}} }
	if n, err := validateV41RoutedSwiGLU(vec(7), vec(7), 10, .5); err != nil || n != 7 {
		t.Fatalf("valid geometry = %d, %v", n, err)
	}
	if _, err := validateV41RoutedSwiGLU(vec(7), vec(7), 0, -2); err != nil {
		t.Fatalf("no-clamp limit refused: %v", err)
	}
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	for name, c := range map[string]struct {
		gate, up      Tensor
		limit, weight float32
	}{
		"empty":         {vec(0), vec(0), 1, 1},
		"width differs": {vec(4), vec(5), 1, 1},
		"matrix":        {Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2, 2}}, vec(4), 1, 1},
		"dtype":         {Tensor{Dtype: Q4_K, Layout: RowMajor, Shape: []int{4}}, vec(4), 1, 1},
		"limit inf":     {vec(4), vec(4), inf, 1},
		"limit nan":     {vec(4), vec(4), nan, 1},
		"weight inf":    {vec(4), vec(4), 1, inf},
		"weight nan":    {vec(4), vec(4), 1, nan},
	} {
		if _, err := validateV41RoutedSwiGLU(c.gate, c.up, c.limit, c.weight); err == nil {
			t.Errorf("%s admitted", name)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41RoutedSwiGLUStatusDecode(t *testing.T) {
	const none = math.MaxUint32
	ok := [4]uint32{v41RoutedSwiGLUStatusMagic, 9, none, none}
	if err := decodeV41RoutedSwiGLUStatus(ok, 9); err != nil {
		t.Fatalf("success record: %v", err)
	}
	for _, c := range []struct {
		words [4]uint32
		stage V41RoutedSwiGLUStage
		index int
	}{
		{[4]uint32{v41RoutedSwiGLUStatusMagic, 9, 4, none}, V41RoutedSwiGLUStageActivation, 4},
		{[4]uint32{v41RoutedSwiGLUStatusMagic, 9, none, 2}, V41RoutedSwiGLUStageDownOperand, 2},
		// Any activation fault precedes a lower-lane down-operand fault.
		{[4]uint32{v41RoutedSwiGLUStatusMagic, 9, 8, 0}, V41RoutedSwiGLUStageActivation, 8},
	} {
		var fault *V41RoutedSwiGLUFault
		if err := decodeV41RoutedSwiGLUStatus(c.words, 9); !errors.As(err, &fault) || fault.Stage != c.stage || fault.Index != c.index {
			t.Errorf("words %v = %v, want stage %v lane %d", c.words, err, c.stage, c.index)
		}
	}
	for _, words := range [][4]uint32{
		{none, none, none, none}, // the fill survived: the shader never ran
		{v41RoutedSwiGLUStatusMagic, 8, none, none},
		{v41RoutedSwiGLUStatusMagic, 9, 9, none},
		{v41RoutedSwiGLUStatusMagic, 9, none, 12},
	} {
		var fault *V41RoutedSwiGLUFault
		if err := decodeV41RoutedSwiGLUStatus(words, 9); err == nil || errors.As(err, &fault) {
			t.Errorf("malformed record %v = %v, want a protocol error", words, err)
		}
	}
}
