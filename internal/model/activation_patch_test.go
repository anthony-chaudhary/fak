package model

import (
	"reflect"
	"testing"
)

func TestActivationPatchCaptureInject(t *testing.T) {
	patch, err := NewActivationPatch(3)
	if err != nil {
		t.Fatal(err)
	}
	clean := []float32{1, 2, 3}
	if patch.Capture(2, clean) {
		t.Fatal("captured random/control layer")
	}
	if !patch.Capture(3, clean) {
		t.Fatal("did not capture selected L3 layer")
	}
	clean[0] = 99
	if got := patch.Activation(); !reflect.DeepEqual(got, []float32{1, 2, 3}) {
		t.Fatalf("capture aliases source: %v", got)
	}
	corrupt := []float32{-1, -2, -3}
	if changed, err := patch.Inject(2, corrupt); err != nil || changed {
		t.Fatalf("control inject=(%v,%v)", changed, err)
	}
	if !reflect.DeepEqual(corrupt, []float32{-1, -2, -3}) {
		t.Fatalf("control site changed: %v", corrupt)
	}
	if changed, err := patch.Inject(3, corrupt); err != nil || !changed {
		t.Fatalf("L3 inject=(%v,%v)", changed, err)
	}
	if !reflect.DeepEqual(corrupt, []float32{1, 2, 3}) {
		t.Fatalf("patched=%v", corrupt)
	}
}

func TestActivationPatchRejectsMissingAndShapeMismatch(t *testing.T) {
	patch, _ := NewActivationPatch(3)
	if _, err := patch.Inject(3, []float32{0}); err == nil {
		t.Fatal("missing capture accepted")
	}
	patch.Capture(3, []float32{1, 2})
	if _, err := patch.Inject(3, []float32{0}); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}
