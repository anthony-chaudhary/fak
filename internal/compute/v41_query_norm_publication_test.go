package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41QueryPublicationOrderedVerdicts(t *testing.T) {
	p := &V41QueryNormPublication{values: make([]float32, 12), rows: 3, width: 4, inputBad: 9, firstBad: 6}
	if err := p.InputRowError(1); err != nil {
		t.Fatal(err)
	}
	_, err := p.Row(1)
	var fault *V41BF16CheckedArithmeticError
	if !errors.As(err, &fault) || fault.Site != V41BF16PublishQueryNormOutput || fault.Index != 2 {
		t.Fatalf("output attribution: %v", err)
	}
	err = p.InputRowError(2)
	if !errors.As(err, &fault) || fault.Site != V41BF16PublishQueryNormInput || fault.Index != 1 {
		t.Fatalf("input attribution: %v", err)
	}
	row, err := p.Row(0)
	if err != nil || len(row) != 4 || cap(row) != 4 {
		t.Fatalf("row: %v", err)
	}
	if _, err = p.Row(2); err == nil {
		t.Fatal("later output exposed")
	}
	if err = p.InputRowError(3); err == nil {
		t.Fatal("invalid input row accepted")
	}
}
