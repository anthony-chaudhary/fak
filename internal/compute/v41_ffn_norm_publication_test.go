package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41FFNNormPublicationRowVerdicts(t *testing.T) {
	for _, rows := range []int{2, 32, 64} {
		const width = 8
		slab := make([]float32, rows*width+4)
		for i := range slab {
			slab[i] = float32(i)
		}
		for _, fault := range []int{-1, 0, width - 1, width, rows*width - 1} {
			p := &V41FFNNormPublication{values: slab[4:], rows: rows, width: width, firstBad: fault}
			for row := 0; row < rows; row++ {
				values, err := p.Row(row)
				if fault < 0 || row < fault/width {
					if err != nil || len(values) != width || cap(values) != width || &values[0] != &slab[4+row*width] {
						t.Fatalf("bad owned view: rows=%d row=%d fault=%d err=%v", rows, row, fault, err)
					}
				} else if row == fault/width {
					var arithmetic *V41BF16CheckedArithmeticError
					var backend *BackendError
					if values != nil || !errors.As(err, &arithmetic) || arithmetic.Index != fault%width || arithmetic.Site != V41BF16PublishFFNNormOutput || !errors.As(err, &backend) || backend.Site != "V41BF16Publish" {
						t.Fatalf("wrong row-local verdict: %v", err)
					}
				} else {
					var protocol *V41BF16CheckedProtocolError
					if values != nil || !errors.As(err, &protocol) {
						t.Fatalf("later row escaped: %v", err)
					}
				}
			}
		}
	}
	var absent *V41FFNNormPublication
	if row, err := absent.Row(0); row != nil || err == nil {
		t.Fatal("nil result exposed")
	}
}
