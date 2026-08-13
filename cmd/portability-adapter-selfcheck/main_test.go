package main

import (
	"context"
	"github.com/anthony-chaudhary/fak/internal/portability"
	"testing"
)

func TestLiveSelfcheck(t *testing.T) {
	r, e := portability.RunReferenceConformance(context.Background())
	if e != nil || len(r) != 9 {
		t.Fatalf("%v %d", e, len(r))
	}
}
