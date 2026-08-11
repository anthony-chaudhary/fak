package c

import "testing"

func TestC(t *testing.T) {
	if Value() != 3 {
		t.Fatal(Value())
	}
}
