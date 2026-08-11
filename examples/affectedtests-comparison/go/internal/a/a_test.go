package a

import "testing"

func TestA(t *testing.T) {
	if Value() != 1 {
		t.Fatal(Value())
	}
}
