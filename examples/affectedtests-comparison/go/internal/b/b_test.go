package b

import "testing"

func TestB(t *testing.T) {
	if Value() != 2 {
		t.Fatal(Value())
	}
}
