package isolated

import "testing"

func TestIsolated(t *testing.T) {
	if Value() != 99 {
		t.Fatal(Value())
	}
}
