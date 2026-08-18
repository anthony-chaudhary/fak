package sortkeys

import "testing"

func TestFileLine(t *testing.T) {
	if !FileLine("a", 2, "z", "b", 1, "a") || !FileLine("a", 1, "z", "a", 2, "a") || !FileLine("a", 1, "a", "a", 1, "b") {
		t.Fatal("FileLine ordering")
	}
}
