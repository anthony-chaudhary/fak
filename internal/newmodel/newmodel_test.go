package newmodel

import "testing"

func TestReady(t *testing.T) {
	if !Ready() {
		t.Fatal("Ready() should report true for the generated skeleton")
	}
}
