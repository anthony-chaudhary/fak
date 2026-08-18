package pathutil

import "testing"

func TestNormalizeScope(t *testing.T) {
	if got := NormalizeScope(` ././internal\foo/ `); got != "internal/foo" {
		t.Fatalf("NormalizeScope=%q", got)
	}
}
