package metrics

import "testing"

func TestParsePromLabels(t *testing.T) {
	got, ok := ParsePromLabels(`model="a", worker="b"`)
	if !ok || got["model"] != "a" || got["worker"] != "b" {
		t.Fatalf("ParsePromLabels = %#v, %v", got, ok)
	}
	if _, ok := ParsePromLabels(`model=a`); ok {
		t.Fatal("unquoted label accepted")
	}
}
