package stablejson

import (
	"bytes"
	"testing"
)

func TestMarshalIndentTrailingNewlineAndSortedKeys(t *testing.T) {
	got, err := Marshal(map[string]any{"b": 1, "a": "x", "nested": map[string]any{"z": true, "y": 2}})
	if err != nil {
		t.Fatal(err)
	}
	want := "{\n  \"a\": \"x\",\n  \"b\": 1,\n  \"nested\": {\n    \"y\": 2,\n    \"z\": true\n  }\n}\n"
	if !bytes.Equal(got, []byte(want)) {
		t.Fatalf("bytes mismatch:\n got %q\nwant %q", got, want)
	}
	// Key order is deterministic: two marshals of the same map are byte-identical.
	again, err := Marshal(map[string]any{"b": 1, "a": "x", "nested": map[string]any{"z": true, "y": 2}})
	if err != nil || !bytes.Equal(got, again) {
		t.Fatalf("not deterministic: %v %v", err, bytes.Equal(got, again))
	}
}

func TestMarshalNilAndErrorPropagation(t *testing.T) {
	got, err := Marshal(nil)
	if err != nil || string(got) != "null\n" {
		t.Fatalf("nil: %q %v", got, err)
	}
	if _, err := Marshal(map[string]any{"ch": make(chan int)}); err == nil {
		t.Fatal("marshal error not propagated")
	}
}
