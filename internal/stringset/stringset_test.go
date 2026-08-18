package stringset

import (
	"reflect"
	"testing"
)

func TestSorted(t *testing.T) {
	got := Sorted(map[string]struct{}{"b": {}, "a": {}})
	if !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("Sorted = %#v", got)
	}
}
