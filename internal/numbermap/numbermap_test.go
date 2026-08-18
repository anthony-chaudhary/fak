package numbermap

import (
	"reflect"
	"testing"
)

func TestInts(t *testing.T) {
	got := Ints(map[string]any{"one": float64(1)}, func(v any) int { return int(v.(float64)) })
	if !reflect.DeepEqual(got, map[string]int{"one": 1}) {
		t.Fatalf("Ints = %#v", got)
	}
}
