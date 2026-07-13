package maputil

import (
	"reflect"
	"testing"
)

func TestSortedKeysOrdersAndPreservesAllKeys(t *testing.T) {
	if got, want := SortedKeys(map[string]bool{"z": true, "a": false, "m": true}), []string{"a", "m", "z"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("SortedKeys = %v, want %v", got, want)
	}
	if got := SortedKeys(map[string]int(nil)); got == nil || len(got) != 0 {
		t.Fatalf("SortedKeys(nil) = %#v, want non-nil empty slice", got)
	}
}
