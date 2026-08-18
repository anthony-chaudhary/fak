package stringlist

import (
	"reflect"
	"testing"
)

func TestSplitCSV(t *testing.T) {
	got := SplitCSV(" one, ,two , three")
	want := []string{"one", "two", "three"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("SplitCSV = %#v, want %#v", got, want)
	}
}
