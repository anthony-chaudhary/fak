//go:build !linux

package modelperfobs

import (
	"reflect"
	"testing"
)

func TestParseIDListOffLinux(t *testing.T) {
	got, err := parseIDList("0-2,8,11-12")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 1, 2, 8, 11, 12}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}
