package main

import (
	"slices"
	"testing"
)

func TestParseIDs(t *testing.T) {
	got, err := parseIDs("1, 2,3")
	if err != nil || !slices.Equal(got, []int{1, 2, 3}) {
		t.Fatalf("got=%v err=%v", got, err)
	}
	for _, bad := range []string{"", "1,x", "1,-2"} {
		if _, err := parseIDs(bad); err == nil {
			t.Fatalf("accepted %q", bad)
		}
	}
}
