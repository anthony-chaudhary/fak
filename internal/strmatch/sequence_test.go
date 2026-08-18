package strmatch

import "testing"

func TestCommonSlicePrefixLen(t *testing.T) {
	if got := CommonSlicePrefixLen([]string{"a", "b", "c"}, []string{"a", "b", "x", "d"}); got != 2 {
		t.Fatalf("prefix=%d, want 2", got)
	}
	if got := CommonSlicePrefixLen([]int{1}, []int{}); got != 0 {
		t.Fatalf("empty prefix=%d", got)
	}
}
