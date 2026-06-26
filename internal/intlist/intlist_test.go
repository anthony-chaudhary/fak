package intlist

import (
	"reflect"
	"testing"
)

func TestParse(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []int
	}{
		{"comma separated", "1,2,4,8", []int{1, 2, 4, 8}},
		{"space separated", "1 2 4 8", []int{1, 2, 4, 8}},
		{"bracketed", "[1, 2, 4]", []int{1, 2, 4}},
		{"multi digit", "16,256,1024", []int{16, 256, 1024}},
		{"trailing separator", "3,5,", []int{3, 5}},
		{"leading separator", ",3,5", []int{3, 5}},
		{"single value", "7", []int{7}},
		{"empty string", "", nil},
		{"no digits", "abc", nil},
	}
	for _, tc := range cases {
		got := Parse(tc.in)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("Parse(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}
