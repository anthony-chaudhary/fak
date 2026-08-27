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

func TestConcat(t *testing.T) {
	if got, want := Concat([]int{1, 2}, nil, []int{3}, []int{}, []int{4, 5}), []int{1, 2, 3, 4, 5}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Concat = %v, want %v", got, want)
	}
	if got := Concat(); got != nil {
		t.Fatalf("Concat() = %v, want nil", got)
	}

	src := []int{9, 9}
	out := Concat(src, []int{1})
	out[0] = 42
	if src[0] != 9 {
		t.Errorf("Concat aliased its input: src[0] = %d, want 9", src[0])
	}
}
