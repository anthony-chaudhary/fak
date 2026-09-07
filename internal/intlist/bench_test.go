package intlist

import (
	"testing"
)

func BenchmarkParse(b *testing.B) {
	cases := []struct {
		name string
		in   string
	}{
		{"small", "1,2,4,8"},
		{"medium", "1,2,4,8,16,32,64,128,256,512,1024,2048,4096"},
		{"bracketed", "[1, 2, 4, 8, 16, 32, 64, 128]"},
		{"empty", ""},
		{"nodigits", "abc,def,ghi"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res := Parse(tc.in)
				if tc.name == "small" && len(res) != 4 {
					b.Fatalf("unexpected len: %d", len(res))
				}
			}
		})
	}
}

func BenchmarkConcat(b *testing.B) {
	part1 := []int{1, 2, 4, 8}
	part2 := []int{16, 32, 64, 128}
	part3 := []int{256, 512, 1024, 2048}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Concat(part1, part2, part3)
		if len(res) != 12 {
			b.Fatalf("unexpected len: %d", len(res))
		}
	}
}
