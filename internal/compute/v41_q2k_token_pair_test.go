package compute

import "testing"

func TestV41Q2KTokenPairGeometry(t *testing.T) {
	for _, out := range []int{1, 3, 5, 2304} {
		for _, rows := range []int{2, 3, 4, 7, 8, 9, 31, 32, 63, 64} {
			p, err := PlanV41Q2KTokenPair(out, 512, rows)
			if err != nil || p.WorkgroupsPairX != uint32(2*((out+1)/2)) || p.WorkgroupsY != uint32((rows+7)/8) || p.OutputBytes != 4*rows*out || p.InputBytes != 4*rows*512 {
				t.Fatalf("geometry out%d rows%d: %+v %v", out, rows, p, err)
			}
			seen := make([]int, 2*rows*out)
			for gx := uint32(0); gx < p.WorkgroupsPairX; gx++ {
				bank := int(gx / p.WorkgroupsX)
				first := int(gx%p.WorkgroupsX) * 2
				for gy := uint32(0); gy < p.WorkgroupsY; gy++ {
					for token := int(gy) * 8; token < rows && token < int(gy+1)*8; token++ {
						for o := first; o < out && o < first+2; o++ {
							seen[bank*rows*out+token*out+o]++
						}
					}
				}
			}
			for i, n := range seen {
				if n != 1 {
					t.Fatalf("bank coverage index%d count%d", i, n)
				}
			}
		}
	}
	for _, dims := range [][3]int{{0, 512, 2}, {5, 511, 2}, {5, 512, 1}, {5, 512, 65}, {1 << 30, 512, 2}, {5, 1 << 30, 8}} {
		if _, err := PlanV41Q2KTokenPair(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("invalid %v", dims)
		}
	}
}

func TestV41Q2KTokenPairValidation(t *testing.T) {
	g := Tensor{Dtype: Q2_K, Layout: RowMajor, Shape: []int{5, 512}}
	u := g
	x := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{8 * 512}}
	if _, e := validateV41Q2KTokenPair(g, u, x, 8); e != nil {
		t.Fatal(e)
	}
	for _, bad := range []Tensor{{Dtype: Q2_K, Layout: RowMajor, Shape: []int{6, 512}}, {Dtype: F32, Layout: RowMajor, Shape: []int{5, 512}}, {Dtype: Q2_K, Layout: RowMajor, Shape: []int{5, 256}}} {
		if _, e := validateV41Q2KTokenPair(g, bad, x, 8); e == nil {
			t.Fatal("up mismatch accepted")
		}
	}
	x.Shape = []int{8*512 - 1}
	if _, e := validateV41Q2KTokenPair(g, u, x, 8); e == nil {
		t.Fatal("short input")
	}
}
