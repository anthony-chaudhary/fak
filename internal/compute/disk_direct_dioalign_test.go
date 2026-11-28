package compute

import "testing"

func TestCheckDioAlign(t *testing.T) {
	t.Parallel()
	for _, c := range []struct {
		name     string
		reported bool
		mem, off uint32
		want     DioAlignVerdict
	}{
		{"unreporting fs defers to the probe", false, 0, 0, DioAlignDeferToProbe},
		{"offset 0 refuses (tmpfs/overlay)", true, 4, 0, DioAlignRefuseNoDirect},
		{"512/4 accepted", true, 4, 512, DioAlignAccept},
		{"1/1 accepted", true, 1, 1, DioAlignAccept},
		{"exactly the padding accepted", true, 4096, 4096, DioAlignAccept},
		{"offset above padding refuses", true, 4, 8192, DioAlignRefuseTooCoarse},
		{"mem above padding refuses", true, 8192, 512, DioAlignRefuseTooCoarse},
	} {
		if got := CheckDioAlign(c.reported, c.mem, c.off); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}
