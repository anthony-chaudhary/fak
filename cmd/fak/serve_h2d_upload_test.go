package main

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestFormatServeH2DUploadSplitsDirectFromStaged(t *testing.T) {
	for _, tc := range []struct {
		direct, total uint64
		want          string
	}{
		{0, 0, "h2d_direct=0 h2d_total=0 h2d_staged=0 scope=process-cumulative"},
		{4096, 6144, "h2d_direct=4096 h2d_total=6144 h2d_staged=2048 scope=process-cumulative"},
		{7, 3, "h2d_direct=7 h2d_total=3 h2d_staged=unknown scope=process-cumulative"},
	} {
		if got := formatServeH2DUpload(tc.direct, tc.total); got != tc.want {
			t.Errorf("formatServeH2DUpload(%d, %d) = %q, want %q", tc.direct, tc.total, got, tc.want)
		}
	}
}
