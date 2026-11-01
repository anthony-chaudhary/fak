package compute

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestCUDARawUploadIsVerbatim(t *testing.T) {
	t.Parallel()
	for value := 0; value < 256; value++ {
		dt := Dtype(value)
		want := false
		switch dt {
		case Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ1_M, IQ4_NL, IQ4_XS:
			want = true
		}
		if got := cudaRawUploadIsVerbatim(dt); got != want {
			t.Errorf("dtype %d: verbatim raw-upload eligibility=%v, want %v", value, got, want)
		}
	}
}
