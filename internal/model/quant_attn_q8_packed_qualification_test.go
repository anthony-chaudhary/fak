//go:build fak_qattn_q8_packed_qualification

package model

import "testing"

// This checks actual build selection, not numerical or performance qualification.
func TestQ8PackedQualificationEnabled(t *testing.T) {
	if !q8PackedDecodeExperimental {
		t.Fatal("packed Q8 control does not match the selected build posture")
	}
	for _, value := range []string{"", "0", "1", "true"} {
		t.Setenv("FAK_QATTN_Q8_PACKED", value)
		if !q8PackedDecodeExperimental {
			t.Fatal("retired environment control changed packed Q8 admission")
		}
	}
}
