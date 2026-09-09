package policy

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestMissionPolicyContract(t *testing.T) {
	if ReasonMissionWriteSetViolation != adjudicator.ReasonMissionWriteSetViolation {
		t.Fatalf("ReasonMissionWriteSetViolation mismatch: %v != %v",
			ReasonMissionWriteSetViolation, adjudicator.ReasonMissionWriteSetViolation)
	}
	if ReasonMissionWriteSetViolationName != adjudicator.ReasonMissionWriteSetViolationName {
		t.Fatalf("ReasonMissionWriteSetViolationName mismatch: %q != %q",
			ReasonMissionWriteSetViolationName, adjudicator.ReasonMissionWriteSetViolationName)
	}

	mc := NewMissionContract([]string{"file1.go"}, []string{"extract/"})
	if len(mc.WriteSet) != 1 || mc.WriteSet[0] != "file1.go" {
		t.Fatalf("unexpected WriteSet: %v", mc.WriteSet)
	}
	if len(mc.ExtractInto) != 1 || mc.ExtractInto[0] != "extract/" {
		t.Fatalf("unexpected ExtractInto: %v", mc.ExtractInto)
	}
}
