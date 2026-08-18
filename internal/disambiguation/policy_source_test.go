package disambiguation

import "testing"

func TestRunPolicySourceSelfTest(t *testing.T) {
	report, err := RunPolicySourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != PolicySourceSelfTestSchemaVersion || len(report.Resolutions) != 6 {
		t.Fatalf("report=%#v", report)
	}
	if !report.StructuralBeforeModel || !report.CapabilityNotVerdict || !report.IncompatibleReasonRejected {
		t.Fatalf("report=%#v", report)
	}
	for _, resolution := range report.Resolutions {
		if resolution.SourcePath == "" || resolution.OwnerLeaf == "" {
			t.Errorf("resolution=%#v", resolution)
		}
	}
}
