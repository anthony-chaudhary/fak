package disambiguation

import "testing"

func TestRunOwnershipSourceSelfTest(t *testing.T) {
	report, err := RunOwnershipSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if report.Binding.ModuleAtRev != "internal/disambiguation@r31+g76d63130cd" || report.Binding.Leaf != "disambiguation" || report.Binding.Lane != "disambiguation" || report.Binding.Stamp != "(fak disambiguation)" {
		t.Fatalf("binding=%#v", report.Binding)
	}
	if !report.LeafMismatchTyped || !report.LaneMismatchTyped || !report.StampMismatchTyped {
		t.Fatalf("report=%#v", report)
	}
}
