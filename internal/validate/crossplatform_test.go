package validate

import (
	"reflect"
	"testing"
)

// TestValidateCrossGOOSCoversTheOtherFleetOS pins the cross_build matrix: whichever OS the build
// phase compiled for, the other fleet OS is compiled too. The 2026-10-10 closed_windows.go break
// compiled on Windows and was red on Linux; a WSL workspace on a Windows host builds Linux, so
// its cross target must be Windows.
func TestValidateCrossGOOSCoversTheOtherFleetOS(t *testing.T) {
	cases := []struct {
		host string
		wsl  bool
		want []string
	}{
		{"windows", false, []string{"linux"}},
		{"windows", true, []string{"windows"}},
		{"linux", false, []string{"windows"}},
		{"darwin", false, []string{"linux", "windows"}},
	}
	for _, tc := range cases {
		if got := validateCrossGOOS(validateBuildGOOS(tc.host, tc.wsl)); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("host=%s wsl=%v: cross GOOS = %v, want %v", tc.host, tc.wsl, got, tc.want)
		}
	}
}
