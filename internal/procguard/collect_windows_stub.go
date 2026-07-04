//go:build !windows

package procguard

func collectWindowsRelationsNative() ([]Proc, bool, string) {
	return nil, false, ""
}

func killTreeWindowsNative(int) (bool, string, bool) {
	return false, "", false
}
