package selfupdate

import "testing"

func TestCompareReleaseVersions(t *testing.T) {
	for _, tc := range []struct {
		left, right string
		want        int
	}{
		{"1.2.3", "1.2.3", 0},
		{"v1.2.4", "1.2.3", 1},
		{"1.2", "1.2.0", 0},
		{"2.0.0-rc1", "2.0.0", -1},
		{"2.0.0-rc2", "2.0.0-rc1", 1},
	} {
		got, err := CompareReleaseVersions(tc.left, tc.right)
		if err != nil || got != tc.want {
			t.Fatalf("CompareReleaseVersions(%q, %q) = %d, %v; want %d", tc.left, tc.right, got, err, tc.want)
		}
	}
	if _, err := CompareReleaseVersions("dev", "1.0.0"); err == nil {
		t.Fatal("opaque version accepted")
	}
}
