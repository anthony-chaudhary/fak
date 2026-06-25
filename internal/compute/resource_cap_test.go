package compute

import (
	"strings"
	"testing"
)

func TestSingleResourceCapExceeded(t *testing.T) {
	for _, tc := range []struct {
		name   string
		bytes  int
		cap    int64
		exceed bool
	}{
		{"unknown cap", 1 << 30, 0, false},
		{"under", 63, 64, false},
		{"equal", 64, 64, false},
		{"over", 65, 64, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := singleResourceCapExceeded(tc.bytes, tc.cap); got != tc.exceed {
				t.Fatalf("singleResourceCapExceeded(%d,%d)=%v want %v", tc.bytes, tc.cap, got, tc.exceed)
			}
		})
	}
}

func TestFormatVulkanResourceCapErrorNamesBufferAndCaps(t *testing.T) {
	got := formatVulkanResourceCapError("Q8_0 weight code buffer [8192,524288]", 4<<30, 2<<30, 2<<30, 3<<30)
	for _, want := range []string{
		"Q8_0 weight code buffer [8192,524288]",
		"4294967296 bytes",
		"2147483648 bytes",
		"maxStorageBufferRange=2147483648",
		"maxMemoryAllocationSize=3221225472",
		"split/chunk",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatted cap error missing %q:\n%s", want, got)
		}
	}
}
