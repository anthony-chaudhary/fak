//go:build windows

package workerworktree

import (
	"syscall"
	"testing"
	"time"
)

func TestFiletimeDurationCombinesKernelAndUserTicks(t *testing.T) {
	kernel := syscall.Filetime{LowDateTime: 20}
	user := syscall.Filetime{LowDateTime: 30}
	if got, want := filetimeDuration(kernel, user), 5*time.Microsecond; got != want {
		t.Fatalf("filetimeDuration = %s, want %s", got, want)
	}
}

func TestCurrentLandResourcesWindowsExposesCPUAndPeakRSS(t *testing.T) {
	sample := currentLandResources()
	if !sample.cpuAvailable || !sample.rssAvailable {
		t.Fatalf("Windows process resource probes unavailable: %+v", sample)
	}
	if sample.cpuTime < 0 || sample.peakRSSBytes <= 0 {
		t.Fatalf("invalid Windows process resource sample: %+v", sample)
	}
}
