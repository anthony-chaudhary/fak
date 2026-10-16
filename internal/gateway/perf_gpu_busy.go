package gateway

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// gpuBusyPath is the first amdgpu gpu_busy_percent counter on the host, the
// same file `fak bench h2h --gpu-busy auto` samples; "" when none exists.
var gpuBusyPath = sync.OnceValue(func() string {
	m, _ := filepath.Glob("/sys/class/drm/card*/device/gpu_busy_percent")
	if len(m) == 0 {
		return ""
	}
	sort.Strings(m)
	return m[0]
})

// sampleGPUBusy is swapped by tests; production reads the drm sysfs counter.
var sampleGPUBusy = func() *int { return readGPUBusy(gpuBusyPath()) }

// readGPUBusy returns the counter's percentage, or nil when the file is absent,
// unreadable, or out of range, so a missing counter never reads as an idle GPU.
func readGPUBusy(path string) *int {
	if path == "" {
		return nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	v, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil || v < 0 || v > 100 {
		return nil
	}
	return &v
}
