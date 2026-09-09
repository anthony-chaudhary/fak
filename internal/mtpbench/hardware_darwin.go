//go:build darwin

package mtpbench

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func hardwareIdentity() (HardwareIdentity, error) {
	host, err := os.Hostname()
	if err != nil {
		return HardwareIdentity{}, err
	}
	version, err := syscall.Sysctl("kern.osproductversion")
	if err != nil {
		return HardwareIdentity{}, err
	}
	build, err := syscall.Sysctl("kern.osversion")
	if err != nil {
		return HardwareIdentity{}, err
	}
	ramRaw, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return HardwareIdentity{}, err
	}
	ram, err := strconv.ParseUint(strings.TrimSpace(ramRaw), 10, 64)
	if err != nil {
		return HardwareIdentity{}, err
	}
	metalMemory, ok := metalgemm.DeviceMemoryTotal()
	device := strings.TrimSpace(metalgemm.DeviceName())
	if !ok || metalMemory == 0 || device == "" || host == "" || strings.TrimSpace(version) == "" || strings.TrimSpace(build) == "" || ram == 0 || runtime.NumCPU() <= 0 {
		return HardwareIdentity{}, fmt.Errorf("mtpbench: incomplete hardware identity")
	}
	return HardwareIdentity{Hostname: host, MetalDevice: device, OSVersion: strings.TrimSpace(version), OSBuild: strings.TrimSpace(build), Arch: runtime.GOARCH, CPUs: uint64(runtime.NumCPU()), PhysicalRAMBytes: ram, MetalMemoryBytes: metalMemory}, nil
}
