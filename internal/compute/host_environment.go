package compute

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const (
	unamePath      = "/usr/bin/uname"
	vulkanInfoPath = "/usr/bin/vulkaninfo"
)

// VulkanHostEnvironment is host and driver evidence observed beside the
// backend-selected Vulkan device. It is not execution authority by itself.
type VulkanHostEnvironment struct {
	OS          string
	Arch        string
	Kernel      string
	Device      string
	VendorID    string
	DeviceID    string
	MesaDriver  string
	MesaVersion string
	Firmware    string
}

// ObserveVulkanHostEnvironment observes a Linux Vulkan environment and binds
// it to identity obtained directly from backend. It accepts no identity labels
// from flags or environment variables and returns a zero value on any error.
func ObserveVulkanHostEnvironment(ctx context.Context, backend Backend) (VulkanHostEnvironment, error) {
	return observeVulkanHostEnvironment(ctx, backend, hostEnvironmentDeps{
		goos:     runtime.GOOS,
		arch:     runtime.GOARCH,
		readFile: os.ReadFile,
		glob:     filepath.Glob,
		run: func(ctx context.Context, path string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, path, args...).Output()
		},
	})
}

type hostEnvironmentDeps struct {
	goos     string
	arch     string
	readFile func(string) ([]byte, error)
	glob     func(string) ([]string, error)
	run      func(context.Context, string, ...string) ([]byte, error)
}

type vulkanSummaryDevice struct {
	name, vendorID, deviceID, driverName, driverInfo string
}

func observeVulkanHostEnvironment(ctx context.Context, backend Backend, deps hostEnvironmentDeps) (VulkanHostEnvironment, error) {
	if ctx == nil {
		return VulkanHostEnvironment{}, errors.New("compute: nil context for Vulkan host observation")
	}
	if err := ctx.Err(); err != nil {
		return VulkanHostEnvironment{}, fmt.Errorf("compute: Vulkan host observation: %w", err)
	}
	if deps.goos != "linux" || strings.TrimSpace(deps.arch) == "" {
		return VulkanHostEnvironment{}, fmt.Errorf("compute: Vulkan host observation requires Linux with a known architecture")
	}
	if deps.readFile == nil || deps.glob == nil || deps.run == nil {
		return VulkanHostEnvironment{}, errors.New("compute: Vulkan host observation dependencies are incomplete")
	}

	snapshot, available, err := CaptureBackendExecutionSnapshot(backend)
	if err != nil {
		return VulkanHostEnvironment{}, fmt.Errorf("compute: capture backend identity for host observation: %w", err)
	}
	if !available || snapshot.Identity.Backend != "vulkan" {
		return VulkanHostEnvironment{}, errors.New("compute: Vulkan backend identity is unavailable")
	}
	backendVendor, backendDevice, err := parseBackendPCIIdentity(snapshot.Identity.Driver)
	if err != nil {
		return VulkanHostEnvironment{}, err
	}

	uname, err := deps.run(ctx, unamePath, "-srm")
	if err != nil {
		return VulkanHostEnvironment{}, fmt.Errorf("compute: observe host kernel identity: %w", err)
	}
	observedOS, observedKernel, observedArch, err := parseUname(uname)
	if err != nil {
		return VulkanHostEnvironment{}, err
	}
	if observedArch != deps.arch {
		return VulkanHostEnvironment{}, errors.New("compute: observed host architecture does not match the running binary")
	}
	kernel, err := readObservedLine(deps.readFile, "/proc/sys/kernel/osrelease", "kernel release")
	if err != nil {
		return VulkanHostEnvironment{}, err
	}
	if kernel != observedKernel {
		return VulkanHostEnvironment{}, errors.New("compute: procfs and uname kernel releases do not match")
	}
	summary, err := deps.run(ctx, vulkanInfoPath, "--summary")
	if err != nil {
		return VulkanHostEnvironment{}, fmt.Errorf("compute: observe Vulkan summary: %w", err)
	}
	devices, err := parseVulkanSummary(summary)
	if err != nil {
		return VulkanHostEnvironment{}, err
	}

	var selected *vulkanSummaryDevice
	for i := range devices {
		device := &devices[i]
		if device.vendorID == backendVendor && device.deviceID == backendDevice && device.name == snapshot.Identity.Device {
			if selected != nil {
				return VulkanHostEnvironment{}, errors.New("compute: Vulkan summary contains ambiguous backend device identity")
			}
			selected = device
		}
	}
	if selected == nil {
		return VulkanHostEnvironment{}, errors.New("compute: Vulkan summary does not match the backend-selected device")
	}

	firmware, err := observeMatchingFirmware(deps, backendVendor, backendDevice)
	if err != nil {
		return VulkanHostEnvironment{}, err
	}
	return VulkanHostEnvironment{
		OS: observedOS, Arch: observedArch, Kernel: kernel, Device: selected.name,
		VendorID: selected.vendorID, DeviceID: selected.deviceID,
		MesaDriver: selected.driverName, MesaVersion: selected.driverInfo, Firmware: firmware,
	}, nil
}

func parseUname(data []byte) (string, string, string, error) {
	fields := strings.Fields(string(data))
	if len(fields) != 3 || fields[0] != "Linux" || strings.ContainsAny(fields[1], "\r\n") {
		return "", "", "", errors.New("compute: uname did not return an exact Linux kernel tuple")
	}
	arch := ""
	switch fields[2] {
	case "x86_64":
		arch = "amd64"
	case "aarch64":
		arch = "arm64"
	default:
		return "", "", "", errors.New("compute: uname returned an unsupported host architecture")
	}
	return "linux", fields[1], arch, nil
}

func parseBackendPCIIdentity(driver string) (string, string, error) {
	fields := strings.Fields(driver)
	if len(fields) != 3 {
		return "", "", errors.New("compute: backend driver identity is not the sealed PCI tuple")
	}
	values := make(map[string]string, len(fields))
	for _, field := range fields {
		key, value, ok := strings.Cut(field, "=")
		if !ok || values[key] != "" {
			return "", "", errors.New("compute: backend driver identity is malformed")
		}
		values[key] = value
	}
	if values["driver"] == "" {
		return "", "", errors.New("compute: backend driver version is unavailable")
	}
	vendor, err := canonicalPCIID(values["vendor"])
	if err != nil {
		return "", "", fmt.Errorf("compute: backend vendor identity: %w", err)
	}
	device, err := canonicalPCIID(values["device"])
	if err != nil {
		return "", "", fmt.Errorf("compute: backend device identity: %w", err)
	}
	return vendor, device, nil
}

func parseVulkanSummary(data []byte) ([]vulkanSummaryDevice, error) {
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	var blocks []map[string]string
	var current map[string]string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if isGPUHeader(line) {
			if current != nil {
				blocks = append(blocks, current)
			}
			current = make(map[string]string)
			continue
		}
		if current == nil {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		switch key {
		case "deviceName", "deviceType", "vendorID", "deviceID", "driverID", "driverName", "driverInfo":
			if _, exists := current[key]; exists {
				return nil, fmt.Errorf("compute: Vulkan summary duplicates %s", key)
			}
			current[key] = value
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("compute: scan Vulkan summary: %w", err)
	}
	if current != nil {
		blocks = append(blocks, current)
	}

	devices := make([]vulkanSummaryDevice, 0, len(blocks))
	for _, block := range blocks {
		if block["driverID"] != "DRIVER_ID_MESA_RADV" || !strings.EqualFold(block["driverName"], "radv") {
			continue
		}
		if block["deviceType"] != "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU" && block["deviceType"] != "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" {
			return nil, errors.New("compute: RADV Vulkan device has unsupported device type")
		}
		name := strings.TrimSpace(block["deviceName"])
		if name == "" {
			return nil, errors.New("compute: RADV Vulkan device name is unavailable")
		}
		vendor, err := canonicalPCIID(block["vendorID"])
		if err != nil || vendor != "0x1002" {
			return nil, errors.New("compute: RADV Vulkan vendor identity is missing or not AMD")
		}
		device, err := canonicalPCIID(block["deviceID"])
		if err != nil {
			return nil, errors.New("compute: RADV Vulkan device identity is missing")
		}
		driverInfo := strings.TrimSpace(block["driverInfo"])
		if len(driverInfo) <= len("Mesa ") || !strings.EqualFold(driverInfo[:len("Mesa ")], "Mesa ") {
			return nil, errors.New("compute: RADV Mesa version is unavailable")
		}
		devices = append(devices, vulkanSummaryDevice{
			name: name, vendorID: vendor, deviceID: device,
			driverName: "radv", driverInfo: strings.TrimSpace(driverInfo[len("Mesa "):]),
		})
	}
	if len(devices) == 0 {
		return nil, errors.New("compute: Vulkan summary contains no complete AMD RADV device")
	}
	return devices, nil
}

func isGPUHeader(line string) bool {
	if len(line) < len("GPU0:") || !strings.HasPrefix(line, "GPU") || line[len(line)-1] != ':' {
		return false
	}
	_, err := strconv.ParseUint(line[3:len(line)-1], 10, 31)
	return err == nil
}

func observeMatchingFirmware(deps hostEnvironmentDeps, vendorID, deviceID string) (string, error) {
	paths, err := deps.glob("/sys/class/drm/card*/device/vendor")
	if err != nil {
		return "", fmt.Errorf("compute: enumerate DRM devices: %w", err)
	}
	seen := make(map[string]struct{}, len(paths))
	matches := make([]string, 0, 1)
	for _, vendorPath := range paths {
		deviceDir := path.Clean(path.Dir(vendorPath))
		if _, duplicate := seen[deviceDir]; duplicate {
			return "", errors.New("compute: DRM device enumeration contains duplicates")
		}
		seen[deviceDir] = struct{}{}
		vendor, err := readObservedPCIID(deps.readFile, vendorPath, "DRM vendor")
		if err != nil {
			return "", err
		}
		device, err := readObservedPCIID(deps.readFile, path.Join(deviceDir, "device"), "DRM device")
		if err != nil {
			return "", err
		}
		if vendor == vendorID && device == deviceID {
			matches = append(matches, deviceDir)
		}
	}
	if len(matches) != 1 {
		return "", fmt.Errorf("compute: expected one DRM device matching the executed Vulkan device, found %d", len(matches))
	}
	return readObservedLine(deps.readFile, path.Join(matches[0], "vbios_version"), "VBIOS firmware")
}

func readObservedPCIID(readFile func(string) ([]byte, error), path, field string) (string, error) {
	value, err := readObservedLine(readFile, path, field)
	if err != nil {
		return "", err
	}
	canonical, err := canonicalPCIID(value)
	if err != nil {
		return "", fmt.Errorf("compute: %s is malformed", field)
	}
	return canonical, nil
}

func readObservedLine(readFile func(string) ([]byte, error), path, field string) (string, error) {
	data, err := readFile(path)
	if err != nil {
		return "", fmt.Errorf("compute: observe %s: %w", field, err)
	}
	value := strings.TrimSpace(string(data))
	if value == "" || strings.ContainsAny(value, "\r\n") {
		return "", fmt.Errorf("compute: observed %s is empty or multiline", field)
	}
	return value, nil
}

func canonicalPCIID(value string) (string, error) {
	value = strings.TrimSpace(value)
	if len(value) < 3 || !strings.EqualFold(value[:2], "0x") {
		return "", errors.New("PCI identity requires a hexadecimal 0x prefix")
	}
	id, err := strconv.ParseUint(value[2:], 16, 16)
	if err != nil {
		return "", errors.New("PCI identity is not a 16-bit hexadecimal value")
	}
	return fmt.Sprintf("0x%04x", id), nil
}
