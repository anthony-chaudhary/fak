package compute

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

const hostEnvironmentVulkanSummary = `Vulkan Instance Version: 1.4.313

Devices:
========
GPU0:
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe
	vendorID           = 0x10005
	deviceID           = 0x0000
	driverID           = DRIVER_ID_MESA_LLVMPIPE
	driverName         = llvmpipe
	driverInfo         = Mesa 25.1.0
GPU1:
	deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName         = AMD Radeon 8060S
	vendorID           = 0x1002
	deviceID           = 0x150e
	driverID           = DRIVER_ID_MESA_RADV
	driverName         = radv
	driverInfo         = Mesa 25.3.1-arch1.1
`

type hostEnvironmentBackend struct {
	Backend
	name     string
	snapshot BackendExecutionSnapshot
	err      error
}

func (b *hostEnvironmentBackend) Name() string            { return b.name }
func (b *hostEnvironmentBackend) Tier() string            { return "fixture" }
func (b *hostEnvironmentBackend) Class() CorrectnessClass { return Approx }
func (b *hostEnvironmentBackend) Caps() Caps              { return Caps{} }
func (b *hostEnvironmentBackend) BackendExecutionSnapshot() (BackendExecutionSnapshot, error) {
	return b.snapshot, b.err
}

func hostEnvironmentFixture() (*hostEnvironmentBackend, hostEnvironmentDeps) {
	backend := &hostEnvironmentBackend{name: "vulkan", snapshot: BackendExecutionSnapshot{Identity: BackendRuntimeIdentity{
		Backend: "vulkan", Device: "AMD Radeon 8060S",
		Driver: "vendor=0x1002 device=0x150e driver=0x00000001", Runtime: "vulkan-1.4.313",
	}}}
	files := map[string]string{
		"/proc/sys/kernel/osrelease":                "6.17.0-test\n",
		"/sys/class/drm/card0/device/vendor":        "0x1002\n",
		"/sys/class/drm/card0/device/device":        "0x150e\n",
		"/sys/class/drm/card0/device/vbios_version": "113-STRIX-TEST\n",
		"/sys/class/drm/card1/device/vendor":        "0x8086\n",
		"/sys/class/drm/card1/device/device":        "0x1234\n",
		"/sys/class/drm/card1/device/vbios_version": "unused\n",
	}
	deps := hostEnvironmentDeps{
		goos: "linux", arch: "amd64",
		readFile: func(path string) ([]byte, error) {
			value, ok := files[path]
			if !ok {
				return nil, fmt.Errorf("fixture has no %s", path)
			}
			return []byte(value), nil
		},
		glob: func(pattern string) ([]string, error) {
			if pattern != "/sys/class/drm/card*/device/vendor" {
				return nil, fmt.Errorf("unexpected glob %q", pattern)
			}
			return []string{"/sys/class/drm/card0/device/vendor", "/sys/class/drm/card1/device/vendor"}, nil
		},
		run: func(ctx context.Context, path string, args ...string) ([]byte, error) {
			switch path {
			case unamePath:
				if !reflect.DeepEqual(args, []string{"-srm"}) {
					return nil, fmt.Errorf("unexpected uname arguments %v", args)
				}
				return []byte("Linux 6.17.0-test x86_64\n"), nil
			case vulkanInfoPath:
				if !reflect.DeepEqual(args, []string{"--summary"}) {
					return nil, fmt.Errorf("unexpected vulkaninfo arguments %v", args)
				}
				return []byte(hostEnvironmentVulkanSummary), nil
			default:
				return nil, fmt.Errorf("unexpected command %q %v", path, args)
			}
		},
	}
	return backend, deps
}

func TestObserveVulkanHostEnvironmentBindsBackendAndObservedBytes(t *testing.T) {
	backend, deps := hostEnvironmentFixture()
	got, err := observeVulkanHostEnvironment(context.Background(), backend, deps)
	if err != nil {
		t.Fatal(err)
	}
	want := VulkanHostEnvironment{
		OS: "linux", Arch: "amd64", Kernel: "6.17.0-test", Device: "AMD Radeon 8060S",
		VendorID: "0x1002", DeviceID: "0x150e", MesaDriver: "radv",
		MesaVersion: "25.3.1-arch1.1", Firmware: "113-STRIX-TEST",
	}
	if got != want {
		t.Fatalf("environment = %+v, want %+v", got, want)
	}
}

func TestObserveVulkanHostEnvironmentFailsClosed(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*hostEnvironmentBackend, *hostEnvironmentDeps)
	}{
		{name: "unsupported OS", mutate: func(_ *hostEnvironmentBackend, d *hostEnvironmentDeps) { d.goos = "windows" }},
		{name: "missing backend observer", mutate: func(b *hostEnvironmentBackend, _ *hostEnvironmentDeps) { b.err = errors.New("unavailable") }},
		{name: "unsupported backend", mutate: func(b *hostEnvironmentBackend, _ *hostEnvironmentDeps) {
			b.name = "cpu-ref"
			b.snapshot.Identity = BackendRuntimeIdentity{Backend: "cpu-ref", Device: "CPU", Driver: "reference", Runtime: "native"}
		}},
		{name: "forged device label", mutate: func(b *hostEnvironmentBackend, _ *hostEnvironmentDeps) { b.snapshot.Identity.Device = "claimed device" }},
		{name: "forged PCI label", mutate: func(b *hostEnvironmentBackend, _ *hostEnvironmentDeps) {
			b.snapshot.Identity.Driver = "vendor=0x1002 device=0xffff driver=0x1"
		}},
		{name: "command failure", mutate: func(_ *hostEnvironmentBackend, d *hostEnvironmentDeps) {
			d.run = func(context.Context, string, ...string) ([]byte, error) { return nil, errors.New("failed") }
		}},
		{name: "kernel identity mismatch", mutate: func(_ *hostEnvironmentBackend, d *hostEnvironmentDeps) {
			originalRun := d.run
			d.run = func(ctx context.Context, path string, args ...string) ([]byte, error) {
				if path == unamePath {
					return []byte("Linux 6.18.0-test x86_64\n"), nil
				}
				return originalRun(ctx, path, args...)
			}
		}},
		{name: "duplicate summary field", mutate: func(_ *hostEnvironmentBackend, d *hostEnvironmentDeps) {
			originalRun := d.run
			d.run = func(ctx context.Context, path string, args ...string) ([]byte, error) {
				if path == vulkanInfoPath {
					return []byte(hostEnvironmentVulkanSummary + "\tdriverInfo = Mesa 99.0\n"), nil
				}
				return originalRun(ctx, path, args...)
			}
		}},
		{name: "ambiguous summary device", mutate: func(_ *hostEnvironmentBackend, d *hostEnvironmentDeps) {
			originalRun := d.run
			d.run = func(ctx context.Context, path string, args ...string) ([]byte, error) {
				if path == vulkanInfoPath {
					return []byte(hostEnvironmentVulkanSummary + `GPU2:
	deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName = AMD Radeon 8060S
	vendorID = 0x1002
	deviceID = 0x150e
	driverID = DRIVER_ID_MESA_RADV
	driverName = radv
	driverInfo = Mesa 25.3.1-arch1.1
`), nil
				}
				return originalRun(ctx, path, args...)
			}
		}},
		{name: "missing kernel", mutate: func(_ *hostEnvironmentBackend, d *hostEnvironmentDeps) {
			d.readFile = func(path string) ([]byte, error) { return nil, fmt.Errorf("missing %s", path) }
		}},
		{name: "ambiguous DRM device", mutate: func(_ *hostEnvironmentBackend, d *hostEnvironmentDeps) {
			originalRead := d.readFile
			d.readFile = func(path string) ([]byte, error) {
				if path == "/sys/class/drm/card1/device/vendor" {
					return []byte("0x1002\n"), nil
				}
				if path == "/sys/class/drm/card1/device/device" {
					return []byte("0x150e\n"), nil
				}
				return originalRead(path)
			}
		}},
		{name: "missing firmware", mutate: func(_ *hostEnvironmentBackend, d *hostEnvironmentDeps) {
			originalRead := d.readFile
			d.readFile = func(path string) ([]byte, error) {
				if path == "/sys/class/drm/card0/device/vbios_version" {
					return nil, errors.New("missing")
				}
				return originalRead(path)
			}
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			backend, deps := hostEnvironmentFixture()
			tc.mutate(backend, &deps)
			got, err := observeVulkanHostEnvironment(context.Background(), backend, deps)
			if err == nil {
				t.Fatalf("expected fail-closed error, got %+v", got)
			}
			if got != (VulkanHostEnvironment{}) {
				t.Fatalf("error returned partial environment: %+v", got)
			}
		})
	}

	backend, deps := hostEnvironmentFixture()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, err := observeVulkanHostEnvironment(ctx, backend, deps)
	if !errors.Is(err, context.Canceled) || got != (VulkanHostEnvironment{}) {
		t.Fatalf("cancellation did not fail closed: env=%+v err=%v", got, err)
	}
}
