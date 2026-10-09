package compute

import (
	"errors"
	"strings"
	"testing"
)

type carveoutBackend struct {
	*hostEnvironmentBackend
	deviceType uint32
}

func (b *carveoutBackend) VulkanPhysicalDeviceType() uint32 { return b.deviceType }

// fak-test:runtime fast est=20ms lane=default
func TestDedicatedVRAMCarveoutFromSysfs(t *testing.T) {
	const selectedVRAM = hostEnvironmentDRMDevice + "/mem_info_vram_total"
	const selectedBytes = int64(96 << 30)
	tests := []struct {
		name        string
		raw         string
		missingVRAM bool
		mutate      func(*carveoutBackend, *hostEnvironmentDeps)
		wantKnown   bool
	}{
		{name: "selected Halo with another AMD card", wantKnown: true},
		{name: "selected Intel cannot borrow unrelated AMD VRAM", mutate: func(b *carveoutBackend, _ *hostEnvironmentDeps) {
			b.snapshot.Identity.Driver = "vendor=0x8086 device=0x1234 driver=0x1"
		}},
		{name: "selected discrete AMD is not a host carveout", mutate: func(b *carveoutBackend, _ *hostEnvironmentDeps) {
			b.deviceType = 2
		}},
		{name: "unknown device type", mutate: func(b *carveoutBackend, _ *hostEnvironmentDeps) {
			b.deviceType = 0
		}},
		{name: "unsupported OS", mutate: func(_ *carveoutBackend, d *hostEnvironmentDeps) {
			d.goos = "windows"
		}},
		{name: "missing backend identity", mutate: func(b *carveoutBackend, _ *hostEnvironmentDeps) {
			b.err = errors.New("unavailable")
		}},
		{name: "malformed backend PCI identity", mutate: func(b *carveoutBackend, _ *hostEnvironmentDeps) {
			b.snapshot.Identity.Driver = "unavailable"
		}},
		{name: "missing selected render node", mutate: func(b *carveoutBackend, _ *hostEnvironmentDeps) {
			b.renderNode = nil
		}},
		{name: "missing selected VRAM never borrows another card", missingVRAM: true},
		{name: "wrong selected PCI identity", mutate: func(b *carveoutBackend, _ *hostEnvironmentDeps) {
			b.snapshot.Identity.Driver = "vendor=0x1002 device=0xffff driver=0x1"
		}},
		{name: "wrong selected driver", mutate: func(_ *carveoutBackend, d *hostEnvironmentDeps) {
			originalResolve := d.evalSymlinks
			d.evalSymlinks = func(p string) (string, error) {
				if p == hostEnvironmentDRMDevice+"/driver" {
					return "/sys/bus/pci/drivers/vfio-pci", nil
				}
				return originalResolve(p)
			}
		}},
		{name: "selected binding changes during read", mutate: func(b *carveoutBackend, d *hostEnvironmentDeps) {
			originalRead := d.readFile
			d.readFile = func(p string) ([]byte, error) {
				if p == selectedVRAM {
					b.renderNode = func() (uint64, uint64, bool) { return 226, 128, true }
				}
				return originalRead(p)
			}
		}},
		{name: "backend identity changes during read", mutate: func(b *carveoutBackend, d *hostEnvironmentDeps) {
			originalRead := d.readFile
			d.readFile = func(p string) ([]byte, error) {
				if p == selectedVRAM {
					b.snapshot.Identity.Driver = "vendor=0x1002 device=0x150e driver=0x2"
				}
				return originalRead(p)
			}
		}},
		{name: "backend disappears during read", mutate: func(b *carveoutBackend, d *hostEnvironmentDeps) {
			originalRead := d.readFile
			d.readFile = func(p string) ([]byte, error) {
				if p == selectedVRAM {
					b.err = errors.New("unavailable")
				}
				return originalRead(p)
			}
		}},
		{name: "device type changes during read", mutate: func(b *carveoutBackend, d *hostEnvironmentDeps) {
			originalRead := d.readFile
			d.readFile = func(p string) ([]byte, error) {
				if p == selectedVRAM {
					b.deviceType = 2
				}
				return originalRead(p)
			}
		}},
		{name: "zero capacity", raw: "0"},
		{name: "negative capacity", raw: "-1"},
		{name: "malformed capacity", raw: "unknown"},
		{name: "overflowing capacity", raw: "9223372036854775808"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base, deps := hostEnvironmentFixture()
			backend := &carveoutBackend{hostEnvironmentBackend: base, deviceType: 1}
			originalRead := deps.readFile
			deps.readFile = func(p string) ([]byte, error) {
				if strings.HasSuffix(p, "/mem_info_gtt_total") || strings.HasPrefix(p, "/sys/class/drm/") || (strings.HasSuffix(p, "/mem_info_vram_total") && p != selectedVRAM) {
					t.Fatalf("attempted capacity read outside the selected dedicated pool: %s", p)
				}
				if p == selectedVRAM {
					if tt.missingVRAM {
						return nil, errors.New("unavailable")
					}
					if tt.raw != "" {
						return []byte(tt.raw), nil
					}
					return []byte("103079215104\n"), nil
				}
				return originalRead(p)
			}
			// This observation must not invoke uname or vulkaninfo.
			deps.run = nil
			if tt.mutate != nil {
				tt.mutate(backend, &deps)
			}
			got, known := dedicatedVRAMCarveoutFromSysfs(backend, deps)
			want := int64(0)
			if tt.wantKnown {
				want = selectedBytes
			}
			if got != want || known != tt.wantKnown {
				t.Fatalf("got (%d, %v), want (%d, %v)", got, known, want, tt.wantKnown)
			}
		})
	}
}
