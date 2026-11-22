package telemetry

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// pciFixtureAddr maps a sysfs bus address to a fixture directory name. Windows
// forbids ':' in file names; '.' is stripped by the same address sort.
func pciFixtureAddr(address string) string {
	if runtime.GOOS == "windows" {
		return strings.ReplaceAll(address, ":", ".")
	}
	return address
}

func writePCIFixture(t *testing.T, root, address string, files map[string]string) {
	t.Helper()
	dir := filepath.Join(root, pciFixtureAddr(address))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestNVLibDiscovery pins the go-nvlib PCI identifiers and MIG profile grammar
// and exercises fail-closed sysfs discovery against a fixture tree.
func TestNVLibDiscovery(t *testing.T) {
	t.Run("pinned_identifiers", func(t *testing.T) {
		if PCIDevicesRoot != "/sys/bus/pci/devices" || PCINvidiaVendorID != 0x10de ||
			PCIVgaControllerClass != 0x030000 || PCI3dControllerClass != 0x030200 || PCINvSwitchClass != 0x068000 {
			t.Fatalf("PCI identifiers drifted from upstream nvpci")
		}
		if MIGAttributeMediaExtensions != "me" || MIGAttributeMediaExtensionsAll != "me.all" || MIGAttributeGraphics != "gfx" {
			t.Fatalf("MIG attributes drifted from upstream mig_profile")
		}
	})

	t.Run("discovers_nvidia_sorted", func(t *testing.T) {
		root := t.TempDir()
		writePCIFixture(t, root, "0000:41:00.0", map[string]string{
			"vendor": "0x10de", "class": "0x030200", "device": "0x2c02", "numa_node": "1",
			"subsystem_vendor": "0x10de", "subsystem_device": "0x2054",
		})
		writePCIFixture(t, root, "0000:01:00.0", map[string]string{
			"vendor": "0x10de", "class": "0x030000", "device": "0x2684", "numa_node": "-1",
		})
		writePCIFixture(t, root, "0000:07:00.0", map[string]string{
			"vendor": "0x10de", "class": "0x068000", "device": "0x22a3", "numa_node": "0",
		})
		writePCIFixture(t, root, "0000:03:00.0", map[string]string{
			"vendor": "0x8086", "class": "0x020000", "device": "0x1572", "numa_node": "0",
		})

		all, err := DiscoverNvidiaPCI(root)
		if err != nil {
			t.Fatalf("DiscoverNvidiaPCI: %v", err)
		}
		var addrs []string
		for _, d := range all {
			addrs = append(addrs, d.Address)
		}
		want := []string{pciFixtureAddr("0000:01:00.0"), pciFixtureAddr("0000:07:00.0"), pciFixtureAddr("0000:41:00.0")}
		if !reflect.DeepEqual(addrs, want) {
			t.Fatalf("addresses = %v, want %v", addrs, want)
		}
		vga, sw, gpu3d := all[0], all[1], all[2]
		if !vga.IsVGAController() || !vga.IsGPU() || vga.NumaNode != -1 || vga.SubsystemVendor != 0 {
			t.Fatalf("vga device = %+v", vga)
		}
		if !sw.IsNVSwitch() || sw.IsGPU() {
			t.Fatalf("nvswitch device = %+v", sw)
		}
		if !gpu3d.Is3DController() || gpu3d.Device != 0x2c02 || gpu3d.NumaNode != 1 ||
			gpu3d.SubsystemDevice != 0x2054 || gpu3d.Driver != "" || gpu3d.IommuGroup != -1 {
			t.Fatalf("3d device = %+v", gpu3d)
		}

		gpus, err := DiscoverNvidiaGPUs(root)
		if err != nil || len(gpus) != 2 || gpus[0].Address != want[0] || gpus[1].Address != want[2] {
			t.Fatalf("DiscoverNvidiaGPUs = %+v, %v", gpus, err)
		}
	})

	t.Run("missing_root_refuses", func(t *testing.T) {
		got, err := DiscoverNvidiaPCI(filepath.Join(t.TempDir(), "absent"))
		if !errors.Is(err, ErrNVLibUnavailable) || got != nil {
			t.Fatalf("missing root = %v, %v; want ErrNVLibUnavailable", got, err)
		}
	})

	t.Run("no_nvidia_refuses", func(t *testing.T) {
		root := t.TempDir()
		writePCIFixture(t, root, "0000:03:00.0", map[string]string{"vendor": "0x8086"})
		if got, err := DiscoverNvidiaPCI(root); !errors.Is(err, ErrNVLibUnavailable) || got != nil {
			t.Fatalf("no NVIDIA = %v, %v; want ErrNVLibUnavailable", got, err)
		}
	})

	t.Run("nvswitch_only_refuses_gpus", func(t *testing.T) {
		root := t.TempDir()
		writePCIFixture(t, root, "0000:07:00.0", map[string]string{
			"vendor": "0x10de", "class": "0x068000", "device": "0x22a3", "numa_node": "0",
		})
		if got, err := DiscoverNvidiaGPUs(root); !errors.Is(err, ErrNVLibUnavailable) || got != nil {
			t.Fatalf("nvswitch only = %v, %v; want ErrNVLibUnavailable", got, err)
		}
	})

	t.Run("malformed_nvidia_entry_errors", func(t *testing.T) {
		root := t.TempDir()
		writePCIFixture(t, root, "0000:41:00.0", map[string]string{
			"vendor": "0x10de", "class": "not-hex", "device": "0x2c02", "numa_node": "0",
		})
		got, err := DiscoverNvidiaPCI(root)
		if err == nil || errors.Is(err, ErrNVLibUnavailable) || got != nil {
			t.Fatalf("malformed class = %v, %v; want a parse error, not unavailable", got, err)
		}
	})

	t.Run("mig_profile_grammar", func(t *testing.T) {
		ok := map[string]MIGProfile{
			"1g.5gb":         {C: 1, G: 1, GB: 5},
			"7g.80gb":        {C: 7, G: 7, GB: 80},
			"1c.2g.10gb":     {C: 1, G: 2, GB: 10},
			"1g.10gb+me":     {C: 1, G: 1, GB: 10, Attributes: []string{"me"}},
			"1g.10gb+me.all": {C: 1, G: 1, GB: 10, Attributes: []string{"me.all"}},
			"2g.20gb-me":     {C: 2, G: 2, GB: 20, NegAttributes: []string{"me"}},
			"1g.5gb+me,gfx":  {C: 1, G: 1, GB: 5, Attributes: []string{"me", "gfx"}},
		}
		for s, want := range ok {
			got, err := ParseMIGProfile(s)
			if err != nil {
				t.Fatalf("ParseMIGProfile(%q): %v", s, err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("ParseMIGProfile(%q) = %+v, want %+v", s, got, want)
			}
			if got.String() != s {
				t.Fatalf("ParseMIGProfile(%q).String() = %q", s, got.String())
			}
		}
		// Upstream drops the compute field when it equals the GPU slice count.
		if p, err := ParseMIGProfile("1c.1g.5gb"); err != nil || p.String() != "1g.5gb" {
			t.Fatalf("ParseMIGProfile(1c.1g.5gb) = %v, %v; want String 1g.5gb", p, err)
		}
		for _, bad := range []string{
			"", "1g", "1g.5", "xg.5gb", " 1g.5gb", "1c.2g.10gb.1", "1g.5gb+me-gfx",
			"1g.5gb+", "1g.5gb+me,me", "1g.5gb+1me", "1g.5gb+.me", "1g.5gb+m_e",
		} {
			if _, err := ParseMIGProfile(bad); !errors.Is(err, ErrMIGProfileInvalid) {
				t.Fatalf("ParseMIGProfile(%q): err = %v, want ErrMIGProfileInvalid", bad, err)
			}
		}
	})

	t.Run("mig_memory_size_gb", func(t *testing.T) {
		const mib = 1 << 20
		cases := []struct {
			totalBytes, migMB, want uint64
		}{
			{81559 * mib, 4864, 10},
			{81559 * mib, 19968, 20},
			{40960 * mib, 4864, 5},
			{0, 4864, 0},
		}
		for _, c := range cases {
			if got := MIGMemorySizeGB(c.totalBytes, c.migMB); got != c.want {
				t.Fatalf("MIGMemorySizeGB(%d, %d) = %d, want %d", c.totalBytes, c.migMB, got, c.want)
			}
		}
	})
}
