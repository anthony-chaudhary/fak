package compute

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// hopJSON renders one PnP parent-chain hop as the query emits it.
func hopJSON(id string, bus, addr int, acpi string) string {
	props := ""
	if bus >= 0 {
		props += `,"DEVPKEY_Device_BusNumber":` + strconv.Itoa(bus) + `,"DEVPKEY_Device_Address":` + strconv.Itoa(addr)
	}
	if acpi != "" {
		props += `,"DEVPKEY_Device_LocationPaths":["PCIROOT(0)",` + jsonString(acpi) + `]`
	}
	return `{"instance_id":` + jsonString(id) + props + `}`
}

// withChain splices a parent_chain array into a pnpFixtureRow object.
func withChain(row string, hops ...string) string {
	return strings.TrimSuffix(row, "}") + `,"parent_chain":[` + strings.Join(hops, ",") + `]}`
}

const (
	fixtureRootComplex = `ACPI\PNP0A08\0`
	fixtureRouterTDM0  = `PCI\VEN_8086&DEV_A73E&SUBSYS_0BDA1028&REV_00&USB4_MS_CM\3&11583659&1&6A`
	fixtureRouterTDM1  = `PCI\VEN_8086&DEV_A76D&SUBSYS_0BDA1028&REV_00&USB4_MS_CM\3&11583659&1&6B`
)

func usb4EnvelopeFixture() string {
	internal := withChain(pnpFixtureRow(fixtureInternalID, "NVIDIA GeForce RTX 4070 Laptop GPU", "oem19.inf", "NVIDIA", 0, 1, fixtureInternalCompat),
		hopJSON(`PCI\VEN_8086&DEV_A70D&SUBSYS_0BDA1028&REV_00\3&11583659&1&08`, 0, 0x10000, "ACPI(_SB_)#ACPI(PC00)#ACPI(PEG1)"),
		hopJSON(fixtureRootComplex, -1, 0, "ACPI(_SB_)#ACPI(PC00)"))
	egpu := withChain(pnpFixtureRow(fixtureEGPU0ID, "NVIDIA GeForce RTX 5070 Ti", "oem308.inf", "NVIDIA", 0, 130, fixtureGPUCompat),
		hopJSON(`PCI\VEN_8086&DEV_5786&SUBSYS_11112222&REV_85\E1BEAA00808714C400`, 129, 0, "ACPI(_SB_)#ACPI(PC00)#ACPI(TRP3)#ACPI(PXSX)#PCI(0000)"),
		hopJSON(`PCI\VEN_8086&DEV_5786&SUBSYS_11112222&REV_85\UE1BEAA00808714C400`, 128, 0, "ACPI(_SB_)#ACPI(PC00)#ACPI(TRP3)#ACPI(PXSX)"),
		hopJSON(`PCI\VEN_8086&DEV_A71F&SUBSYS_0BDA1028&REV_00\3&11583659&1&3B`, 0, 0x70003, "ACPI(_SB_)#ACPI(PC00)#ACPI(TRP3)"),
		hopJSON(fixtureRootComplex, -1, 0, "ACPI(_SB_)#ACPI(PC00)"))
	routers := hopJSON(fixtureRouterTDM0, 0, 0xd0002, "ACPI(_SB_)#ACPI(PC00)#ACPI(TDM0)") + "," +
		hopJSON(fixtureRouterTDM1, 0, 0xd0003, "ACPI(_SB_)#ACPI(PC00)#ACPI(TDM1)")
	return `{"devices":[` + internal + "," + egpu + `],"usb4_host_routers":[` + routers + `]}`
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_ParentChainAndUSB4HostRouter(t *testing.T) {
	rep := windowsBindingReport(t, usb4EnvelopeFixture(), postRebootINFs(t), 0, smiFixture(smiThreeGPUs, nil))
	assertVerdicts(t, rep, GPUReady, GPUReady)
	internal, egpu := rep.Devices[0], rep.Devices[1]
	if internal.RootPort != "00:01.0" || internal.USB4HostRouter != "" {
		t.Fatalf("internal root_port=%q router=%q, want 00:01.0 and none", internal.RootPort, internal.USB4HostRouter)
	}
	var path []string
	for _, h := range egpu.ParentChain {
		path = append(path, h.PCIAddress)
	}
	if want := []string{"81:00.0", "80:00.0", "00:07.3", ""}; !reflect.DeepEqual(path, want) {
		t.Fatalf("eGPU chain = %v, want %v", path, want)
	}
	if egpu.RootPort != "00:07.3" || egpu.USB4HostRouter != fixtureRouterTDM1 {
		t.Fatalf("eGPU root_port=%q router=%q, want 00:07.3 and the TDM1 router", egpu.RootPort, egpu.USB4HostRouter)
	}
	if len(rep.USB4HostRouters) != 2 || rep.USB4HostRouters[1].PCIAddress != "00:0d.3" {
		t.Fatalf("usb4 host routers = %+v, want 2 with TDM1 at 00:0d.3", rep.USB4HostRouters)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestGPUBinding_ACPITailIndex(t *testing.T) {
	for _, c := range []struct {
		path, prefix string
		n            int
		ok           bool
	}{
		{"ACPI(_SB_)#ACPI(PC00)#ACPI(TRP3)", "TRP", 3, true},
		{"ACPI(_SB_)#ACPI(PC00)#ACPI(TDM1)", "TDM", 1, true},
		{"ACPI(_SB_)#ACPI(PC00)#ACPI(TRP3)#ACPI(PXSX)", "TRP", 0, false},
		{"ACPI(_SB_)#ACPI(PC00)#ACPI(PEG1)", "TRP", 0, false},
		{"", "TRP", 0, false},
	} {
		if n, ok := acpiTailIndex(c.path, c.prefix); n != c.n || ok != c.ok {
			t.Errorf("acpiTailIndex(%q, %q) = %d, %v; want %d, %v", c.path, c.prefix, n, ok, c.n, c.ok)
		}
	}
}

func writePackageINF(t *testing.T, dir, name, class string, ids ...string) {
	t.Helper()
	body := strings.Replace(fixtureINFHeader, "Class       = Display", "Class       = "+class, 1)
	for _, id := range ids {
		body += "%NVIDIA_DEV% = Section001, " + id + "\r\n"
	}
	if err := os.MkdirAll(filepath.Join(dir, filepath.Dir(name)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestGPUBinding_DriverPackageRanksMostSpecificINF(t *testing.T) {
	// The 621.01 package shape: the OEM INF lists only the internal GPU, the
	// generic INF lists DEV_2C05, and a board-partner INF lists one SUBSYS.
	pkg := t.TempDir()
	writePackageINF(t, pkg, "Display.Driver/nvdmi.inf", "Display", `PCI\VEN_10DE&DEV_2820&SUBSYS_0BDA1028`)
	writePackageINF(t, pkg, "Display.Driver/nv_dispi.inf", "Display", `PCI\VEN_10DE&DEV_2C05`)
	writePackageINF(t, pkg, "Display.Driver/nvmdi.inf", "Display", `PCI\VEN_10DE&DEV_2C05&SUBSYS_53101462`)
	writePackageINF(t, pkg, "HDAudio/nvhda.inf", "MEDIA", `PCI\VEN_10DE&DEV_2820`)
	const raider = `PCI\VEN_10DE&DEV_2C58&SUBSYS_14D21462&REV_A1\4&2`
	pnp := pnpFixture(
		pnpFixtureRow(fixtureInternalID, "NVIDIA GeForce RTX 4070 Laptop GPU", "oem19.inf", "NVIDIA", 0, 1, fixtureInternalCompat),
		pnpFixtureRow(fixtureEGPU0ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 130, fixtureGPUCompat),
		pnpFixtureRow(fixtureEGPU1ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 88, fixtureGPUCompat),
		pnpFixtureRow(raider, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 2, fixtureGPUCompat),
	)
	rep := diagnoseGPUBindingWindows(GPUBindingOptions{
		INFDir:        postRebootINFs(t),
		PackageINFDir: pkg,
		pnpQuery:      func(context.Context) (string, error) { return pnp, nil },
		smiQuery:      smiFixture(smiInternalOnly, nil),
	})
	cov := rep.Package
	if cov == nil || cov.DisplayINFs != 3 {
		t.Fatalf("package coverage = %+v, want 3 display INFs", cov)
	}
	chosen := map[string]string{}
	for _, d := range cov.Devices {
		chosen[d.PCIAddress] = d.Chosen
	}
	want := map[string]string{
		"01:00.0": "Display.Driver/nvdmi.inf",
		"02:00.0": "",
		"58:00.0": "Display.Driver/nvmdi.inf",
		"82:00.0": "Display.Driver/nv_dispi.inf",
	}
	if !reflect.DeepEqual(chosen, want) {
		t.Fatalf("chosen INFs = %v, want %v", chosen, want)
	}
	var plan []string
	for _, line := range cov.PnputilCommands {
		plan = append(plan, filepath.Base(strings.Trim(strings.Fields(line)[2], `"`)))
	}
	if !reflect.DeepEqual(plan, []string{"nvdmi.inf", "nvmdi.inf", "nv_dispi.inf"}) || !reflect.DeepEqual(cov.Uncovered, []string{"02:00.0"}) {
		t.Fatalf("plan=%v uncovered=%v, want nvdmi/nvmdi/nv_dispi and 02:00.0", plan, cov.Uncovered)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_DriverPackageIsWindowsOnly(t *testing.T) {
	root := t.TempDir()
	writeSysfsGPU(t, root, "0000:58:00.0", fixtureModalias)
	rep := diagnoseGPUBindingLinux(root, GPUBindingOptions{PackageINFDir: t.TempDir(), smiQuery: smiMustNotRun(t)})
	if rep.Package == nil || len(rep.Package.Issues) != 1 || len(rep.Package.Devices) != 0 {
		t.Fatalf("linux package coverage = %+v, want one issue and no devices", rep.Package)
	}
}
