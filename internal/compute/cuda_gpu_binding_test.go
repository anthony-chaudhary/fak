package compute

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"unicode/utf16"
)

// The Windows fixtures replay the states of a laptop with an internal RTX 4070
// Laptop (DEV_2820, PCI class 0302) and two USB4 RTX 5070 Ti eGPUs (DEV_2C05).

const (
	fixtureInternalID = `PCI\VEN_10DE&DEV_2820&SUBSYS_0BDA1028&REV_A1\4&6A7979E&0&0008`
	fixtureEGPU0ID    = `PCI\VEN_10DE&DEV_2C05&SUBSYS_143C196E&REV_A1\39AFB300CD2DB04800`
	fixtureEGPU1ID    = `PCI\VEN_10DE&DEV_2C05&SUBSYS_53101462&REV_A1\B3657CB1262DB04800`
	fixtureAudioID    = `PCI\VEN_10DE&DEV_22E9&SUBSYS_000010DE&REV_A1\6&AE046EA&0&0100003B`

	fixtureInternalCompat = `["PCI\\CC_030200","PCI\\CC_0302"]`
	fixtureGPUCompat      = `["PCI\\CC_030000","PCI\\CC_0300"]`
	fixtureAudioCompat    = `["PCI\\VEN_10DE&DEV_22E9","PCI\\CC_040300","PCI\\CC_0403"]`

	fixtureINFHeader = "[Version]\r\nSignature = \"$Windows NT$\"\r\nProvider    = %NVIDIA%\r\nClass       = Display\r\n\r\n[NVIDIA_Devices.NTamd64]\r\n"
)

func jsonString(s string) string { return strconv.Quote(s) }

// pnpFixtureRow renders one PnP query row. HardwareIds is a bare string, the
// one-element form PowerShell may emit.
func pnpFixtureRow(instanceID, name, inf, provider string, problem int, bus int, compat string) string {
	infJSON, provJSON := "null", "null"
	if inf != "" {
		infJSON = jsonString(inf)
	}
	if provider != "" {
		provJSON = jsonString(provider)
	}
	hw := instanceID[:strings.LastIndex(instanceID, `\`)]
	return `{"instance_id":` + jsonString(instanceID) + `,"friendly_name":` + jsonString(name) +
		`,"DEVPKEY_Device_ProblemCode":` + strconv.Itoa(problem) +
		`,"DEVPKEY_Device_DriverInfPath":` + infJSON + `,"DEVPKEY_Device_DriverProvider":` + provJSON +
		`,"DEVPKEY_Device_DriverVersion":"32.0.16.2101","DEVPKEY_Device_LocationInfo":"PCI bus ` + strconv.Itoa(bus) + `, device 0, function 0"` +
		`,"DEVPKEY_Device_HardwareIds":` + jsonString(hw) + `,"DEVPKEY_Device_CompatibleIds":` + compat + `}`
}

func pnpFixture(rows ...string) string { return "[" + strings.Join(rows, ",") + "]" }

func writeINF(t *testing.T, dir, name string, ids ...string) {
	t.Helper()
	body := fixtureINFHeader
	for i, id := range ids {
		body += "%NVIDIA_DEV." + strconv.Itoa(i) + "% = Section001, " + id + "\r\n"
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func smiFixture(stdout string, err error) func(context.Context) (gpuQueryOutput, error) {
	return func(context.Context) (gpuQueryOutput, error) { return gpuQueryOutput{Stdout: stdout}, err }
}

func smiExit(stdout, stderr string, code int) func(context.Context) (gpuQueryOutput, error) {
	return func(context.Context) (gpuQueryOutput, error) {
		return gpuQueryOutput{Stdout: stdout, Stderr: stderr, ExitCode: code}, nil
	}
}

func smiMustNotRun(t *testing.T) func(context.Context) (gpuQueryOutput, error) {
	return func(context.Context) (gpuQueryOutput, error) {
		t.Error("nvidia-smi queried although no device has a started NVIDIA driver")
		return gpuQueryOutput{}, errors.New("unexpected")
	}
}

func windowsBindingReport(t *testing.T, pnp, infDir string, expect int, smi func(context.Context) (gpuQueryOutput, error)) GPUBindingReport {
	t.Helper()
	return diagnoseGPUBindingWindows(GPUBindingOptions{
		ExpectGPUs: expect,
		INFDir:     infDir,
		pnpQuery:   func(context.Context) (string, error) { return pnp, nil },
		smiQuery:   smi,
	})
}

func reportVerdicts(rep GPUBindingReport) []GPUBindingVerdict {
	out := make([]GPUBindingVerdict, 0, len(rep.Devices))
	for _, d := range rep.Devices {
		out = append(out, d.Verdict)
	}
	return out
}

func assertVerdicts(t *testing.T, rep GPUBindingReport, want ...GPUBindingVerdict) {
	t.Helper()
	if got := reportVerdicts(rep); !reflect.DeepEqual(got, want) {
		t.Fatalf("verdicts = %v, want %v (status %s)", got, want, rep.Status)
	}
}

const (
	smiInternalOnly = "0, 00000000:01:00.0, GPU-366b3cfa, NVIDIA GeForce RTX 4070 Laptop GPU\n"
	smiThreeGPUs    = smiInternalOnly +
		"1, 00000000:58:00.0, GPU-248188e5, NVIDIA GeForce RTX 5070 Ti\n" +
		"2, 00000000:82:00.0, GPU-a9127ec5, NVIDIA GeForce RTX 5070 Ti\n"
)

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_PreInstallOEMDriverLacksEGPUID(t *testing.T) {
	// OEM laptop driver bound to the internal GPU; both eGPUs on Microsoft Basic
	// Display with problem code 10; the OEM INF lists DEV_2820 only.
	infDir := t.TempDir()
	writeINF(t, infDir, "oem308.inf", `PCI\VEN_10DE&DEV_2820&SUBSYS_0BDA1028`)
	pnp := pnpFixture(
		pnpFixtureRow(fixtureInternalID, "NVIDIA GeForce RTX 4070 Laptop GPU", "oem308.inf", "NVIDIA", 0, 1, fixtureInternalCompat),
		pnpFixtureRow(fixtureEGPU0ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 130, fixtureGPUCompat),
		pnpFixtureRow(fixtureEGPU1ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 88, fixtureGPUCompat),
		pnpFixtureRow(fixtureAudioID, "High Definition Audio Controller", "hdaudbus.inf", "Microsoft", 0, 130, fixtureAudioCompat),
	)
	rep := windowsBindingReport(t, pnp, infDir, 0, smiFixture(smiInternalOnly, nil))
	assertVerdicts(t, rep, GPUReady, GPUDriverLacksDeviceID, GPUDriverLacksDeviceID)
	if rep.Status != CUDADiscoveryObserved || rep.Ready {
		t.Fatalf("status=%s ready=%v, want observed/false", rep.Status, rep.Ready)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_CleanInstallRemovedInternalDriver(t *testing.T) {
	// -clean removed the OEM package; Windows Update's desktop package binds the
	// eGPUs (code 12, not enough resources) and lacks DEV_2820 (code 28).
	infDir := t.TempDir()
	writeINF(t, infDir, "oem400.inf", `PCI\VEN_10DE&DEV_2C05`)
	pnp := pnpFixture(
		pnpFixtureRow(fixtureInternalID, "Video Controller", "", "", 28, 1, fixtureInternalCompat),
		pnpFixtureRow(fixtureEGPU0ID, "NVIDIA GeForce RTX 5070 Ti", "oem400.inf", "NVIDIA", 12, 130, fixtureGPUCompat),
		pnpFixtureRow(fixtureEGPU1ID, "NVIDIA GeForce RTX 5070 Ti", "oem400.inf", "NVIDIA", 12, 88, fixtureGPUCompat),
	)
	rep := windowsBindingReport(t, pnp, infDir, 0, smiMustNotRun(t))
	assertVerdicts(t, rep, GPUDriverLacksDeviceID, GPUUnbound, GPUUnbound)
	for _, d := range rep.Devices[1:] {
		if d.ProblemCode == nil || *d.ProblemCode != 12 {
			t.Fatalf("eGPU problem code = %v, want 12", d.ProblemCode)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_StagedDriverPendingRebootIsUnbound(t *testing.T) {
	pnp := pnpFixture(
		pnpFixtureRow(fixtureInternalID, "NVIDIA GeForce RTX 4070 Laptop GPU", "oem19.inf", "NVIDIA", 31, 1, fixtureInternalCompat),
		pnpFixtureRow(fixtureEGPU0ID, "NVIDIA GeForce RTX 5070 Ti", "oem308.inf", "NVIDIA", 12, 130, fixtureGPUCompat),
		pnpFixtureRow(fixtureEGPU1ID, "NVIDIA GeForce RTX 5070 Ti", "oem327.inf", "NVIDIA", 12, 88, fixtureGPUCompat),
	)
	rep := windowsBindingReport(t, pnp, postRebootINFs(t), 0, smiMustNotRun(t))
	assertVerdicts(t, rep, GPUUnbound, GPUUnbound, GPUUnbound)
	for _, d := range rep.Devices {
		if d.IDListed != GPULayerYes {
			t.Fatalf("%s device_id_listed = %s, want yes", d.PCIAddress, d.IDListed)
		}
	}
}

func postRebootPnP() string {
	return pnpFixture(
		pnpFixtureRow(fixtureInternalID, "NVIDIA GeForce RTX 4070 Laptop GPU", "oem19.inf", "NVIDIA", 0, 1, fixtureInternalCompat),
		pnpFixtureRow(fixtureEGPU0ID, "NVIDIA GeForce RTX 5070 Ti", "oem308.inf", "NVIDIA", 0, 130, fixtureGPUCompat),
		pnpFixtureRow(fixtureEGPU1ID, "NVIDIA GeForce RTX 5070 Ti", "oem327.inf", "NVIDIA", 0, 88, fixtureGPUCompat),
	)
}

func postRebootINFs(t *testing.T) string {
	infDir := t.TempDir()
	writeINF(t, infDir, "oem19.inf", `PCI\VEN_10DE&DEV_2820&SUBSYS_0BDA1028`)
	writeINF(t, infDir, "oem308.inf", `PCI\VEN_10DE&DEV_2C05`)
	writeINF(t, infDir, "oem327.inf", `PCI\VEN_10DE&DEV_2C05&SUBSYS_53101462`)
	return infDir
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_PostRebootAllReady(t *testing.T) {
	rep := windowsBindingReport(t, postRebootPnP(), postRebootINFs(t), 3, smiFixture(smiThreeGPUs, nil))
	assertVerdicts(t, rep, GPUReady, GPUReady, GPUReady)
	if !rep.Ready || rep.Verdicts[GPUReady] != 3 {
		t.Fatalf("ready=%v verdicts=%v, want ready with 3 GPU_READY", rep.Ready, rep.Verdicts)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_BoundButMissingFromNVML(t *testing.T) {
	rep := windowsBindingReport(t, postRebootPnP(), postRebootINFs(t), 0, smiFixture(smiInternalOnly, nil))
	assertVerdicts(t, rep, GPUReady, GPUBoundNotCUDAVisible, GPUBoundNotCUDAVisible)
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_NVMLPartialListingSurvivesExitAndWarnings(t *testing.T) {
	// One GPU fails inside NVML: nvidia-smi still prints the healthy rows (and a
	// warning row) but exits non-zero. Healthy GPUs stay ready.
	stdout := smiInternalOnly + "WARNING: infoROM is corrupted at gpu 0000:58:00.0\n" +
		"1, 00000000:58:00.0, GPU-248188e5, NVIDIA GeForce RTX 5070 Ti\n"
	smi := smiExit(stdout, "Unable to determine the device handle for GPU2: 0000:82:00.0: Unknown Error\n", 15)
	rep := windowsBindingReport(t, postRebootPnP(), postRebootINFs(t), 0, smi)
	assertVerdicts(t, rep, GPUReady, GPUReady, GPUBoundNotCUDAVisible)
	if rep.CUDA.State != GPULayerYes || len(rep.CUDA.Unparsed) != 1 {
		t.Fatalf("cuda state=%s unparsed=%d, want yes with 1 unparsed row", rep.CUDA.State, len(rep.CUDA.Unparsed))
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_NVMLInitFailureIsNotVisible(t *testing.T) {
	rep := windowsBindingReport(t, postRebootPnP(), postRebootINFs(t), 0, smiExit("Failed to initialize NVML: Not Found\n", "", 9))
	assertVerdicts(t, rep, GPUBoundNotCUDAVisible, GPUBoundNotCUDAVisible, GPUBoundNotCUDAVisible)
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_NVMLUnavailableIsUnknownNotReady(t *testing.T) {
	notFound := &exec.Error{Name: "nvidia-smi", Err: exec.ErrNotFound}
	rep := windowsBindingReport(t, postRebootPnP(), postRebootINFs(t), 0, smiFixture("", notFound))
	assertVerdicts(t, rep, GPUBindingUnknown, GPUBindingUnknown, GPUBindingUnknown)
	if rep.Ready || rep.CUDA.State != GPULayerUnknown {
		t.Fatalf("ready=%v cuda=%s, want not ready and unknown NVML", rep.Ready, rep.CUDA.State)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_EnumerationFailureReportsNoDevices(t *testing.T) {
	rep := diagnoseGPUBindingWindows(GPUBindingOptions{
		ExpectGPUs: 3,
		INFDir:     t.TempDir(),
		pnpQuery:   func(context.Context) (string, error) { return "", errors.New("powershell: exit status 1") },
		smiQuery:   smiMustNotRun(t),
	})
	if rep.Status != CUDADiscoveryError || rep.Ready || len(rep.Devices) != 0 {
		t.Fatalf("status=%s ready=%v devices=%d, want error/false/0", rep.Status, rep.Ready, len(rep.Devices))
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_PropertyReadFailureIsUnknown(t *testing.T) {
	// Get-PnpDeviceProperty failed for this device: every DEVPKEY is absent, so
	// an empty INF path is not evidence of "no driver".
	pnp := `[{"instance_id":` + jsonString(fixtureEGPU0ID) + `,"friendly_name":"NVIDIA GeForce RTX 5070 Ti","class":"Display"}]`
	rep := windowsBindingReport(t, pnp, postRebootINFs(t), 0, smiMustNotRun(t))
	assertVerdicts(t, rep, GPUBindingUnknown)
	if rep.Ready {
		t.Fatal("ready with an unobserved binding")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_ExpectedButAbsentIsNotEnumerated(t *testing.T) {
	pnp := pnpFixture(pnpFixtureRow(fixtureInternalID, "NVIDIA GeForce RTX 4070 Laptop GPU", "oem19.inf", "NVIDIA", 0, 1, fixtureInternalCompat))
	rep := windowsBindingReport(t, pnp, postRebootINFs(t), 3, smiFixture(smiInternalOnly, nil))
	assertVerdicts(t, rep, GPUReady, GPUNotEnumerated, GPUNotEnumerated)
	if rep.Ready {
		t.Fatal("ready with two expected GPUs absent")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_SubsystemSpecificINFEntry(t *testing.T) {
	// An INF that lists DEV_2C05 only for one board's subsystem does not cover
	// another board with the same device ID.
	infDir := t.TempDir()
	writeINF(t, infDir, "oem327.inf", `PCI\VEN_10DE&DEV_2C05&SUBSYS_53101462`)
	pnp := pnpFixture(
		pnpFixtureRow(fixtureEGPU0ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 130, fixtureGPUCompat),
		pnpFixtureRow(fixtureEGPU1ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 88, fixtureGPUCompat),
	)
	rep := windowsBindingReport(t, pnp, infDir, 0, smiMustNotRun(t))
	assertVerdicts(t, rep, GPUUnbound, GPUDriverLacksDeviceID)
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_CommentedINFEntryIsNotAListing(t *testing.T) {
	infDir := t.TempDir()
	body := fixtureINFHeader + ";%NVIDIA_DEV.2C05% = Section094, PCI\\VEN_10DE&DEV_2C05\r\n"
	if err := os.WriteFile(filepath.Join(infDir, "oem308.inf"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	pnp := pnpFixture(pnpFixtureRow(fixtureEGPU0ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 130, fixtureGPUCompat))
	rep := windowsBindingReport(t, pnp, infDir, 0, smiMustNotRun(t))
	assertVerdicts(t, rep, GPUDriverLacksDeviceID)
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_UTF16INFIsRead(t *testing.T) {
	infDir := t.TempDir()
	text := fixtureINFHeader + "%NVIDIA_DEV.2C05% = Section001, PCI\\VEN_10DE&DEV_2C05\r\n"
	u := utf16.Encode([]rune(text))
	b := []byte{0xFF, 0xFE}
	for _, c := range u {
		b = append(b, byte(c), byte(c>>8))
	}
	if err := os.WriteFile(filepath.Join(infDir, "oem500.inf"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	pnp := pnpFixture(pnpFixtureRow(fixtureEGPU0ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 130, fixtureGPUCompat))
	rep := windowsBindingReport(t, pnp, infDir, 0, smiMustNotRun(t))
	assertVerdicts(t, rep, GPUUnbound)
}

// failOpenFS refuses to open one file, standing in for an unreadable INF.
type failOpenFS struct {
	fs.FS
	bad string
}

func (f failOpenFS) Open(name string) (fs.File, error) {
	if name == f.bad {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrPermission}
	}
	return f.FS.Open(name)
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_UnreadableVendorINFLeavesCoverageUnknown(t *testing.T) {
	infDir := t.TempDir()
	writeINF(t, infDir, "oem308.inf", `PCI\VEN_10DE&DEV_2820&SUBSYS_0BDA1028`)
	writeINF(t, infDir, "oem999.inf", `PCI\VEN_10DE&DEV_2C05`)
	pnp := pnpFixture(pnpFixtureRow(fixtureEGPU0ID, "Microsoft Basic Display Adapter", "display.inf", "Microsoft", 10, 130, fixtureGPUCompat))
	rep := diagnoseGPUBindingWindows(GPUBindingOptions{
		INFDir:   infDir,
		infFS:    failOpenFS{FS: os.DirFS(infDir), bad: "oem999.inf"},
		pnpQuery: func(context.Context) (string, error) { return pnp, nil },
		smiQuery: smiMustNotRun(t),
	})
	assertVerdicts(t, rep, GPUUnbound)
	if rep.Devices[0].IDListed != GPULayerUnknown || rep.Status != CUDADiscoveryPartial {
		t.Fatalf("device_id_listed=%s status=%s, want unknown/partial", rep.Devices[0].IDListed, rep.Status)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_BusNumberBeatsLocalizedLocation(t *testing.T) {
	row := func(id string, busProps string) string {
		return `{"instance_id":` + jsonString(id) + `,"friendly_name":"NVIDIA GeForce RTX 5070 Ti","DEVPKEY_Device_ProblemCode":0,` +
			`"DEVPKEY_Device_DriverInfPath":"oem308.inf","DEVPKEY_Device_DriverProvider":"NVIDIA",` +
			`"DEVPKEY_Device_LocationInfo":"PCI-Bus 130, Gerät 0, Funktion 0",` + busProps +
			`"DEVPKEY_Device_CompatibleIds":` + fixtureGPUCompat + `}`
	}
	pnp := pnpFixture(row(fixtureEGPU0ID, `"DEVPKEY_Device_BusNumber":130,"DEVPKEY_Device_Address":0,`), row(fixtureEGPU1ID, ""))
	rep := windowsBindingReport(t, pnp, postRebootINFs(t), 0, smiFixture(smiThreeGPUs, nil))
	assertVerdicts(t, rep, GPUBindingUnknown, GPUReady)
	if rep.Devices[1].PCIAddress != "82:00.0" {
		t.Fatalf("pci_address = %q, want 82:00.0", rep.Devices[1].PCIAddress)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBinding_NonDisplayFunctionsDropped(t *testing.T) {
	// Without compatible IDs the class code comes from the hardware IDs, then
	// from the PnP setup class.
	pnp := pnpFixture(
		pnpFixtureRow(fixtureInternalID, "NVIDIA GeForce RTX 4070 Laptop GPU", "oem19.inf", "NVIDIA", 0, 1, fixtureInternalCompat),
		`{"instance_id":`+jsonString(fixtureAudioID)+`,"DEVPKEY_Device_ProblemCode":0,"DEVPKEY_Device_HardwareIds":["PCI\\VEN_10DE&DEV_22E9&SUBSYS_000010DE&REV_A1","PCI\\VEN_10DE&DEV_22E9&CC_040300"]}`,
		`{"instance_id":"PCI\\VEN_10DE&DEV_1AED&SUBSYS_0BDA1028&REV_A1\\4&1","class":"USB","DEVPKEY_Device_ProblemCode":0}`,
	)
	rep := windowsBindingReport(t, pnp, postRebootINFs(t), 0, smiFixture(smiInternalOnly, nil))
	assertVerdicts(t, rep, GPUReady)
}

// fak-test:runtime fast est=5ms lane=default
func TestClassifyGPUBinding_ClosedSet(t *testing.T) {
	code12 := 12
	cases := []struct {
		name string
		dev  GPUBindingDevice
		want GPUBindingVerdict
	}{
		{"ready", GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerYes, DriverStarted: GPULayerYes, CUDAVisible: GPULayerYes}, GPUReady},
		{"absent", GPUBindingDevice{TunnelAuthorized: GPULayerUnknown}, GPUNotEnumerated},
		{"tunnel", GPUBindingDevice{TunnelAuthorized: GPULayerNo}, GPUTunnelUnauthorized},
		{"no driver, id listed", GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerNo, DriverStarted: GPULayerNo, IDListed: GPULayerYes}, GPUUnbound},
		{"no driver, coverage unknown", GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerNo, DriverStarted: GPULayerNo, IDListed: GPULayerUnknown}, GPUUnbound},
		{"bound, problem code", GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerYes, DriverStarted: GPULayerNo, ProblemCode: &code12, IDListed: GPULayerYes}, GPUUnbound},
		{"no driver lists id", GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerNo, DriverStarted: GPULayerYes, IDListed: GPULayerNo}, GPUDriverLacksDeviceID},
		{"bound, NVML blind", GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerYes, DriverStarted: GPULayerYes, CUDAVisible: GPULayerNo}, GPUBoundNotCUDAVisible},
		{"bound, NVML unknown", GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerYes, DriverStarted: GPULayerYes, CUDAVisible: GPULayerUnknown}, GPUBindingUnknown},
		{"provider unknown", GPUBindingDevice{Enumerated: true, VendorDriver: GPULayerUnknown, DriverStarted: GPULayerYes, CUDAVisible: GPULayerYes}, GPUBindingUnknown},
	}
	for _, c := range cases {
		if got := ClassifyGPUBinding(c.dev); got != c.want {
			t.Errorf("%s: verdict = %s, want %s", c.name, got, c.want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestGPUBinding_PCIAddressMatchIgnoresMissingDomain(t *testing.T) {
	for _, c := range []struct {
		a, b string
		want bool
	}{
		{"82:00.0", "00000000:82:00.0", true},
		{"0000:58:00.0", "00000000:58:00.0", true},
		{"0001:58:00.0", "00000000:58:00.0", false},
		{"58:00.0", "00000000:82:00.0", false},
		{"", "00000000:82:00.0", false},
	} {
		if got := samePCIDevice(c.a, c.b); got != c.want {
			t.Errorf("samePCIDevice(%q, %q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

// ---- Linux sysfs fixtures ----

func writeSysfsGPU(t *testing.T, root, bdf, modalias string) string {
	t.Helper()
	dir := filepath.Join(root, "bus", "pci", "devices", strings.ReplaceAll(bdf, ":", "_"))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	attrs := map[string]string{
		"address": bdf, "vendor": "0x10de", "device": "0x2c05", "class": "0x030000",
		"subsystem_vendor": "0x196e", "subsystem_device": "0x143c", "modalias": modalias,
	}
	for k, v := range attrs {
		if err := os.WriteFile(filepath.Join(dir, k), []byte(v+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

const fixtureModalias = "pci:v000010DEd00002C05sv0000196Esd0000143Cbc03sc00i00"

func linuxReportWithAliases(t *testing.T, root, aliases string) GPUBindingReport {
	t.Helper()
	alias := filepath.Join(t.TempDir(), "modules.alias")
	if err := os.WriteFile(alias, []byte(aliases), 0o644); err != nil {
		t.Fatal(err)
	}
	return diagnoseGPUBindingLinux(root, GPUBindingOptions{ModulesAliasPath: alias, smiQuery: smiMustNotRun(t)})
}

// fak-test:runtime fast est=20ms lane=default
func TestGPUBindingLinux_UnboundCoverageFromModulesAlias(t *testing.T) {
	root := t.TempDir()
	writeSysfsGPU(t, root, "0000:58:00.0", fixtureModalias)

	// The nvidia module aliases by display class: that proves nothing per device.
	rep := linuxReportWithAliases(t, root, "alias pci:v000010DEd*sv*sd*bc03sc00i00* nvidia\n")
	assertVerdicts(t, rep, GPUUnbound)
	if rep.Devices[0].IDListed != GPULayerUnknown {
		t.Fatalf("class-wide alias: device_id_listed = %s, want unknown", rep.Devices[0].IDListed)
	}

	rep = linuxReportWithAliases(t, root, "alias pci:v000010DEd00002C05sv*sd*bc*sc*i* nvidia\n")
	assertVerdicts(t, rep, GPUUnbound)
	if rep.Devices[0].IDListed != GPULayerYes {
		t.Fatalf("device-specific alias: device_id_listed = %s, want yes", rep.Devices[0].IDListed)
	}

	// No nvidia module for the running kernel.
	rep = linuxReportWithAliases(t, root, "alias pci:v000010DEd*sv*sd*bc03sc00i00* nouveau\n")
	assertVerdicts(t, rep, GPUDriverLacksDeviceID)

	rep = diagnoseGPUBindingLinux(root, GPUBindingOptions{smiQuery: smiMustNotRun(t)})
	assertVerdicts(t, rep, GPUUnbound)
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBindingLinux_SupportedGPUsJSONDecidesCoverage(t *testing.T) {
	root := t.TempDir()
	writeSysfsGPU(t, root, "0000:58:00.0", fixtureModalias)
	supported := filepath.Join(t.TempDir(), "supported-gpus.json")
	for _, c := range []struct {
		body string
		want GPUBindingVerdict
	}{
		{`{"chips":[{"name":"NVIDIA GeForce RTX 5070 Ti","devid":"0x2C05"}]}`, GPUUnbound},
		{`{"chips":[{"name":"NVIDIA GeForce RTX 5070 Ti","devid":"0x2C05","legacybranch":"470.xx"}]}`, GPUDriverLacksDeviceID},
		{`{"chips":[{"name":"NVIDIA GeForce RTX 4070","devid":"0x2786"}]}`, GPUDriverLacksDeviceID},
	} {
		if err := os.WriteFile(supported, []byte(c.body), 0o644); err != nil {
			t.Fatal(err)
		}
		rep := diagnoseGPUBindingLinux(root, GPUBindingOptions{SupportedGPUsPath: supported, smiQuery: smiMustNotRun(t)})
		assertVerdicts(t, rep, c.want)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBindingLinux_UnauthorizedTunnelExplainsAbsentGPU(t *testing.T) {
	root := t.TempDir()
	writeSysfsGPU(t, root, "0000:01:00.0", fixtureModalias)
	tb := filepath.Join(root, "bus", "thunderbolt", "devices", "0-1")
	if err := os.MkdirAll(tb, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tb, "authorized"), []byte("0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rep := diagnoseGPUBindingLinux(root, GPUBindingOptions{ExpectGPUs: 3, smiQuery: smiMustNotRun(t)})
	assertVerdicts(t, rep, GPUUnbound, GPUTunnelUnauthorized, GPUNotEnumerated)
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBindingLinux_PartialEnumerationClaimsNoAbsentRows(t *testing.T) {
	root := t.TempDir()
	writeSysfsGPU(t, root, "0000:01:00.0", fixtureModalias)
	// A PCI function whose vendor attribute cannot be read may be the missing GPU.
	if err := os.MkdirAll(filepath.Join(root, "bus", "pci", "devices", "0000_02_00.0", "vendor"), 0o755); err != nil {
		t.Fatal(err)
	}
	rep := diagnoseGPUBindingLinux(root, GPUBindingOptions{ExpectGPUs: 2, smiQuery: smiMustNotRun(t)})
	assertVerdicts(t, rep, GPUUnbound)
	if rep.Ready || rep.Status != CUDADiscoveryPartial {
		t.Fatalf("ready=%v status=%s, want not ready/partial", rep.Ready, rep.Status)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGPUBindingLinux_MissingSysfsReportsNoDevices(t *testing.T) {
	rep := diagnoseGPUBindingLinux(filepath.Join(t.TempDir(), "absent"), GPUBindingOptions{ExpectGPUs: 2, smiQuery: smiMustNotRun(t)})
	if rep.Status != CUDADiscoveryError || rep.Ready || len(rep.Devices) != 0 {
		t.Fatalf("status=%s ready=%v devices=%d, want error/false/0", rep.Status, rep.Ready, len(rep.Devices))
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestGPUBindingLinux_DriverSymlinkDecidesBinding(t *testing.T) {
	root := t.TempDir()
	drivers := filepath.Join(root, "bus", "pci", "drivers")
	for _, drv := range []string{"nvidia", "nouveau"} {
		if err := os.MkdirAll(filepath.Join(drivers, drv), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	nv := writeSysfsGPU(t, root, "0000:01:00.0", fixtureModalias)
	nouveau := writeSysfsGPU(t, root, "0000:02:00.0", fixtureModalias)
	if err := os.Symlink(filepath.Join(drivers, "nvidia"), filepath.Join(nv, "driver")); err != nil {
		t.Skipf("directory symlinks need SeCreateSymbolicLinkPrivilege on Windows: %v", err)
	}
	if err := os.Symlink(filepath.Join(drivers, "nouveau"), filepath.Join(nouveau, "driver")); err != nil {
		t.Fatal(err)
	}
	smi := smiFixture("0, 00000000:01:00.0, GPU-1, NVIDIA GeForce RTX 5070 Ti\n", nil)
	rep := diagnoseGPUBindingLinux(root, GPUBindingOptions{smiQuery: smi})
	assertVerdicts(t, rep, GPUReady, GPUUnbound)
}

// ---- Payload floor (driver branch, module flavor, compute capability) ----

func readyFloorRow(driverVersion, flavor string, cc int) GPUBindingDevice {
	return GPUBindingDevice{
		Enumerated: true, VendorDriver: GPULayerYes, DriverStarted: GPULayerYes,
		CUDAVisible: GPULayerYes, DriverVersion: driverVersion, DriverFlavor: flavor,
		ComputeCapability: cc,
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestClassifyGPUBinding_DriverFloorDemotesReadyRows(t *testing.T) {
	cases := []struct {
		name  string
		dev   GPUBindingDevice
		floor GPUDriverFloor
		want  GPUBindingVerdict
	}{
		{"below floor", readyFloorRow("610.43.02", DriverFlavorOpen, 120), BlackwellDriverFloor, GPUDriverBelowFloor},
		{"at floor", readyFloorRow("615.71.09", DriverFlavorOpen, 120), BlackwellDriverFloor, GPUReady},
		{"above floor", readyFloorRow("615.75.01", DriverFlavorOpen, 120), BlackwellDriverFloor, GPUReady},
		{"proprietary flavor", readyFloorRow("615.71.09", DriverFlavorProprietary, 120), BlackwellDriverFloor, GPUModuleFlavorProprietary},
		{"low compute capability", readyFloorRow("615.71.09", DriverFlavorOpen, 90), BlackwellDriverFloor, GPUComputeCapInsufficient},
		{"zero floor keeps legacy behavior", readyFloorRow("610.43.02", DriverFlavorOpen, 90), GPUDriverFloor{}, GPUReady},
		{"unparseable version is unknown not below", readyFloorRow("weird-build", DriverFlavorOpen, 120), BlackwellDriverFloor, GPUReady},
		{"priority: version beats flavor and capability", readyFloorRow("610.43.02", DriverFlavorProprietary, 90), BlackwellDriverFloor, GPUDriverBelowFloor},
		{"priority: flavor beats capability", readyFloorRow("615.71.09", DriverFlavorProprietary, 90), BlackwellDriverFloor, GPUModuleFlavorProprietary},
		{"floor does not mask a binding failure", func() GPUBindingDevice {
			d := readyFloorRow("610.43.02", DriverFlavorOpen, 120)
			d.DriverStarted = GPULayerNo
			return d
		}(), BlackwellDriverFloor, GPUUnbound},
	}
	for _, c := range cases {
		if got := ClassifyGPUBindingWithFloor(c.dev, c.floor); got != c.want {
			t.Errorf("%s: verdict = %s, want %s", c.name, got, c.want)
		}
	}
	// The base classifier is untouched by any floor.
	if got := ClassifyGPUBinding(readyFloorRow("610.43.02", DriverFlavorOpen, 90)); got != GPUReady {
		t.Errorf("base verdict = %s, want GPU_READY (no floor)", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestParseDriverVersionBranches(t *testing.T) {
	for _, c := range []struct {
		in    string
		major int
		minor int
		patch int
		ok    bool
	}{
		{"615.71.09", 615, 71, 9, true},
		{"610.43.02", 610, 43, 2, true},
		{" 615.71.09 ", 615, 71, 9, true},
		{"615.71.09+cuda", 0, 0, 0, false},
		{"615.71", 0, 0, 0, false},
		{"", 0, 0, 0, false},
		{"32.0.16.2101", 32, 0, 16, true},
	} {
		gotMajor, gotMinor, gotPatch, gotOK := ParseDriverVersion(c.in)
		if gotMajor != c.major || gotMinor != c.minor || gotPatch != c.patch || gotOK != c.ok {
			t.Errorf("ParseDriverVersion(%q) = %d.%d.%d ok=%v, want %d.%d.%d ok=%v", c.in, gotMajor, gotMinor, gotPatch, gotOK, c.major, c.minor, c.patch, c.ok)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestParseNvidiaSMIBusIDs_ComputeCapColumn(t *testing.T) {
	devs, unparsed := ParseNvidiaSMIBusIDs("0, 00000000:01:00.0, GPU-1, NVIDIA GeForce RTX 5090, 12.0\n")
	if len(unparsed) != 0 || len(devs) != 1 {
		t.Fatalf("devs=%v unparsed=%v, want one parsed row", devs, unparsed)
	}
	if devs[0].Name != "NVIDIA GeForce RTX 5090" || devs[0].ComputeCapability != 120 {
		t.Fatalf("row = %+v, want name + compute capability 120", devs[0])
	}
	// Legacy four-column rows keep parsing with an unobserved capability.
	old, _ := ParseNvidiaSMIBusIDs(smiInternalOnly)
	if len(old) != 1 || old[0].Name != "NVIDIA GeForce RTX 4070 Laptop GPU" || old[0].ComputeCapability != 0 {
		t.Fatalf("legacy row = %+v, want name kept and capability 0", old)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestGPUBindingReport_FloorDemotesReadyRows(t *testing.T) {
	base := func() GPUBindingInputs {
		return GPUBindingInputs{
			Method:     CUDASourceSysfs,
			ExpectGPUs: 1,
			CUDA: GPUCUDAView{State: GPULayerYes, Devices: []GPUCUDADevice{{
				Index: 0, BusID: "0000:01:00.0", UUID: "GPU-1", Name: "NVIDIA GeForce RTX 5090",
				ComputeCapability: 120,
			}}},
			Devices: []GPUBindingDevice{{
				PCIAddress: "0000:01:00.0", Enumerated: true,
				VendorDriver: GPULayerYes, DriverStarted: GPULayerYes, IDListed: GPULayerYes,
				DriverVersion: "610.43.02", DriverFlavor: DriverFlavorOpen,
			}},
		}
	}
	rep := BuildGPUBindingReport(base())
	assertVerdicts(t, rep, GPUReady)
	if !rep.Ready {
		t.Fatal("no floor: report must stay ready")
	}
	floored := base()
	floored.Floor = BlackwellDriverFloor
	rep = BuildGPUBindingReport(floored)
	assertVerdicts(t, rep, GPUDriverBelowFloor)
	if rep.Ready || rep.Verdicts[GPUDriverBelowFloor] != 1 {
		t.Fatalf("ready=%v verdicts=%v, want not ready with one below-floor row", rep.Ready, rep.Verdicts)
	}
}

// linkDirBestEffort creates a directory link; on Windows it falls back to a
// junction (mklink /J), which needs no SeCreateSymbolicLinkPrivilege.
func linkDirBestEffort(t *testing.T, target, link string) {
	t.Helper()
	err := os.Symlink(target, link)
	if err == nil {
		return
	}
	if runtime.GOOS == "windows" {
		if out, jerr := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput(); jerr == nil {
			return
		} else {
			t.Skipf("symlink/junction unavailable: %v (%s)", jerr, strings.TrimSpace(string(out)))
		}
	}
	t.Skipf("directory symlinks need SeCreateSymbolicLinkPrivilege on Windows: %v", err)
}

// fak-test:runtime fast est=20ms lane=default
func TestGPUBindingLinux_FloorUsesObservedDriverAndNVMLCapability(t *testing.T) {
	root := t.TempDir()
	nv := writeSysfsGPU(t, root, "0000:01:00.0", fixtureModalias)
	drivers := filepath.Join(root, "bus", "pci", "drivers")
	if err := os.MkdirAll(filepath.Join(drivers, "nvidia"), 0o755); err != nil {
		t.Fatal(err)
	}
	linkDirBestEffort(t, filepath.Join(drivers, "nvidia"), filepath.Join(nv, "driver"))
	modDir := filepath.Join(root, "module", "nvidia")
	if err := os.MkdirAll(modDir, 0o755); err != nil {
		t.Fatal(err)
	}
	observedDriver := "610.43.02" // fixture fact the binding report must echo back
	if err := os.WriteFile(filepath.Join(modDir, "version"), []byte(observedDriver+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	linkDirBestEffort(t, modDir, filepath.Join(nv, "driver", "module"))
	smi := smiFixture("0, 00000000:01:00.0, GPU-1, NVIDIA GeForce RTX 5090, 12.0\n", nil)

	// Without a floor the observed driver branch still reads ready.
	rep := diagnoseGPUBindingLinux(root, GPUBindingOptions{smiQuery: smi})
	assertVerdicts(t, rep, GPUReady)
	if rep.Devices[0].DriverVersion != observedDriver || rep.Devices[0].ComputeCapability != 120 {
		t.Fatalf("device = %+v, want observed driver version and NVML capability", rep.Devices[0])
	}

	// With the sm_120 floor the same host is demoted to the typed token.
	rep = diagnoseGPUBindingLinux(root, GPUBindingOptions{smiQuery: smi, Floor: BlackwellDriverFloor})
	assertVerdicts(t, rep, GPUDriverBelowFloor)
	if rep.Ready {
		t.Fatal("floored host must not be ready")
	}
}
