package gpubinding

import (
	"bufio"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

// ---- Linux: sysfs driver symlink, module, ID coverage, TB authorization ----

// linuxSupportedGPUsPaths are where NVIDIA driver installs place supported-gpus.json.
var linuxSupportedGPUsPaths = []string{
	"/usr/share/doc/NVIDIA_GLX-1.0/supported-gpus/supported-gpus.json",
	"/usr/share/nvidia/supported-gpus/supported-gpus.json",
}

// nvidiaSupportedGPUs is the device-ID view of the driver's supported-gpus.json.
// Chips carrying a legacybranch are served only by that older branch.
type nvidiaSupportedGPUs struct {
	Chips []struct {
		DevID        string `json:"devid"`
		LegacyBranch string `json:"legacybranch"`
	} `json:"chips"`
}

// listsDevice reports whether the current branch lists the PCI device ID ("0x2c05").
func (s nvidiaSupportedGPUs) listsDevice(deviceID string) bool {
	want, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(deviceID), "0x"), 16, 16)
	if err != nil {
		return false
	}
	for _, c := range s.Chips {
		got, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(c.DevID), "0x"), 16, 16)
		if err == nil && got == want && c.LegacyBranch == "" {
			return true
		}
	}
	return false
}

// linuxModuleAliases returns the modalias patterns modules.alias assigns to module.
func linuxModuleAliases(text, module string) []string {
	var pats []string
	sc := bufio.NewScanner(strings.NewReader(text))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) == 3 && f[0] == "alias" && f[2] == module && strings.HasPrefix(f[1], "pci:") {
			pats = append(pats, f[1])
		}
	}
	return pats
}

// modaliasDeviceWildcard reports whether a pci: alias pattern leaves the device
// ID open (pci:v000010DEd*sv...): such a class-wide alias matches every NVIDIA
// display device and proves nothing about one device ID.
func modaliasDeviceWildcard(pattern string) bool {
	rest := strings.TrimPrefix(pattern, "pci:v")
	i := strings.IndexByte(rest, 'd')
	if i < 0 {
		return true
	}
	dev := rest[i+1:]
	if j := strings.Index(dev, "sv"); j >= 0 {
		dev = dev[:j]
	}
	return dev == "" || strings.ContainsAny(dev, "*?[")
}

// aliasCoverage judges a modalias against a module's aliases: yes on a
// device-specific match, unknown when only class-wide aliases match, no when
// none match (or the module has no aliases, i.e. is not installed).
func aliasCoverage(modalias string, patterns []string) LayerState {
	state := LayerNo
	for _, p := range patterns {
		if ok, err := path.Match(p, modalias); err != nil || !ok {
			continue
		}
		if !modaliasDeviceWildcard(p) {
			return LayerYes
		}
		state = LayerUnknown
	}
	return state
}

// sysfsReader collects read failures into issues while reading one sysfs tree;
// it replaces the discovery orchestrator's per-device attribute reader so this
// package stays self-contained.
type sysfsReader struct {
	issues []Issue
}

func (r *sysfsReader) addIssue(p string, err error) {
	r.issues = append(r.issues, Issue{Path: filepath.ToSlash(p), Error: err.Error()})
}

// attr reads one trimmed sysfs attribute. A missing attribute returns ok=false
// with no issue; any other read failure is recorded as an issue.
func (r *sysfsReader) attr(devDir, name string) (string, bool) {
	p := filepath.Join(devDir, name)
	data, err := os.ReadFile(p)
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			r.addIssue(p, err)
		}
		return "", false
	}
	return strings.TrimSpace(string(data)), true
}

func diagnoseLinux(sysfsRoot string, opts Options) Report {
	rd := &sysfsReader{}
	pciDevicesDir := filepath.Join(sysfsRoot, "bus", "pci", "devices")
	entries, err := os.ReadDir(pciDevicesDir)
	if err != nil {
		rd.addIssue(pciDevicesDir, err)
		return enumerationFailed(SourceSysfs, opts.ExpectGPUs, rd.issues)
	}
	live := sysfsRoot == "/sys"

	var supported *nvidiaSupportedGPUs
	supportedPaths := linuxSupportedGPUsPaths
	if opts.SupportedGPUsPath != "" {
		supportedPaths = []string{opts.SupportedGPUsPath}
	} else if !live {
		supportedPaths = nil
	}
	for _, p := range supportedPaths {
		b, err := os.ReadFile(p)
		if err != nil {
			if !errors.Is(err, fs.ErrNotExist) {
				rd.addIssue(p, err)
			}
			continue
		}
		var s nvidiaSupportedGPUs
		if err := json.Unmarshal(b, &s); err != nil {
			rd.addIssue(p, err)
			continue
		}
		supported = &s
		break
	}

	aliasLoaded, nvidiaAliases := false, []string(nil)
	aliasPath := opts.ModulesAliasPath
	if aliasPath == "" && live {
		if rel, err := os.ReadFile("/proc/sys/kernel/osrelease"); err == nil {
			aliasPath = filepath.Join("/lib/modules", strings.TrimSpace(string(rel)), "modules.alias")
		}
	}
	if aliasPath != "" {
		if b, err := os.ReadFile(aliasPath); err == nil {
			nvidiaAliases = linuxModuleAliases(string(b), "nvidia")
			aliasLoaded = true
		} else if !errors.Is(err, fs.ErrNotExist) {
			rd.addIssue(aliasPath, err)
		}
	}

	incomplete := false
	var devs []Device
	for _, entry := range entries {
		devDir := filepath.Join(pciDevicesDir, entry.Name())
		before := len(rd.issues)
		vendor, ok := rd.attr(devDir, "vendor")
		class, _ := rd.attr(devDir, "class")
		if len(rd.issues) > before {
			incomplete = true
		}
		if !ok || !strings.EqualFold(vendor, "0x10de") {
			continue
		}
		if class != "" && !strings.HasPrefix(strings.ToLower(class), "0x03") {
			continue
		}
		dev := Device{
			PCIAddress: strings.ReplaceAll(entry.Name(), "_", ":"),
			VendorID:   strings.ToLower(vendor),
			Source:     SourceSysfs,
		}
		if addr, ok := rd.attr(devDir, "address"); ok && addr != "" {
			dev.PCIAddress = addr
		}
		if resolved, err := filepath.EvalSymlinks(devDir); err == nil {
			dev.ParentChain, dev.RootPort = linuxParentChain(resolved)
		}
		dev.DeviceID, _ = rd.attr(devDir, "device")
		dev.DeviceID = strings.ToLower(dev.DeviceID)
		sv, _ := rd.attr(devDir, "subsystem_vendor")
		sd, _ := rd.attr(devDir, "subsystem_device")
		if sv != "" && sd != "" {
			dev.SubsystemID = strings.ToLower(strings.TrimPrefix(sd, "0x") + strings.TrimPrefix(sv, "0x"))
		}
		dev.Name, _ = rd.attr(devDir, "label")
		dev.modalias, _ = rd.attr(devDir, "modalias")

		var unknown []string
		link, err := os.Readlink(filepath.Join(devDir, "driver"))
		switch {
		case err == nil:
			dev.Driver = path.Base(filepath.ToSlash(link))
			dev.DriverStarted = LayerYes
			dev.VendorDriver = LayerNo
			if dev.Driver == "nvidia" {
				dev.VendorDriver = LayerYes
			}
			if mod, err := os.Readlink(filepath.Join(devDir, "driver", "module")); err == nil {
				dev.DriverModule = path.Base(filepath.ToSlash(mod))
				dev.DriverVersion, _ = rd.attr(filepath.Join(sysfsRoot, "module", dev.DriverModule), "version")
			}
		case errors.Is(err, fs.ErrNotExist):
			dev.VendorDriver, dev.DriverStarted = LayerNo, LayerNo
		default:
			rd.addIssue(filepath.Join(devDir, "driver"), err)
			dev.VendorDriver, dev.DriverStarted = LayerUnknown, LayerUnknown
			unknown = append(unknown, "driver")
		}

		dev.IDListed = LayerUnknown
		switch {
		case supported != nil && dev.DeviceID != "":
			dev.IDCheckedIn = []string{"supported-gpus.json"}
			dev.IDListed = LayerNo
			if supported.listsDevice(dev.DeviceID) {
				dev.IDListed = LayerYes
				dev.IDListedBy = dev.IDCheckedIn
			}
		case aliasLoaded && dev.modalias != "":
			dev.IDCheckedIn = []string{"modules.alias:nvidia"}
			if dev.IDListed = aliasCoverage(dev.modalias, nvidiaAliases); dev.IDListed == LayerYes {
				dev.IDListedBy = dev.IDCheckedIn
			}
		}
		if dev.IDListed == LayerUnknown {
			unknown = append(unknown, "device_id_listed")
		}
		dev.UnknownFields = unknown
		devs = append(devs, dev)
	}

	var pkg *DriverPackage
	if opts.PackageINFDir != "" {
		pkg = &DriverPackage{Dir: opts.PackageINFDir, Issues: []Issue{{Path: filepath.ToSlash(opts.PackageINFDir), Error: "package coverage matches Windows PnP hardware IDs; sysfs devices carry none"}}}
	}
	return BuildReport(Inputs{
		Package:               pkg,
		Method:                SourceSysfs,
		Devices:               devs,
		CUDA:                  cudaViewIfNeeded(devs, opts.smiQuery),
		ExpectGPUs:            opts.ExpectGPUs,
		UnauthorizedTunnels:   linuxUnauthorizedTunnels(rd, sysfsRoot),
		Issues:                rd.issues,
		EnumerationIncomplete: incomplete,
		Floor:                 opts.Floor,
	})
}

// linuxUnauthorizedTunnels lists Thunderbolt/USB4 devices whose sysfs
// authorized attribute is 0; a PCIe device behind one cannot enumerate.
func linuxUnauthorizedTunnels(rd *sysfsReader, sysfsRoot string) []string {
	dir := filepath.Join(sysfsRoot, "bus", "thunderbolt", "devices")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			rd.addIssue(dir, err)
		}
		return nil
	}
	var out []string
	for _, e := range entries {
		if auth, ok := rd.attr(filepath.Join(dir, e.Name()), "authorized"); ok && auth == "0" {
			out = append(out, e.Name())
		}
	}
	return out
}
