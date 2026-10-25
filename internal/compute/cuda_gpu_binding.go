package compute

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf16"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// GPUBindingVerdict is the closed per-device outcome of the GPU binding doctor,
// which compares four views of one NVIDIA PCI display device: PCI enumeration,
// driver binding, whether the vendor driver lists the device ID, and NVML
// visibility. A GPU missing from nvidia-smi can fail at any of these layers and
// the fix differs per layer.
type GPUBindingVerdict string

const (
	// GPUReady means enumerated, bound to the started NVIDIA driver, and NVML-visible.
	GPUReady GPUBindingVerdict = "GPU_READY"
	// GPUNotEnumerated means an expected GPU is absent from a complete PCI enumeration.
	GPUNotEnumerated GPUBindingVerdict = "GPU_NOT_ENUMERATED"
	// GPUUnbound means enumerated but the NVIDIA driver is not bound or not
	// started (Windows carries the problem code), while an installed driver lists
	// the ID or coverage is unknown.
	GPUUnbound GPUBindingVerdict = "GPU_UNBOUND"
	// GPUDriverLacksDeviceID means enumerated and not driven, and no installed
	// NVIDIA INF/driver lists the device ID (including none installed): only a
	// package that lists the ID can bind it.
	GPUDriverLacksDeviceID GPUBindingVerdict = "GPU_DRIVER_LACKS_DEVICE_ID"
	// GPUBoundNotCUDAVisible means bound to the started NVIDIA driver but absent from NVML.
	GPUBoundNotCUDAVisible GPUBindingVerdict = "GPU_BOUND_NOT_CUDA_VISIBLE"
	// GPUTunnelUnauthorized means an expected GPU is absent while a Thunderbolt/USB4
	// device on the host is not authorized (Linux sysfs authorized=0, as boltctl shows).
	GPUTunnelUnauthorized GPUBindingVerdict = "GPU_TUNNEL_UNAUTHORIZED"
	// GPUBindingUnknown means a layer the verdict depends on was not observed.
	GPUBindingUnknown GPUBindingVerdict = "unknown"
)

// GPULayerState is a tri-state observation of one layer; unknown is never
// collapsed into yes or no.
type GPULayerState string

const (
	GPULayerYes     GPULayerState = "yes"
	GPULayerNo      GPULayerState = "no"
	GPULayerUnknown GPULayerState = "unknown"
)

// Binding-doctor provenance labels.
const (
	CUDASourceWindowsPnP = "windows_pnp"
	CUDASourceNvidiaSMI  = "nvidia_smi"
	gpuBindingExpected   = "expected"
)

// GPUBindingDevice is one NVIDIA PCI display device (or one expected but absent
// GPU) with every layer observation that decides its verdict.
type GPUBindingDevice struct {
	PCIAddress       string            `json:"pci_address,omitempty"`
	InstanceID       string            `json:"instance_id,omitempty"`
	VendorID         string            `json:"vendor_id,omitempty"`
	DeviceID         string            `json:"device_id,omitempty"`
	SubsystemID      string            `json:"subsystem_id,omitempty"`
	Name             string            `json:"name,omitempty"`
	Enumerated       bool              `json:"enumerated"`
	Driver           string            `json:"driver,omitempty"`
	DriverModule     string            `json:"driver_module,omitempty"`
	DriverProvider   string            `json:"driver_provider,omitempty"`
	DriverVersion    string            `json:"driver_version,omitempty"`
	VendorDriver     GPULayerState     `json:"vendor_driver"`
	DriverStarted    GPULayerState     `json:"driver_started"`
	ProblemCode      *int              `json:"problem_code,omitempty"`
	ProblemName      string            `json:"problem_name,omitempty"`
	IDListed         GPULayerState     `json:"device_id_listed"`
	IDListedBy       []string          `json:"device_id_listed_by,omitempty"`
	IDCheckedIn      []string          `json:"device_id_checked_in,omitempty"`
	CUDAVisible      GPULayerState     `json:"cuda_visible"`
	CUDAIndex        *int              `json:"cuda_index,omitempty"`
	CUDAUUID         string            `json:"cuda_uuid,omitempty"`
	TunnelAuthorized GPULayerState     `json:"tunnel_authorized,omitempty"`
	Tunnel           string            `json:"tunnel,omitempty"`
	Source           string            `json:"source,omitempty"`
	Verdict          GPUBindingVerdict `json:"verdict"`
	UnknownFields    []string          `json:"unknown_fields,omitempty"`

	// hardwareIDs are the upper-cased PnP hardware/compatible IDs that carry a
	// device ID; an INF entry equal to one of them lists this device.
	hardwareIDs []string
	// modalias is the Linux PCI modalias matched against modules.alias.
	modalias string
}

// GPUCUDADevice is one GPU as NVML (nvidia-smi) reports it.
type GPUCUDADevice struct {
	Index int    `json:"index"`
	BusID string `json:"pci_bus_id"`
	UUID  string `json:"uuid,omitempty"`
	Name  string `json:"name,omitempty"`
}

// GPUCUDAView is the NVML layer: State is yes when nvidia-smi listed devices
// (even if it then exited non-zero), no when it ran and listed none, unknown
// when it could not run, its output did not parse, or it was not needed
// because no device had a started driver.
type GPUCUDAView struct {
	State    GPULayerState   `json:"state"`
	Source   string          `json:"source,omitempty"`
	Error    string          `json:"error,omitempty"`
	Devices  []GPUCUDADevice `json:"devices,omitempty"`
	Unparsed []string        `json:"unparsed_rows,omitempty"`
}

// GPUBindingReport is the binding doctor's host-level result.
type GPUBindingReport struct {
	Status     CUDADiscoveryStatus       `json:"status"`
	Method     string                    `json:"method,omitempty"`
	Issues     []CUDADiscoveryIssue      `json:"issues,omitempty"`
	ExpectGPUs int                       `json:"expect_gpus,omitempty"`
	CUDA       GPUCUDAView               `json:"cuda"`
	Devices    []GPUBindingDevice        `json:"devices"`
	Verdicts   map[GPUBindingVerdict]int `json:"verdicts"`
	Ready      bool                      `json:"ready"`
}

// GPUBindingInputs are the collected observations BuildGPUBindingReport folds.
type GPUBindingInputs struct {
	Method              string
	Devices             []GPUBindingDevice
	CUDA                GPUCUDAView
	ExpectGPUs          int
	UnauthorizedTunnels []string
	Issues              []CUDADiscoveryIssue
	// EnumerationIncomplete means some PCI function could not be read, so a
	// missing expected GPU may be that function: no absent rows are claimed.
	EnumerationIncomplete bool
}

// GPUBindingOptions tunes DiagnoseGPUBinding; zero values select host defaults.
type GPUBindingOptions struct {
	// ExpectGPUs adds GPU_NOT_ENUMERATED / GPU_TUNNEL_UNAUTHORIZED rows when a
	// complete enumeration finds fewer NVIDIA display devices.
	ExpectGPUs int
	// ModulesAliasPath is the Linux modules.alias; empty derives
	// /lib/modules/<release>/modules.alias for the live /sys only.
	ModulesAliasPath string
	// SupportedGPUsPath is the NVIDIA driver's supported-gpus.json; empty tries
	// the driver's install locations for the live /sys only.
	SupportedGPUsPath string
	// INFDir is the Windows INF directory; empty means %SystemRoot%\INF.
	INFDir string

	pnpQuery func(ctx context.Context) (string, error)
	smiQuery func(ctx context.Context) (gpuQueryOutput, error)
	infFS    fs.FS
}

// gpuQueryOutput is one external query's result; the query's error is reserved
// for "could not run", so a non-zero exit is data, not a failure to observe.
type gpuQueryOutput struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

// gpuBindingQueryTimeout bounds each external query (PowerShell PnP, nvidia-smi).
const gpuBindingQueryTimeout = 60 * time.Second

// ClassifyGPUBinding folds one device's layer observations into its verdict.
func ClassifyGPUBinding(dev GPUBindingDevice) GPUBindingVerdict {
	if !dev.Enumerated {
		if dev.TunnelAuthorized == GPULayerNo {
			return GPUTunnelUnauthorized
		}
		return GPUNotEnumerated
	}
	if dev.VendorDriver == GPULayerNo || dev.DriverStarted == GPULayerNo {
		if dev.IDListed == GPULayerNo {
			return GPUDriverLacksDeviceID
		}
		return GPUUnbound
	}
	if dev.VendorDriver != GPULayerYes || dev.DriverStarted != GPULayerYes {
		return GPUBindingUnknown
	}
	switch dev.CUDAVisible {
	case GPULayerYes:
		return GPUReady
	case GPULayerNo:
		return GPUBoundNotCUDAVisible
	}
	return GPUBindingUnknown
}

// gpuDriverStarted reports whether a device has a started NVIDIA driver, the
// only case in which NVML visibility decides the verdict.
func gpuDriverStarted(dev GPUBindingDevice) bool {
	return dev.VendorDriver == GPULayerYes && dev.DriverStarted == GPULayerYes
}

// BuildGPUBindingReport joins the enumerated devices with the NVML view, adds
// rows for expected but absent GPUs after a complete enumeration
// (tunnel-unauthorized first, one per unauthorized tunnel device), and
// classifies every row. Status follows the topology discovery rule: zero
// devices with read issues is an error, never absence. Ready needs every row
// GPU_READY and at least ExpectGPUs devices enumerated.
func BuildGPUBindingReport(in GPUBindingInputs) GPUBindingReport {
	rep := GPUBindingReport{
		Method:     in.Method,
		Issues:     in.Issues,
		ExpectGPUs: in.ExpectGPUs,
		CUDA:       in.CUDA,
		Devices:    make([]GPUBindingDevice, 0, len(in.Devices)),
		Verdicts:   map[GPUBindingVerdict]int{},
	}
	for _, dev := range in.Devices {
		dev.Enumerated = true
		dev.CUDAVisible, dev.CUDAIndex, dev.CUDAUUID = matchCUDAView(dev.PCIAddress, in.CUDA)
		if dev.CUDAVisible == GPULayerUnknown {
			dev.UnknownFields = appendUnique(dev.UnknownFields, "cuda_visible")
		}
		rep.Devices = append(rep.Devices, dev)
	}
	sort.SliceStable(rep.Devices, func(i, j int) bool { return rep.Devices[i].PCIAddress < rep.Devices[j].PCIAddress })
	enumerated := len(rep.Devices)
	if !in.EnumerationIncomplete {
		for i := 0; i < in.ExpectGPUs-enumerated; i++ {
			missing := GPUBindingDevice{
				VendorDriver:     GPULayerUnknown,
				DriverStarted:    GPULayerUnknown,
				IDListed:         GPULayerUnknown,
				CUDAVisible:      GPULayerUnknown,
				TunnelAuthorized: GPULayerUnknown,
				Source:           gpuBindingExpected,
			}
			if i < len(in.UnauthorizedTunnels) {
				missing.TunnelAuthorized = GPULayerNo
				missing.Tunnel = in.UnauthorizedTunnels[i]
			}
			rep.Devices = append(rep.Devices, missing)
		}
	}
	rep.Ready = len(rep.Devices) > 0 && enumerated >= in.ExpectGPUs
	for i := range rep.Devices {
		v := ClassifyGPUBinding(rep.Devices[i])
		rep.Devices[i].Verdict = v
		rep.Verdicts[v]++
		if v != GPUReady {
			rep.Ready = false
		}
	}
	switch {
	case enumerated > 0 && len(in.Issues) > 0:
		rep.Status = CUDADiscoveryPartial
	case enumerated > 0:
		rep.Status = CUDADiscoveryObserved
	case len(in.Issues) > 0:
		rep.Status = CUDADiscoveryError
	default:
		rep.Status = CUDADiscoveryAbsent
	}
	if rep.Status == CUDADiscoveryError {
		rep.Ready = false
	}
	return rep
}

func matchCUDAView(addr string, cuda GPUCUDAView) (GPULayerState, *int, string) {
	switch cuda.State {
	case GPULayerNo:
		return GPULayerNo, nil, ""
	case GPULayerYes:
	default:
		return GPULayerUnknown, nil, ""
	}
	if addr == "" {
		return GPULayerUnknown, nil, ""
	}
	for _, c := range cuda.Devices {
		if samePCIDevice(addr, c.BusID) {
			idx := c.Index
			return GPULayerYes, &idx, c.UUID
		}
	}
	return GPULayerNo, nil, ""
}

// pciBusKey splits a PCI address into its domain (leading zeros trimmed, "" when
// absent) and its lower-cased bus:dev.fn part, so nvidia-smi's
// "00000000:58:00.0", sysfs "0000:58:00.0" and Windows-derived "58:00.0" agree.
func pciBusKey(addr string) (domain, bdf string) {
	parts := strings.Split(strings.ToLower(strings.TrimSpace(addr)), ":")
	switch len(parts) {
	case 3:
		domain = strings.TrimLeft(parts[0], "0")
		if domain == "" {
			domain = "0"
		}
		return domain, parts[1] + ":" + parts[2]
	case 2:
		return "", parts[0] + ":" + parts[1]
	}
	return "", ""
}

func samePCIDevice(a, b string) bool {
	da, ba := pciBusKey(a)
	db, bb := pciBusKey(b)
	if ba == "" || ba != bb {
		return false
	}
	return da == "" || db == "" || da == db
}

// ParseNvidiaSMIBusIDs parses `nvidia-smi --query-gpu=index,pci.bus_id,uuid,name
// --format=csv,noheader` output. A row that is not a GPU record (a warning, an
// NVML error line) is returned in unparsed instead of voiding the other rows.
func ParseNvidiaSMIBusIDs(out string) (devs []GPUCUDADevice, unparsed []string) {
	sc := bufio.NewScanner(strings.NewReader(out))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) < 3 {
			unparsed = append(unparsed, line)
			continue
		}
		idx, err := strconv.Atoi(strings.TrimSpace(f[0]))
		_, bdf := pciBusKey(f[1])
		if err != nil || bdf == "" {
			unparsed = append(unparsed, line)
			continue
		}
		d := GPUCUDADevice{Index: idx, BusID: strings.TrimSpace(f[1]), UUID: strings.TrimSpace(f[2])}
		if len(f) > 3 {
			d.Name = strings.TrimSpace(strings.Join(f[3:], ","))
		}
		devs = append(devs, d)
	}
	return devs, unparsed
}

// nvidiaSMIBindingArgs is the NVML query the doctor runs.
var nvidiaSMIBindingArgs = []string{"--query-gpu=index,pci.bus_id,uuid,name", "--format=csv,noheader"}

func execNvidiaSMIBinding(ctx context.Context) (gpuQueryOutput, error) {
	cmd := exec.CommandContext(ctx, "nvidia-smi", nvidiaSMIBindingArgs...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	res := gpuQueryOutput{Stdout: string(out), Stderr: stderr.String()}
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && ctx.Err() == nil {
			res.ExitCode = exitErr.ExitCode()
			return res, nil
		}
		return res, err
	}
	return res, nil
}

// collectCUDAView runs the NVML query. Rows nvidia-smi printed count even when
// it exits non-zero; a non-zero exit with no GPU row (for example "Failed to
// initialize NVML: Not Found") observes that NVML sees no GPU; a query that
// cannot start or whose output does not parse leaves the layer unknown.
func collectCUDAView(query func(ctx context.Context) (gpuQueryOutput, error)) GPUCUDAView {
	view := GPUCUDAView{State: GPULayerUnknown, Source: CUDASourceNvidiaSMI}
	ctx, cancel := context.WithTimeout(context.Background(), gpuBindingQueryTimeout)
	defer cancel()
	res, err := query(ctx)
	if err != nil {
		view.Error = err.Error()
		return view
	}
	view.Devices, view.Unparsed = ParseNvidiaSMIBusIDs(res.Stdout)
	exitNote := fmt.Sprintf("nvidia-smi exit %d", res.ExitCode)
	switch {
	case len(view.Devices) > 0:
		view.State = GPULayerYes
		if res.ExitCode != 0 {
			view.Error = exitNote + ": " + firstNonEmptyLine(res.Stderr+"\n"+strings.Join(view.Unparsed, "\n"), "partial listing")
		}
	case res.ExitCode != 0:
		view.State = GPULayerNo
		view.Error = exitNote + ": " + firstNonEmptyLine(res.Stdout+"\n"+res.Stderr, "no output")
	case len(view.Unparsed) > 0:
		view.Error = "nvidia-smi output not parseable"
	default:
		view.State = GPULayerNo
		view.Error = "nvidia-smi listed no GPU"
	}
	return view
}

func firstNonEmptyLine(out, fallback string) string {
	for _, l := range strings.Split(out, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			return l
		}
	}
	return fallback
}

// DiagnoseGPUBinding runs the binding doctor with the discovery's platform
// selection: Linux sysfs when a sysfs root is set or the host is Linux, Windows
// PnP on Windows, otherwise an unsupported report. It never substitutes a
// representative device; an unobserved layer stays unknown.
func (d *CUDAPCIeTopologyDiscovery) DiagnoseGPUBinding(opts GPUBindingOptions) GPUBindingReport {
	if opts.smiQuery == nil {
		opts.smiQuery = execNvidiaSMIBinding
	}
	switch {
	case d.SysfsRoot != "" || runtime.GOOS == "linux":
		root := d.SysfsRoot
		if root == "" {
			root = "/sys"
		}
		return diagnoseGPUBindingLinux(root, opts)
	case runtime.GOOS == "windows":
		return diagnoseGPUBindingWindows(opts)
	}
	// No platform method: topology discovery owns the unsupported status.
	_ = d.Discover()
	rep := gpuBindingEnumerationFailed(d.Method, opts.ExpectGPUs, nil)
	rep.Status = d.Status
	return rep
}

// gpuBindingEnumerationFailed reports an enumeration that did not complete: it
// carries no rows, not even expected-but-absent ones, since absence was not observed.
func gpuBindingEnumerationFailed(method string, expect int, issues []CUDADiscoveryIssue) GPUBindingReport {
	return BuildGPUBindingReport(GPUBindingInputs{
		Method:                method,
		CUDA:                  GPUCUDAView{State: GPULayerUnknown},
		ExpectGPUs:            expect,
		Issues:                issues,
		EnumerationIncomplete: true,
	})
}

// cudaViewIfNeeded queries NVML only when some device has a started NVIDIA
// driver; otherwise NVML cannot change any verdict.
func cudaViewIfNeeded(devs []GPUBindingDevice, query func(ctx context.Context) (gpuQueryOutput, error)) GPUCUDAView {
	for _, dev := range devs {
		if gpuDriverStarted(dev) {
			return collectCUDAView(query)
		}
	}
	return GPUCUDAView{State: GPULayerUnknown, Source: CUDASourceNvidiaSMI, Error: "not queried: no device has a started NVIDIA driver"}
}

// ---- Windows: PnP problem code, bound INF, INF device-ID coverage ----

// windowsPnPBindingScript lists present NVIDIA PCI functions with the device
// properties the doctor reads. A failed enumeration must exit non-zero, never
// print an empty list; a failed per-device property read leaves that
// property null (unknown). -InputObject keeps a one-device result an array.
const windowsPnPBindingScript = `$ErrorActionPreference='Stop'; ` +
	`$k='DEVPKEY_Device_ProblemCode','DEVPKEY_Device_DriverInfPath','DEVPKEY_Device_DriverVersion','DEVPKEY_Device_DriverProvider','DEVPKEY_Device_BusNumber','DEVPKEY_Device_Address','DEVPKEY_Device_LocationInfo','DEVPKEY_Device_HardwareIds','DEVPKEY_Device_CompatibleIds','DEVPKEY_Device_DeviceDesc'; ` +
	`$r=@(Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'PCI\VEN_10DE*' } | ForEach-Object { $id=$_.InstanceId; $o=[ordered]@{instance_id=$id; friendly_name=$_.FriendlyName; class=$_.Class}; foreach($p in (Get-PnpDeviceProperty -InstanceId $id -KeyName $k -ErrorAction SilentlyContinue)){ $o[$p.KeyName]=$p.Data }; [pscustomobject]$o }); ` +
	`ConvertTo-Json -InputObject $r -Compress -Depth 4`

func execWindowsPnPBinding(ctx context.Context) (string, error) {
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", windowsPnPBindingScript)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	return string(out), err
}

// psStrings accepts a PowerShell string-list property serialized either as a
// JSON array or, for one element, a bare string.
type psStrings []string

func (s *psStrings) UnmarshalJSON(b []byte) error {
	var list []string
	if err := json.Unmarshal(b, &list); err == nil {
		*s = list
		return nil
	}
	var one string
	if err := json.Unmarshal(b, &one); err != nil {
		return err
	}
	*s = psStrings{one}
	return nil
}

type winPnPBindingRaw struct {
	InstanceID    string    `json:"instance_id"`
	FriendlyName  string    `json:"friendly_name"`
	Class         string    `json:"class"`
	ProblemCode   *int      `json:"DEVPKEY_Device_ProblemCode"`
	DriverInfPath string    `json:"DEVPKEY_Device_DriverInfPath"`
	DriverVersion string    `json:"DEVPKEY_Device_DriverVersion"`
	Provider      string    `json:"DEVPKEY_Device_DriverProvider"`
	BusNumber     *uint32   `json:"DEVPKEY_Device_BusNumber"`
	Address       *uint32   `json:"DEVPKEY_Device_Address"`
	LocationInfo  string    `json:"DEVPKEY_Device_LocationInfo"`
	HardwareIDs   psStrings `json:"DEVPKEY_Device_HardwareIds"`
	CompatibleIDs psStrings `json:"DEVPKEY_Device_CompatibleIds"`
	DeviceDesc    string    `json:"DEVPKEY_Device_DeviceDesc"`
}

// windowsProblemNames names the CM_PROB_* codes seen in GPU bring-up.
var windowsProblemNames = map[int]string{
	1:  "CM_PROB_NOT_CONFIGURED",
	3:  "CM_PROB_OUT_OF_MEMORY",
	10: "CM_PROB_FAILED_START",
	12: "CM_PROB_NORMAL_CONFLICT",
	14: "CM_PROB_NEED_RESTART",
	18: "CM_PROB_REINSTALL",
	22: "CM_PROB_DISABLED",
	24: "CM_PROB_DEVICE_NOT_THERE",
	28: "CM_PROB_FAILED_INSTALL",
	31: "CM_PROB_FAILED_ADD",
	37: "CM_PROB_FAILED_DRIVER_ENTRY",
	39: "CM_PROB_DRIVER_FAILED_LOAD",
	43: "CM_PROB_FAILED_POST_START",
	48: "CM_PROB_DRIVER_BLOCKED",
	52: "CM_PROB_UNSIGNED_DRIVER",
}

// ParseWindowsGPUBindingJSON parses the PnP binding query into enumerated
// NVIDIA display devices (PCI class 03; audio and USB functions are dropped).
// Device-ID coverage is not judged here; it stays unknown until an INF check.
func ParseWindowsGPUBindingJSON(raw string) ([]GPUBindingDevice, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil, errors.New("cudatopology: empty PnP binding JSON")
	}
	var items []winPnPBindingRaw
	switch {
	case strings.HasPrefix(trimmed, "["):
		if err := json.Unmarshal([]byte(trimmed), &items); err != nil {
			return nil, err
		}
	case strings.HasPrefix(trimmed, "{"):
		var one winPnPBindingRaw
		if err := json.Unmarshal([]byte(trimmed), &one); err != nil {
			return nil, err
		}
		items = append(items, one)
	default:
		return nil, errors.New("cudatopology: invalid PnP binding JSON structure")
	}

	devs := make([]GPUBindingDevice, 0, len(items))
	for _, it := range items {
		classKnown, display := windowsDisplayClass(it.Class, it.HardwareIDs, it.CompatibleIDs)
		if classKnown && !display {
			continue
		}
		dev := GPUBindingDevice{
			InstanceID:     it.InstanceID,
			VendorID:       pnpHardwareID(it.InstanceID, "VEN_"),
			DeviceID:       pnpHardwareID(it.InstanceID, "DEV_"),
			SubsystemID:    pnpSubsystemID(it.InstanceID),
			Name:           firstNonEmpty(it.FriendlyName, it.DeviceDesc),
			Driver:         it.DriverInfPath,
			DriverProvider: it.Provider,
			DriverVersion:  it.DriverVersion,
			IDListed:       GPULayerUnknown,
			Source:         CUDASourceWindowsPnP,
			hardwareIDs:    windowsDeviceHardwareIDs(it.InstanceID, it.HardwareIDs, it.CompatibleIDs),
		}
		var unknown []string
		if !classKnown {
			unknown = append(unknown, "pci_class")
		}
		if dev.PCIAddress = windowsPCIAddress(it.BusNumber, it.Address, it.LocationInfo); dev.PCIAddress == "" {
			unknown = append(unknown, "pci_address")
		}
		// An empty INF path means "no driver" only when the problem code was
		// read for the same device; otherwise the property read itself failed.
		switch {
		case dev.Driver == "" && it.ProblemCode != nil:
			dev.VendorDriver = GPULayerNo
		case dev.Driver == "":
			dev.VendorDriver = GPULayerUnknown
			unknown = append(unknown, "driver")
		case strings.Contains(strings.ToUpper(dev.DriverProvider), "NVIDIA"):
			dev.VendorDriver = GPULayerYes
		case dev.DriverProvider != "":
			dev.VendorDriver = GPULayerNo
		default:
			dev.VendorDriver = GPULayerUnknown
			unknown = append(unknown, "driver_provider")
		}
		if it.ProblemCode == nil {
			dev.DriverStarted = GPULayerUnknown
			unknown = append(unknown, "problem_code")
		} else {
			code := *it.ProblemCode
			dev.ProblemCode = &code
			dev.ProblemName = windowsProblemNames[code]
			dev.DriverStarted = GPULayerNo
			if code == 0 {
				dev.DriverStarted = GPULayerYes
			}
		}
		dev.UnknownFields = unknown
		devs = append(devs, dev)
	}
	return devs, nil
}

// pciClassFromIDs returns the two-hex-digit PCI base class carried by a
// "PCI\CC_xx..." or "...&CC_xx..." ID, or "".
func pciClassFromIDs(lists ...[]string) string {
	for _, list := range lists {
		for _, id := range list {
			u := strings.ToUpper(id)
			for off := 0; ; {
				i := strings.Index(u[off:], "CC_")
				if i < 0 {
					break
				}
				at := off + i
				if at > 0 && (u[at-1] == '\\' || u[at-1] == '&') && len(u) >= at+5 {
					return u[at+3 : at+5]
				}
				off = at + 3
			}
		}
	}
	return ""
}

// windowsDisplayClass reports whether the PCI class is known and is a display
// controller (class 03). The class code in the hardware/compatible IDs decides;
// without one, a PnP setup class other than Display marks a non-GPU function.
func windowsDisplayClass(pnpClass string, hw, compat []string) (known, display bool) {
	if cc := pciClassFromIDs(hw, compat); cc != "" {
		return true, cc == "03"
	}
	switch {
	case strings.EqualFold(pnpClass, "Display"):
		return true, true
	case pnpClass != "":
		return true, false
	}
	return false, false
}

// windowsPCIAddress formats "BB:DD.F" from the locale-independent bus number
// and PCI address (device in the high word, function in the low word), falling
// back to the English LocationInfo text. Windows reports no PCI domain, so none
// is claimed.
func windowsPCIAddress(bus, addr *uint32, loc string) string {
	if bus != nil && addr != nil {
		return fmt.Sprintf("%02x:%02x.%x", *bus, *addr>>16, *addr&0xffff)
	}
	var b, d, f int
	if _, err := fmt.Sscanf(strings.TrimSpace(loc), "PCI bus %d, device %d, function %d", &b, &d, &f); err != nil {
		return ""
	}
	return fmt.Sprintf("%02x:%02x.%x", b, d, f)
}

// pnpSubsystemID extracts the 8-hex-digit SUBSYS_ value, lower-cased.
func pnpSubsystemID(pnp string) string {
	u := strings.ToUpper(pnp)
	i := strings.Index(u, "SUBSYS_")
	if i < 0 || len(u) < i+15 {
		return ""
	}
	id := u[i+7 : i+15]
	if _, err := strconv.ParseUint(id, 16, 32); err != nil {
		return ""
	}
	return strings.ToLower(id)
}

// windowsDeviceHardwareIDs returns the device-ID-bearing hardware IDs an INF
// entry can match: the reported hardware and compatible IDs plus those implied
// by the instance ID (full, without REV, and VEN&DEV).
func windowsDeviceHardwareIDs(instanceID string, hw, compat []string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(id string) {
		u := strings.ToUpper(strings.TrimSpace(id))
		if u == "" || !strings.Contains(u, "&DEV_") || seen[u] {
			return
		}
		seen[u] = true
		out = append(out, u)
	}
	for _, id := range hw {
		add(id)
	}
	for _, id := range compat {
		add(id)
	}
	base := strings.ToUpper(instanceID)
	if i := strings.Index(base, `\`); i >= 0 {
		if j := strings.Index(base[i+1:], `\`); j >= 0 {
			base = base[:i+1+j]
		}
	}
	add(base)
	var noRev, venDev []string
	for _, s := range strings.Split(base, "&") {
		if !strings.HasPrefix(s, "REV_") {
			noRev = append(noRev, s)
		}
		if strings.HasPrefix(s, `PCI\VEN_`) || strings.HasPrefix(s, "DEV_") {
			venDev = append(venDev, s)
		}
	}
	add(strings.Join(noRev, "&"))
	add(strings.Join(venDev, "&"))
	return out
}

// infHardwareIDs extracts every upper-cased PCI\VEN_10DE&DEV_ hardware ID an INF
// names outside ";" comments.
func infHardwareIDs(text string) map[string]bool {
	ids := map[string]bool{}
	const marker = `PCI\VEN_10DE&DEV_`
	sc := bufio.NewScanner(strings.NewReader(text))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line, _, _ := strings.Cut(sc.Text(), ";")
		u := strings.ToUpper(line)
		for off := 0; ; {
			i := strings.Index(u[off:], marker)
			if i < 0 {
				break
			}
			start := off + i
			end := start + len(marker)
			for end < len(u) && !strings.ContainsRune(", \t\"%", rune(u[end])) {
				end++
			}
			ids[u[start:end]] = true
			off = end
		}
	}
	return ids
}

// decodeINF returns an INF's text, decoding UTF-16 (BOM-marked) INFs.
func decodeINF(b []byte) string {
	var be bool
	switch {
	case len(b) >= 2 && b[0] == 0xFF && b[1] == 0xFE:
	case len(b) >= 2 && b[0] == 0xFE && b[1] == 0xFF:
		be = true
	default:
		return string(bytes.TrimPrefix(b, []byte{0xEF, 0xBB, 0xBF}))
	}
	b = b[2:]
	u := make([]uint16, len(b)/2)
	for i := range u {
		if be {
			u[i] = uint16(b[2*i])<<8 | uint16(b[2*i+1])
		} else {
			u[i] = uint16(b[2*i+1])<<8 | uint16(b[2*i])
		}
	}
	return string(utf16.Decode(u))
}

// infIsNVIDIADisplay reports whether an INF's [Version] declares Class=Display
// with an NVIDIA provider.
func infIsNVIDIADisplay(text string) bool {
	var display, nvidia bool
	sc := bufio.NewScanner(strings.NewReader(text))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for n := 0; sc.Scan() && n < 200; n++ {
		line, _, _ := strings.Cut(sc.Text(), ";")
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "class":
			display = strings.EqualFold(strings.TrimSpace(val), "Display")
		case "provider":
			nvidia = strings.Contains(strings.ToUpper(val), "NVIDIA")
		}
	}
	return display && nvidia
}

// infIndex lazily reads INFs from one directory and caches their NVIDIA IDs.
// unreadable counts oem*.inf files whose header could not be read: any of them
// may be the NVIDIA INF that lists a device.
type infIndex struct {
	fsys       fs.FS
	ids        map[string]map[string]bool
	vendor     []string
	unreadable int
	listed     bool
	err        error
	issues     []CUDADiscoveryIssue
}

func (x *infIndex) hardwareIDs(name string) (map[string]bool, bool) {
	if ids, ok := x.ids[name]; ok {
		return ids, ids != nil
	}
	b, err := fs.ReadFile(x.fsys, name)
	if err != nil {
		x.ids[name] = nil
		if !errors.Is(err, fs.ErrNotExist) {
			x.issues = append(x.issues, CUDADiscoveryIssue{Path: name, Error: err.Error()})
		}
		return nil, false
	}
	ids := infHardwareIDs(decodeINF(b))
	x.ids[name] = ids
	return ids, true
}

// vendorINFs lists installed third-party INFs (oem*.inf) that declare an NVIDIA display driver.
func (x *infIndex) vendorINFs() ([]string, error) {
	if x.listed {
		return x.vendor, x.err
	}
	x.listed = true
	entries, err := fs.ReadDir(x.fsys, ".")
	if err != nil {
		x.err = err
		return nil, err
	}
	for _, e := range entries {
		name := strings.ToLower(e.Name())
		if e.IsDir() || !strings.HasPrefix(name, "oem") || !strings.HasSuffix(name, ".inf") {
			continue
		}
		head, err := readINFHead(x.fsys, e.Name())
		if err != nil {
			x.unreadable++
			x.issues = append(x.issues, CUDADiscoveryIssue{Path: e.Name(), Error: err.Error()})
			continue
		}
		if infIsNVIDIADisplay(decodeINF(head)) {
			x.vendor = append(x.vendor, e.Name())
		}
	}
	sort.Strings(x.vendor)
	return x.vendor, nil
}

func readINFHead(fsys fs.FS, name string) ([]byte, error) {
	f, err := fsys.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	head := make([]byte, 16*1024)
	n, err := io.ReadFull(f, head)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return nil, err
	}
	return head[:n], nil
}

// applyINFCoverage judges whether the device ID is listed: by the bound INF for
// a device bound to the NVIDIA driver, else by any installed NVIDIA display INF.
// No match is "no" only when every candidate (and, for the scan, every oem INF)
// was read.
func applyINFCoverage(dev *GPUBindingDevice, x *infIndex) {
	var candidates []string
	scanned := false
	if dev.VendorDriver == GPULayerYes && dev.Driver != "" {
		candidates = []string{dev.Driver}
	} else {
		vendor, err := x.vendorINFs()
		if err != nil {
			dev.IDListed = GPULayerUnknown
			dev.UnknownFields = appendUnique(dev.UnknownFields, "device_id_listed")
			return
		}
		candidates, scanned = vendor, true
	}
	dev.IDListed = GPULayerNo
	read := 0
	for _, name := range candidates {
		ids, ok := x.hardwareIDs(name)
		if !ok {
			continue
		}
		read++
		dev.IDCheckedIn = append(dev.IDCheckedIn, name)
		for _, hw := range dev.hardwareIDs {
			if ids[hw] {
				dev.IDListedBy = append(dev.IDListedBy, name)
				dev.IDListed = GPULayerYes
				break
			}
		}
	}
	incomplete := read < len(candidates) || (scanned && x.unreadable > 0)
	if incomplete && dev.IDListed != GPULayerYes {
		dev.IDListed = GPULayerUnknown
		dev.UnknownFields = appendUnique(dev.UnknownFields, "device_id_listed")
	}
}

func diagnoseGPUBindingWindows(opts GPUBindingOptions) GPUBindingReport {
	query := opts.pnpQuery
	if query == nil {
		query = execWindowsPnPBinding
	}
	ctx, cancel := context.WithTimeout(context.Background(), gpuBindingQueryTimeout)
	out, err := query(ctx)
	cancel()
	if err != nil {
		return gpuBindingEnumerationFailed(CUDASourceWindowsPnP, opts.ExpectGPUs, []CUDADiscoveryIssue{{Path: "Get-PnpDevice", Error: err.Error()}})
	}
	devs, err := ParseWindowsGPUBindingJSON(out)
	if err != nil {
		return gpuBindingEnumerationFailed(CUDASourceWindowsPnP, opts.ExpectGPUs, []CUDADiscoveryIssue{{Path: "Get-PnpDevice", Error: err.Error()}})
	}
	infDir := opts.INFDir
	if infDir == "" {
		root := os.Getenv("SystemRoot")
		if root == "" {
			root = `C:\Windows`
		}
		infDir = filepath.Join(root, "INF")
	}
	fsys := opts.infFS
	if fsys == nil {
		fsys = os.DirFS(infDir)
	}
	x := &infIndex{fsys: fsys, ids: map[string]map[string]bool{}}
	for i := range devs {
		applyINFCoverage(&devs[i], x)
	}
	var issues []CUDADiscoveryIssue
	for _, is := range x.issues {
		issues = append(issues, CUDADiscoveryIssue{Path: filepath.ToSlash(filepath.Join(infDir, is.Path)), Error: is.Error})
	}
	if x.err != nil {
		issues = append(issues, CUDADiscoveryIssue{Path: filepath.ToSlash(infDir), Error: x.err.Error()})
	}
	return BuildGPUBindingReport(GPUBindingInputs{
		Method:     CUDASourceWindowsPnP,
		Devices:    devs,
		CUDA:       cudaViewIfNeeded(devs, opts.smiQuery),
		ExpectGPUs: opts.ExpectGPUs,
		Issues:     issues,
	})
}

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
func aliasCoverage(modalias string, patterns []string) GPULayerState {
	state := GPULayerNo
	for _, p := range patterns {
		if ok, err := path.Match(p, modalias); err != nil || !ok {
			continue
		}
		if !modaliasDeviceWildcard(p) {
			return GPULayerYes
		}
		state = GPULayerUnknown
	}
	return state
}

func diagnoseGPUBindingLinux(sysfsRoot string, opts GPUBindingOptions) GPUBindingReport {
	d := NewCUDAPCIeTopologyDiscovery(sysfsRoot)
	pciDevicesDir := filepath.Join(sysfsRoot, "bus", "pci", "devices")
	entries, err := os.ReadDir(pciDevicesDir)
	if err != nil {
		d.addIssue(pciDevicesDir, err)
		return gpuBindingEnumerationFailed(CUDASourceSysfs, opts.ExpectGPUs, d.Issues)
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
				d.addIssue(p, err)
			}
			continue
		}
		var s nvidiaSupportedGPUs
		if err := json.Unmarshal(b, &s); err != nil {
			d.addIssue(p, err)
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
			d.addIssue(aliasPath, err)
		}
	}

	incomplete := false
	var devs []GPUBindingDevice
	for _, entry := range entries {
		devDir := filepath.Join(pciDevicesDir, entry.Name())
		before := len(d.Issues)
		vendor, ok := d.readSysfsAttr(devDir, "vendor")
		class, _ := d.readSysfsAttr(devDir, "class")
		if len(d.Issues) > before {
			incomplete = true
		}
		if !ok || !strings.EqualFold(vendor, "0x10de") {
			continue
		}
		if class != "" && !strings.HasPrefix(strings.ToLower(class), "0x03") {
			continue
		}
		dev := GPUBindingDevice{
			PCIAddress: strings.ReplaceAll(entry.Name(), "_", ":"),
			VendorID:   strings.ToLower(vendor),
			Source:     CUDASourceSysfs,
		}
		if addr, ok := d.readSysfsAttr(devDir, "address"); ok && addr != "" {
			dev.PCIAddress = addr
		}
		dev.DeviceID, _ = d.readSysfsAttr(devDir, "device")
		dev.DeviceID = strings.ToLower(dev.DeviceID)
		sv, _ := d.readSysfsAttr(devDir, "subsystem_vendor")
		sd, _ := d.readSysfsAttr(devDir, "subsystem_device")
		if sv != "" && sd != "" {
			dev.SubsystemID = strings.ToLower(strings.TrimPrefix(sd, "0x") + strings.TrimPrefix(sv, "0x"))
		}
		dev.Name, _ = d.readSysfsAttr(devDir, "label")
		dev.modalias, _ = d.readSysfsAttr(devDir, "modalias")

		var unknown []string
		link, err := os.Readlink(filepath.Join(devDir, "driver"))
		switch {
		case err == nil:
			dev.Driver = path.Base(filepath.ToSlash(link))
			dev.DriverStarted = GPULayerYes
			dev.VendorDriver = GPULayerNo
			if dev.Driver == "nvidia" {
				dev.VendorDriver = GPULayerYes
			}
			if mod, err := os.Readlink(filepath.Join(devDir, "driver", "module")); err == nil {
				dev.DriverModule = path.Base(filepath.ToSlash(mod))
				dev.DriverVersion, _ = d.readSysfsAttr(filepath.Join(sysfsRoot, "module", dev.DriverModule), "version")
			}
		case errors.Is(err, fs.ErrNotExist):
			dev.VendorDriver, dev.DriverStarted = GPULayerNo, GPULayerNo
		default:
			d.addIssue(filepath.Join(devDir, "driver"), err)
			dev.VendorDriver, dev.DriverStarted = GPULayerUnknown, GPULayerUnknown
			unknown = append(unknown, "driver")
		}

		dev.IDListed = GPULayerUnknown
		switch {
		case supported != nil && dev.DeviceID != "":
			dev.IDCheckedIn = []string{"supported-gpus.json"}
			dev.IDListed = GPULayerNo
			if supported.listsDevice(dev.DeviceID) {
				dev.IDListed = GPULayerYes
				dev.IDListedBy = dev.IDCheckedIn
			}
		case aliasLoaded && dev.modalias != "":
			dev.IDCheckedIn = []string{"modules.alias:nvidia"}
			if dev.IDListed = aliasCoverage(dev.modalias, nvidiaAliases); dev.IDListed == GPULayerYes {
				dev.IDListedBy = dev.IDCheckedIn
			}
		}
		if dev.IDListed == GPULayerUnknown {
			unknown = append(unknown, "device_id_listed")
		}
		dev.UnknownFields = unknown
		devs = append(devs, dev)
	}

	return BuildGPUBindingReport(GPUBindingInputs{
		Method:                CUDASourceSysfs,
		Devices:               devs,
		CUDA:                  cudaViewIfNeeded(devs, opts.smiQuery),
		ExpectGPUs:            opts.ExpectGPUs,
		UnauthorizedTunnels:   linuxUnauthorizedTunnels(d, sysfsRoot),
		Issues:                d.Issues,
		EnumerationIncomplete: incomplete,
	})
}

// linuxUnauthorizedTunnels lists Thunderbolt/USB4 devices whose sysfs
// authorized attribute is 0; a PCIe device behind one cannot enumerate.
func linuxUnauthorizedTunnels(d *CUDAPCIeTopologyDiscovery, sysfsRoot string) []string {
	dir := filepath.Join(sysfsRoot, "bus", "thunderbolt", "devices")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			d.addIssue(dir, err)
		}
		return nil
	}
	var out []string
	for _, e := range entries {
		if auth, ok := d.readSysfsAttr(filepath.Join(dir, e.Name()), "authorized"); ok && auth == "0" {
			out = append(out, e.Name())
		}
	}
	return out
}

func appendUnique(list []string, s string) []string {
	for _, v := range list {
		if v == s {
			return list
		}
	}
	return append(list, s)
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
