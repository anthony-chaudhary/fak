package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// runCUDABinding runs the GPU binding doctor: per NVIDIA PCI display device it
// compares enumeration, driver binding, device-ID coverage and NVML visibility
// and prints one closed verdict. Exit 0 only when every device is GPU_READY.
func runCUDABinding(stdout io.Writer, sysfsRoot string, opts compute.GPUBindingOptions, jsonOut bool) int {
	if sysfsRoot == "/sys" && runtime.GOOS != "linux" {
		sysfsRoot = ""
	}
	rep := compute.NewCUDAPCIeTopologyDiscovery(sysfsRoot).DiagnoseGPUBinding(opts)
	code := 1
	if rep.Ready {
		code = 0
	}
	if jsonOut {
		data, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Fprintln(stdout, string(data))
		return code
	}

	fmt.Fprintf(stdout, "CUDA GPU binding doctor (method: %s, status: %s):\n", orUnknown(rep.Method), rep.Status)
	for _, is := range rep.Issues {
		fmt.Fprintf(stdout, "  read error: %s: %s\n", is.Path, is.Error)
	}
	nvml := fmt.Sprintf("%s (%d listed)", rep.CUDA.State, len(rep.CUDA.Devices))
	if rep.CUDA.Error != "" {
		nvml += ": " + rep.CUDA.Error
	}
	fmt.Fprintf(stdout, "  NVML: %s\n", nvml)
	for _, d := range rep.Devices {
		id := "unknown"
		if d.VendorID != "" && d.DeviceID != "" {
			id = strings.TrimPrefix(d.VendorID, "0x") + ":" + strings.TrimPrefix(d.DeviceID, "0x")
		}
		name := d.Name
		if !d.Enumerated {
			name = "expected GPU not on PCI"
			if d.Tunnel != "" {
				name += " (unauthorized tunnel " + d.Tunnel + ")"
			}
		}
		fmt.Fprintf(stdout, "  %-26s %-12s %-9s %s\n", d.Verdict, orUnknown(d.PCIAddress), id, name)
		if !d.Enumerated {
			continue
		}
		driver := orUnknown(d.Driver)
		if d.Driver == "" && d.VendorDriver == compute.GPULayerNo {
			driver = "none"
		}
		if d.DriverProvider != "" || d.DriverVersion != "" {
			driver += " (" + strings.TrimSpace(d.DriverProvider+" "+d.DriverVersion) + ")"
		}
		problem := ""
		if d.ProblemCode != nil {
			problem = fmt.Sprintf(" problem=%d", *d.ProblemCode)
			if d.ProblemName != "" {
				problem += " " + d.ProblemName
			}
		}
		listed := string(d.IDListed)
		if len(d.IDListedBy) > 0 {
			listed += " by " + strings.Join(d.IDListedBy, ",")
		} else if len(d.IDCheckedIn) > 0 {
			listed += " in " + strings.Join(d.IDCheckedIn, ",")
		}
		cuda := string(d.CUDAVisible)
		if d.CUDAIndex != nil {
			cuda += fmt.Sprintf(" index=%d", *d.CUDAIndex)
		}
		fmt.Fprintf(stdout, "      driver=%s vendor=%s started=%s%s id_listed=%s cuda=%s\n",
			driver, d.VendorDriver, d.DriverStarted, problem, listed, cuda)
		if path := bindingPath(d); path != "" {
			fmt.Fprintf(stdout, "      path: %s\n", path)
		}
		if d.USB4HostRouter != "" {
			fmt.Fprintf(stdout, "      usb4 host router: %s (%s)\n", d.USB4HostRouter, d.USB4RouterBasis)
		}
		if len(d.UnknownFields) > 0 {
			fmt.Fprintf(stdout, "      not observed: %s\n", strings.Join(d.UnknownFields, ", "))
		}
	}
	for _, r := range rep.USB4HostRouters {
		fmt.Fprintf(stdout, "  USB4 host router: %s %s %s\n", orUnknown(r.PCIAddress), orUnknown(r.ACPIPath), r.InstanceID)
	}
	if rep.Package != nil {
		printDriverPackage(stdout, rep.Package)
	}
	verdicts := make([]string, 0, len(rep.Verdicts))
	for v, n := range rep.Verdicts {
		verdicts = append(verdicts, fmt.Sprintf("%s=%d", v, n))
	}
	sort.Strings(verdicts)
	fmt.Fprintf(stdout, "  Verdicts: %s\n", orUnknown(strings.Join(verdicts, " ")))
	fmt.Fprintf(stdout, "  Ready: %v\n", rep.Ready)
	return code
}

// bindingPath renders the parent chain root-first, e.g. "00:07.3 > 80:00.0 > 81:00.0 > GPU".
func bindingPath(d compute.GPUBindingDevice) string {
	var hops []string
	for i := len(d.ParentChain) - 1; i >= 0; i-- {
		if h := d.ParentChain[i]; h.PCIAddress != "" {
			hops = append(hops, h.PCIAddress)
		}
	}
	if len(hops) == 0 {
		return ""
	}
	return strings.Join(append(hops, "GPU"), " > ")
}

func printDriverPackage(stdout io.Writer, pkg *compute.GPUDriverPackage) {
	fmt.Fprintf(stdout, "  Package %s (%d NVIDIA display INFs):\n", pkg.Dir, pkg.DisplayINFs)
	for _, is := range pkg.Issues {
		fmt.Fprintf(stdout, "    read error: %s: %s\n", is.Path, is.Error)
	}
	for _, d := range pkg.Devices {
		if d.Chosen == "" {
			fmt.Fprintf(stdout, "    %s: no package INF lists this GPU\n", orUnknown(d.PCIAddress))
			continue
		}
		fmt.Fprintf(stdout, "    %s: %s via %s (listed by %s)\n", orUnknown(d.PCIAddress), d.Chosen, d.MatchedID, strings.Join(d.ListedBy, ", "))
	}
	if len(pkg.PnputilCommands) > 0 {
		fmt.Fprintln(stdout, "  Install plan (not run; elevated shell, reboot after):")
		for _, line := range pkg.PnputilCommands {
			fmt.Fprintf(stdout, "    %s\n", line)
		}
	}
}
