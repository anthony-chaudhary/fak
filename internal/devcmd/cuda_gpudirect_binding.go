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
func runCUDABinding(stdout io.Writer, sysfsRoot string, expect int, jsonOut bool) int {
	if sysfsRoot == "/sys" && runtime.GOOS != "linux" {
		sysfsRoot = ""
	}
	rep := compute.NewCUDAPCIeTopologyDiscovery(sysfsRoot).DiagnoseGPUBinding(compute.GPUBindingOptions{ExpectGPUs: expect})
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
		if len(d.UnknownFields) > 0 {
			fmt.Fprintf(stdout, "      not observed: %s\n", strings.Join(d.UnknownFields, ", "))
		}
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
