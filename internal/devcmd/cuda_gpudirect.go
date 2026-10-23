package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Qwen38CUDADirectSwapReceiptSchema identifies the official Qwen3.8 CUDA Direct swap receipt schema.
const Qwen38CUDADirectSwapReceiptSchema = "fak.modelengine.qwen38-cudadirect-swap/1"

// HostTopologyInfo holds observed host fields, or explicitly modeled fixture values.
type HostTopologyInfo struct {
	CPU         string `json:"cpu"`
	Motherboard string `json:"motherboard"`
	RAM         string `json:"ram"`
}

// CUDAInspectOutput separates observed inventory from unverified hardware capabilities.
type CUDAInspectOutput struct {
	Provenance        string                    `json:"provenance"`
	Verdict           string                    `json:"verdict"`
	IsPhysicalSilicon bool                      `json:"is_physical_silicon"`
	UnknownFields     []string                  `json:"unknown_fields"`
	ProbeError        string                    `json:"probe_error,omitempty"`
	Host              HostTopologyInfo          `json:"host"`
	GPUs              []compute.CUDADeviceNode  `json:"gpus"`
	NVMeDevices       []compute.NVMeStorageNode `json:"nvme_devices"`
	Routes            []compute.P2PRouteVerdict `json:"routes"`
	ACSStatus         string                    `json:"acs_status"`
}

// CUDAAuditCheck represents a single hardware or driver setting verification item.
type CUDAAuditCheck struct {
	Name        string `json:"name"`
	Status      string `json:"status"` // PASS, WARN, FAIL, or UNKNOWN
	Detail      string `json:"detail"`
	Remediation string `json:"remediation,omitempty"`
}

// CUDAHardwareAuditReport aggregates system configuration checks for CUDA GPU Direct and BaM P2PDMA.
type CUDAHardwareAuditReport struct {
	Provenance        string           `json:"provenance"`
	IsPhysicalSilicon bool             `json:"is_physical_silicon"`
	UnknownFields     []string         `json:"unknown_fields"`
	ProbeError        string           `json:"probe_error,omitempty"`
	Healthy           bool             `json:"healthy"`
	Verdict           string           `json:"verdict"` // PASS, WARN, FAIL, UNVERIFIED, or MODELED
	Host              HostTopologyInfo `json:"host"`
	Checks            []CUDAAuditCheck `json:"checks"`
	RemediationSteps  []string         `json:"remediation_steps,omitempty"`
}

// CUDABaMBenchResult records software queue activity, not physical DMA performance.
type CUDABaMBenchResult struct {
	Provenance        string   `json:"provenance"`
	Verdict           string   `json:"verdict"`
	IsPhysicalSilicon bool     `json:"is_physical_silicon"`
	UnknownFields     []string `json:"unknown_fields"`
	Device            string   `json:"device"`
	Storage           string   `json:"storage"`
	RouteKind         string   `json:"route_kind"`
	QueueArchitecture string   `json:"queue_architecture"`
	TotalBytesMoved   uint64   `json:"total_bytes_moved"`
	CommandsExecuted  int      `json:"commands_executed"`
	DurationNanos     int64    `json:"duration_nanos"`
	ThroughputGBps    float64  `json:"throughput_gbps"`
	IOPS              float64  `json:"iops"`
	StagingCopyCount  int      `json:"staging_copy_count"`
	DoorbellRings     uint64   `json:"doorbell_rings"`
	ZeroCopyVerified  bool     `json:"zero_copy_verified"`
}

// Qwen38CUDADirectSwapReceipt represents the structured, machine-readable receipt for Qwen3.8 CUDA Direct swapping.
type Qwen38CUDADirectSwapReceipt struct {
	IsPhysicalSilicon      bool                             `json:"is_physical_silicon"`
	UnknownFields          []string                         `json:"unknown_fields"`
	Schema                 string                           `json:"schema"`
	Provenance             string                           `json:"provenance"`
	Verdict                string                           `json:"verdict"`
	Model                  string                           `json:"model"`
	Architecture           string                           `json:"architecture"`
	StagingCopyCount       int                              `json:"staging_copy_count"`
	BytesMoved             uint64                           `json:"bytes_moved"`
	DirectDMABandwidthGBps float64                          `json:"direct_dma_bandwidth_gbps"`
	SpeedupVsBaseline      Qwen38BenchSpeedupMetrics        `json:"speedup_vs_baseline"`
	SpeedupVsReference     Qwen38BenchSpeedupMetrics        `json:"speedup_vs_reference"`
	Arms                   map[string]Qwen38BenchArmMetrics `json:"arms"`
	Baseline               Qwen38BenchArmMetrics            `json:"baseline"`
	FakNative              Qwen38BenchArmMetrics            `json:"fak_native"`
	Reference              Qwen38BenchArmMetrics            `json:"reference"`
	Evidence               []string                         `json:"evidence"`
}

var modeledHostTopology = HostTopologyInfo{
	CPU:         "AMD Ryzen 9 5950X (Zen 3, 16c/32t)",
	Motherboard: "Gigabyte X570 Aorus Elite WiFi",
	RAM:         "128GB DDR4",
}

var modeledCUDAGPU = compute.CUDADeviceNode{
	PCIAddress:       "0000:09:00.0",
	DeviceName:       "NVIDIA GeForce RTX 5090 FE",
	Arch:             "sm_120",
	TotalVRAMBytes:   32 * 1024 * 1024 * 1024,
	BAR1SizeBytes:    32 * 1024 * 1024 * 1024,
	ReBAREnabled:     true,
	UpstreamRootPort: "0000:00:01.1",
	NUMANode:         0,
}

var modeledNVMeStorage = compute.NVMeStorageNode{
	PCIAddress:       "0000:01:00.0",
	Model:            "Samsung SSD 990 PRO 2TB",
	SlotType:         compute.NVMeSlotM2ACPU,
	UpstreamRootPort: "0000:00:01.2",
	PeakReadGBps:     7.45,
	NUMANode:         0,
}

// Probe results never receive fallback hardware. A partial inventory remains useful
// even when the probe returns an error, and an empty inventory remains empty.
func probeCUDATopology(sysfsRoot string) (*compute.CUDAPCIeTopologyDiscovery, error) {
	disco := compute.NewCUDAPCIeTopologyDiscovery(sysfsRoot)
	err := disco.Discover()
	return disco, err
}

func modeledCUDATopology() *compute.CUDAPCIeTopologyDiscovery {
	disco := compute.NewCUDAPCIeTopologyDiscovery("")
	gpu, nvme := modeledCUDAGPU, modeledNVMeStorage
	gpu.EvidenceSource, nvme.EvidenceSource = "MODELED", "MODELED"
	disco.GPUs = append(disco.GPUs, gpu)
	disco.NVMeDevices = append(disco.NVMeDevices, nvme)
	return disco
}

type cudaTopologyObservation struct {
	discovery *compute.CUDAPCIeTopologyDiscovery
	probeErr  error
	simulated bool
}

// RunCUDAGPUDirect inspects inventory and runs explicitly selected software simulations.
// Hardware qualification requires probes and physical execution this command does not implement.
func RunCUDAGPUDirect(stdout, stderr io.Writer, argv []string) int {
	return runCUDAGPUDirectWithProbe(stdout, stderr, argv, probeCUDATopology)
}

func runCUDAGPUDirectWithProbe(stdout, stderr io.Writer, argv []string, probe func(string) (*compute.CUDAPCIeTopologyDiscovery, error)) int {
	var sub string
	flagArgs := argv
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		sub, flagArgs = argv[0], argv[1:]
	}
	fs := flag.NewFlagSet("cuda-gpudirect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON output")
	mode := fs.String("mode", "inspect", "mode: inspect, audit, bench, qwen38, qwen38-bench, or qwen38-sim")
	sysfsRoot := fs.String("sysfs", "", "Linux sysfs root override (default: platform discovery)")
	simulate := fs.Bool("simulate", false, "use modeled fixtures only; never probe or qualify hardware")
	if err := fs.Parse(flagArgs); err != nil {
		return 2
	}
	if sub == "" {
		sub = *mode
		if fs.NArg() == 1 {
			sub = fs.Arg(0)
		}
	} else if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak cuda-gpudirect: unexpected positional arguments")
		return 2
	}
	if fs.NArg() > 1 {
		fmt.Fprintln(stderr, "fak cuda-gpudirect: unexpected positional arguments")
		return 2
	}
	sub = strings.ToLower(sub)
	switch sub {
	case "qwen38-sim":
		*simulate = true
	case "bench", "qwen38", "qwen38-bench":
		if !*simulate {
			fmt.Fprintf(stderr, "fak cuda-gpudirect: %s has no physical execution backend; use --simulate for the software model\n", sub)
			return 2
		}
	case "inspect", "audit":
	default:
		fmt.Fprintf(stderr, "fak cuda-gpudirect: unknown mode %q (supported: inspect, audit, bench, qwen38, qwen38-bench, qwen38-sim)\n", sub)
		return 2
	}

	observation := cudaTopologyObservation{simulated: *simulate}
	if *simulate {
		observation.discovery = modeledCUDATopology()
	} else {
		observation.discovery, observation.probeErr = probe(*sysfsRoot)
		if observation.probeErr != nil {
			fmt.Fprintf(stderr, "fak cuda-gpudirect: topology probe: %v\n", observation.probeErr)
		}
	}
	switch sub {
	case "inspect":
		return runCUDAInspect(stdout, observation, *jsonOut)
	case "audit":
		return runCUDAAudit(stdout, observation, *jsonOut)
	case "bench":
		return runCUDABench(stdout, stderr, observation.discovery, *jsonOut)
	default:
		return runCUDA_Qwen38Bench(stdout, stderr, observation.discovery, *jsonOut)
	}
}

func cudaInspection(observation cudaTopologyObservation) CUDAInspectOutput {
	disco := observation.discovery
	out := CUDAInspectOutput{
		Provenance: "OBSERVED", Verdict: "UNVERIFIED",
		GPUs: disco.GPUs, NVMeDevices: disco.NVMeDevices,
		Routes: disco.EvaluateAllRoutes(), ACSStatus: "UNKNOWN",
		UnknownFields: []string{"host.cpu", "host.motherboard", "host.ram", "acs_status", "physical_p2p_dma"},
	}
	if observation.probeErr != nil {
		out.ProbeError = observation.probeErr.Error()
	}
	if observation.simulated {
		out.Provenance, out.Verdict, out.ACSStatus = "MODELED", "MODELED", "MODELED_NOT_VERIFIED"
		out.Host = modeledHostTopology
		out.UnknownFields = []string{"physical_p2p_dma"}
		out.Routes = make([]compute.P2PRouteVerdict, 0, len(disco.GPUs)*len(disco.NVMeDevices))
		for _, gpu := range disco.GPUs {
			for _, nvme := range disco.NVMeDevices {
				out.Routes = append(out.Routes, compute.EvaluateModeledP2PRoute(gpu, nvme))
			}
		}
	}
	return out
}

func cudaKnownText(value string) string {
	if value == "" {
		return "UNKNOWN"
	}
	return value
}

func runCUDAInspect(stdout io.Writer, observation cudaTopologyObservation, jsonOut bool) int {
	out := cudaInspection(observation)
	if jsonOut {
		data, _ := json.MarshalIndent(out, "", "  ")
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintf(stdout, "CUDA GPU Direct Topology & Interconnect Prober: %s (%s)\n", out.Verdict, out.Provenance)
		fmt.Fprintln(stdout, "  Physical P2P DMA: UNKNOWN (no physical silicon execution witness)")
		fmt.Fprintf(stdout, "  Host CPU: %s | Motherboard: %s | Host Memory: %s\n", cudaKnownText(out.Host.CPU), cudaKnownText(out.Host.Motherboard), cudaKnownText(out.Host.RAM))
		if observation.simulated {
			fmt.Fprintln(stdout, "  MODELED fixtures only; no hardware was probed.")
		}
		if out.ProbeError != "" {
			fmt.Fprintf(stdout, "  Probe Error: %s\n", out.ProbeError)
		}
		fmt.Fprintf(stdout, "  CUDA Devices (%d inventory entries):\n", len(out.GPUs))
		for i, gpu := range out.GPUs {
			fmt.Fprintf(stdout, "    GPU %d: %s BDF=%s Arch=%s\n", i, cudaKnownText(gpu.DeviceName), cudaKnownText(gpu.PCIAddress), cudaKnownText(gpu.Arch))
			fmt.Fprintf(stdout, "           Upstream Root Port: %s\n", cudaKnownText(gpu.UpstreamRootPort))
			if len(gpu.UnknownFields) > 0 {
				fmt.Fprintf(stdout, "           UNKNOWN: %s\n", strings.Join(gpu.UnknownFields, ", "))
			}
		}
		fmt.Fprintf(stdout, "  NVMe Storage Devices (%d inventory entries):\n", len(out.NVMeDevices))
		for i, nvme := range out.NVMeDevices {
			fmt.Fprintf(stdout, "    NVMe %d: %s BDF=%s Slot=%s\n", i, cudaKnownText(nvme.Model), cudaKnownText(nvme.PCIAddress), cudaKnownText(string(nvme.SlotType)))
			if len(nvme.UnknownFields) > 0 {
				fmt.Fprintf(stdout, "            UNKNOWN: %s\n", strings.Join(nvme.UnknownFields, ", "))
			}
		}
		fmt.Fprintf(stdout, "  PCIe P2P Interconnect Routes (%d evaluated):\n", len(out.Routes))
		for i, route := range out.Routes {
			fmt.Fprintf(stdout, "    Route %d: %s (%s); physical P2P validation: UNKNOWN\n", i, route.RouteKind, route.EvidenceSource)
			for _, note := range route.DiagnosticNotes {
				fmt.Fprintf(stdout, "             - %s\n", note)
			}
		}
		fmt.Fprintf(stdout, "  Summary ACS Status: %s\n", out.ACSStatus)
	}
	if observation.probeErr != nil {
		return 1
	}
	return 0
}

// Unknown, warning, modeled, and empty check sets never establish healthy hardware.
func cudaAuditVerdict(checks []CUDAAuditCheck) (string, bool) {
	verdict := "PASS"
	if len(checks) == 0 {
		return "UNVERIFIED", false
	}
	for _, check := range checks {
		switch check.Status {
		case "FAIL":
			return "FAIL", false
		case "PASS":
		case "WARN":
			if verdict == "PASS" {
				verdict = "WARN"
			}
		default:
			verdict = "UNVERIFIED"
		}
	}
	return verdict, verdict == "PASS"
}

func runCUDAAudit(stdout io.Writer, observation cudaTopologyObservation, jsonOut bool) int {
	out := cudaInspection(observation)
	checks := []CUDAAuditCheck{
		{Name: "Above 4G Decoding", Status: "UNKNOWN", Detail: "BIOS MMIO configuration was not probed."},
		{Name: "Resizable BAR (ReBAR)", Status: "UNKNOWN", Detail: "BAR allocation alone does not verify VRAM capacity or Resizable BAR enablement."},
		{Name: "PCIe Ten Bit Tag", Status: "UNKNOWN", Detail: "PCIe extended tag configuration was not probed."},
		{Name: "IOMMU / ACS", Status: "UNKNOWN", Detail: "Root-port or NUMA proximity does not verify IOMMU mappings or ACS redirection."},
		{Name: "NVreg_EnableResizableBar=1", Status: "UNKNOWN", Detail: "NVIDIA driver module parameters were not probed."},
		{Name: "Physical P2P DMA", Status: "UNKNOWN", Detail: "No physical GPU/NVMe execution or data-integrity witness was collected."},
	}
	if len(out.GPUs) == 0 || len(out.NVMeDevices) == 0 {
		checks = append(checks, CUDAAuditCheck{Name: "GPU / NVMe inventory", Status: "UNKNOWN", Detail: fmt.Sprintf("Inventory contains %d GPU and %d NVMe entries; missing devices are not replaced with fixtures.", len(out.GPUs), len(out.NVMeDevices))})
	}
	if observation.probeErr != nil {
		checks = append(checks, CUDAAuditCheck{Name: "Topology probe", Status: "UNKNOWN", Detail: observation.probeErr.Error()})
	}
	verdict, healthy := cudaAuditVerdict(checks)
	if observation.simulated {
		verdict, healthy = "MODELED", false
	}
	report := CUDAHardwareAuditReport{
		Provenance: out.Provenance, Verdict: verdict, Healthy: healthy, Host: out.Host,
		UnknownFields: append(out.UnknownFields, "above_4g_decoding", "rebar_enabled", "pcie_ten_bit_tag", "iommu_acs", "nvidia_driver_parameters"),
		ProbeError:    out.ProbeError, Checks: checks,
		RemediationSteps: []string{"Collect platform, driver, and physical P2P DMA evidence before qualifying this configuration."},
	}
	if jsonOut {
		data, _ := json.MarshalIndent(report, "", "  ")
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintln(stdout, "CUDA GPU Direct & BaM P2PDMA Hardware Audit:")
		fmt.Fprintf(stdout, "  Healthy: %v (Verdict: %s; Provenance: %s)\n", healthy, verdict, report.Provenance)
		if observation.simulated {
			fmt.Fprintln(stdout, "  MODELED fixtures only; no hardware was probed.")
		}
		for _, check := range checks {
			fmt.Fprintf(stdout, "    [%s] %s: %s\n", check.Status, check.Name, check.Detail)
		}
		for _, step := range report.RemediationSteps {
			fmt.Fprintf(stdout, "  Next: %s\n", step)
		}
	}
	if healthy {
		return 0
	}
	return 1
}

func runCUDABench(stdout, stderr io.Writer, disco *compute.CUDAPCIeTopologyDiscovery, jsonOut bool) int {
	gpu := modeledCUDAGPU
	if len(disco.GPUs) > 0 {
		gpu = disco.GPUs[0]
	}
	nvme := modeledNVMeStorage
	if len(disco.NVMeDevices) > 0 {
		nvme = disco.NVMeDevices[0]
	}
	verdict := compute.EvaluateModeledP2PRoute(gpu, nvme)

	const totalBlocks = 1024
	const blockSize = 64 * 1024
	const numCmds = 32

	slabCfg := compute.CUDADirectStorageConfig{
		NodeID:          0,
		BlockSize:       blockSize,
		TotalBlocks:     totalBlocks,
		BaseAddress:     compute.CUDADefaultBAR1BaseAddr,
		Arch:            gpu.Arch,
		DeviceName:      gpu.DeviceName,
		QueueCapacity:   512,
		DoorbellAddress: 0xD0000000,
	}

	slab, err := compute.NewCUDADirectStorageMemorySlab(slabCfg)
	if err != nil {
		fmt.Fprintf(stderr, "cuda-gpudirect bench: failed to initialize storage slab: %v\n", err)
		return 1
	}

	cmds := make([]*compute.CUDANVMeP2PCommand, numCmds)
	var totalBytes uint64
	for i := 0; i < numCmds; i++ {
		opcode := compute.CUDANVMeOpcodeRead
		if i%2 == 1 {
			opcode = compute.CUDANVMeOpcodeWrite
		}
		cmd := &compute.CUDANVMeP2PCommand{
			CommandID:      uint16(i + 1),
			Opcode:         opcode,
			NamespaceID:    1,
			StartingLBA:    uint64(i * 128),
			BlockCount:     128,
			TargetVRAMAddr: compute.CUDADefaultBAR1BaseAddr + uintptr(uint64(i)*blockSize),
			ByteLength:     blockSize,
		}
		cmds[i] = cmd
		totalBytes += blockSize
	}

	start := time.Now()
	if err := slab.Queue().SubmitBatch(cmds); err != nil {
		fmt.Fprintf(stderr, "cuda-gpudirect bench: submit batch failed: %v\n", err)
		return 1
	}
	resolved := slab.Queue().PollCompletions(numCmds)
	dur := time.Since(start)

	if resolved != numCmds {
		fmt.Fprintf(stderr, "cuda-gpudirect bench: resolved %d of %d commands\n", resolved, numCmds)
		return 1
	}

	var stagingCopies int
	for _, cmd := range cmds {
		stagingCopies += cmd.StagingCopyCount()
	}
	stagingCopies += slab.Queue().StagingCopyCount()
	stagingCopies += slab.StagingCopyCount()

	res := CUDABaMBenchResult{
		Provenance: "MODELED", Verdict: "MODELED", IsPhysicalSilicon: false,
		UnknownFields:     []string{"throughput_gbps", "iops", "zero_copy_verified", "physical_p2p_dma", "physical_bytes_moved"},
		Device:            fmt.Sprintf("%s (%s)", gpu.DeviceName, gpu.Arch),
		Storage:           fmt.Sprintf("%s (%s)", nvme.Model, nvme.SlotType),
		RouteKind:         string(verdict.RouteKind),
		QueueArchitecture: "Software model of CUDA BaM submission/completion queues",
		TotalBytesMoved:   totalBytes,
		CommandsExecuted:  numCmds,
		DurationNanos:     dur.Nanoseconds(),
		ThroughputGBps:    0,
		IOPS:              0,
		StagingCopyCount:  stagingCopies,
		DoorbellRings:     slab.Queue().DoorbellRings(),
		ZeroCopyVerified:  false,
	}

	if jsonOut {
		data, _ := json.MarshalIndent(res, "", "  ")
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	fmt.Fprintln(stdout, "CUDA BaM NVMe P2PDMA Software Queue Simulation: MODELED")
	fmt.Fprintln(stdout, "  is_physical_silicon=false; no hardware was probed or executed.")
	fmt.Fprintf(stdout, "  Fixture Device: %s | Fixture Storage: %s\n", res.Device, res.Storage)
	fmt.Fprintf(stdout, "  Modeled Route: %s\n", res.RouteKind)
	fmt.Fprintf(stdout, "  Software Commands: %d | Bytes Represented: %d\n", res.CommandsExecuted, res.TotalBytesMoved)
	fmt.Fprintf(stdout, "  Software Queue Duration: %d ns\n", res.DurationNanos)
	fmt.Fprintf(stdout, "  Model Staging Copies: %d | Model Doorbell Rings: %d\n", res.StagingCopyCount, res.DoorbellRings)
	fmt.Fprintln(stdout, "  Physical Throughput / IOPS / Zero-Copy: UNKNOWN (not measured or verified)")
	return 0
}

func runCUDA_Qwen38Bench(stdout, stderr io.Writer, disco *compute.CUDAPCIeTopologyDiscovery, jsonOut bool) int {
	gpu := modeledCUDAGPU
	if len(disco.GPUs) > 0 {
		gpu = disco.GPUs[0]
	}

	coord27b, err := model.NewBlackwellModelCoordinator(model.ModelArchQwen38_27B, nil, nil)
	if err != nil {
		fmt.Fprintf(stderr, "cuda-gpudirect qwen38: failed to initialize 27B coordinator: %v\n", err)
		return 1
	}

	kvPages := make([][]byte, 16)
	for i := range kvPages {
		page := make([]byte, 1024)
		for j := range page {
			page[j] = byte((i*13 + j*17) & 0xFF)
		}
		kvPages[i] = page
	}
	gdnConv := []byte("qwen38-gdn-conv-state-cuda-direct-simulation")
	gdnRecurrent := []byte("qwen38-gdn-recurrent-state-cuda-direct-simulation")

	sessionID := "qwen38-cudadirect-simulation-session"
	tokenCount := 256
	desc, err := coord27b.SwapOut(sessionID, tokenCount, kvPages, gdnConv, gdnRecurrent)
	if err != nil {
		fmt.Fprintf(stderr, "cuda-gpudirect qwen38: SwapOut failed: %v\n", err)
		return 1
	}

	_, _, _, err = coord27b.SwapIn(desc)
	if err != nil {
		fmt.Fprintf(stderr, "cuda-gpudirect qwen38: SwapIn failed: %v\n", err)
		return 1
	}

	bytesMoved := desc.TotalBytes()

	coordMoE, err := model.NewBlackwellModelCoordinator(model.ModelArchQwen38FlashNext, nil, nil)
	if err != nil {
		fmt.Fprintf(stderr, "cuda-gpudirect qwen38: failed to initialize Flash Next coordinator: %v\n", err)
		return 1
	}
	if err := coordMoE.StreamMoEExperts(0, []int{2, 5, 8, 11}); err != nil {
		fmt.Fprintf(stderr, "cuda-gpudirect qwen38: StreamMoEExperts failed: %v\n", err)
		return 1
	}

	archName := fmt.Sprintf("%s (%s)", gpu.Arch, gpu.DeviceName)
	receipt := cudaQwen38ModeledReceipt(archName, bytesMoved, desc.StagingCopyCount())
	if jsonOut {
		data, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "cuda-gpudirect qwen38: json marshal failed: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(data))
		return 0
	}
	fmt.Fprintln(stdout, "Qwen3.8 CUDA Direct Storage & Cache Swap Software Simulation: MODELED")
	fmt.Fprintln(stdout, "  is_physical_silicon=false; no hardware was probed or executed.")
	fmt.Fprintf(stdout, "  Fixture Architecture: %s\n", receipt.Architecture)
	fmt.Fprintf(stdout, "  Software Descriptor Bytes: %d | Model Staging Copies: %d\n", receipt.BytesMoved, receipt.StagingCopyCount)
	fmt.Fprintln(stdout, "  Physical DMA, restoration integrity, bandwidth, token rates, latency and comparative speedups: UNKNOWN")
	for _, evidence := range receipt.Evidence {
		fmt.Fprintf(stdout, "  - %s\n", evidence)
	}
	return 0
}

func cudaQwen38ModeledReceipt(archName string, bytesMoved uint64, stagingCopies int) Qwen38CUDADirectSwapReceipt {
	// Legacy numeric keys stay numeric for compatibility. Zero means unmeasured,
	// not a throughput measurement, speedup, or proof of physical zero-copy DMA.
	baseline := Qwen38BenchArmMetrics{Name: "Baseline (not executed)", Details: "Unmeasured numeric compatibility placeholders."}
	reference := Qwen38BenchArmMetrics{Name: "Reference (not executed)", Details: "No reference engine was executed; no comparison was performed."}
	native := Qwen38BenchArmMetrics{Name: "fak-native (software model)", StagingCopyCount: stagingCopies, Details: "Software counter only; not proof of physical DMA or absence of host copies."}
	return Qwen38CUDADirectSwapReceipt{
		Schema: Qwen38CUDADirectSwapReceiptSchema, Provenance: "MODELED", Verdict: "MODELED", IsPhysicalSilicon: false,
		Model: "Qwen3.8 (27B & Flash Next)", Architecture: archName,
		StagingCopyCount: stagingCopies, BytesMoved: bytesMoved,
		UnknownFields: []string{"direct_dma_bandwidth_gbps", "speedup_vs_baseline", "speedup_vs_reference", "arms.*.performance", "fak_native.performance", "baseline", "reference", "physical_p2p_dma", "restoration_integrity"},
		Arms:          map[string]Qwen38BenchArmMetrics{"baseline": baseline, "fak_native": native, "reference": reference},
		Baseline:      baseline, FakNative: native, Reference: reference,
		Evidence: []string{
			"Software swap and expert-streaming methods returned without error; no model inference was executed.",
			"Descriptor sizes and staging counters describe the software model only.",
			"No physical P2P DMA, byte-for-byte restoration, performance, or comparison was verified.",
		},
	}
}
