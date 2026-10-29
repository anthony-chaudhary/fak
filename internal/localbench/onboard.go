package localbench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// Onboarding writes a machine registry entry plus a smoke receipt under
// experiments/benchmark/machines/<machine-id>/ relative to the working directory
// (run from the repo root), so tools/bench_catalog.py picks the machine up with no
// further migration. The specs.json layout follows tools/schemas/machine-specs.v1.json
// ($schema benchmark/machine-specs.v1); the smoke receipt reuses the receipt v1
// wrapper so `fak bench local verify` checks it like any other receipt.
//
// Idempotency choice (documented): specs.json is updated in place, but
// registered_at and the first-seen hostname are preserved from any existing file;
// an existing unreadable specs.json is refused, never clobbered. Each run
// version-stamps a NEW smoke receipt into onboard-receipts/ and never rewrites or
// deletes prior receipts, so re-runs append history instead of corrupting it.

const (
	machineSpecsSchema = "benchmark/machine-specs.v1"
	onboardBenchmark   = "onboard-smoke"
)

var (
	machineIDPattern = regexp.MustCompile(`^[a-z0-9-]+$`)
	cudaArchPattern  = regexp.MustCompile(`^sm_[0-9]{3}$`)
	rtx5090Compact   = regexp.MustCompile(`[^a-z0-9]`)
)

// MachineSpecs mirrors tools/schemas/machine-specs.v1.json.
type MachineSpecs struct {
	Schema       string        `json:"$schema"`
	MachineID    string        `json:"machine_id"`
	Hostname     string        `json:"hostname"`
	RegisteredAt string        `json:"registered_at"`
	Hardware     SpecsHardware `json:"hardware"`
	OS           SpecsOS       `json:"os"`
	Runtime      SpecsRuntime  `json:"runtime"`
	Tags         []string      `json:"tags"`
}

type SpecsHardware struct {
	CPU   SpecsCPU   `json:"cpu"`
	GPU   []SpecsGPU `json:"gpu,omitempty"`
	RAMGB float64    `json:"ram_gb"`
}

type SpecsCPU struct {
	Model         string `json:"model"`
	CoresPhysical int    `json:"cores_physical"`
	CoresLogical  int    `json:"cores_logical"`
	Architecture  string `json:"architecture"`
}

type SpecsGPU struct {
	Model             string  `json:"model"`
	MemoryGB          float64 `json:"memory_gb,omitempty"`
	ComputeCapability string  `json:"compute_capability,omitempty"`
}

type SpecsOS struct {
	Name string `json:"name"`
}

type SpecsRuntime struct {
	GoVersion string `json:"go_version"`
}

// OnboardSummary is the --json output of `fak bench local onboard`.
type OnboardSummary struct {
	MachineID     string  `json:"machine_id"`
	Arch          string  `json:"arch,omitempty"`
	GPUModel      string  `json:"gpu_model,omitempty"`
	VRAMGiB       float64 `json:"vram_gib,omitempty"`
	SpecsPath     string  `json:"specs_path"`
	ReceiptPath   string  `json:"receipt_path"`
	ReceiptSHA256 string  `json:"receipt_sha256"`
}

func onboardCommand(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("onboard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	machineID := fs.String("machine-id", "", "machine slug matching ^[a-z0-9-]+$; becomes the registry directory name")
	gpuModel := fs.String("gpu-model", "", "declared GPU model, e.g. 'NVIDIA GeForce RTX 5090'")
	vramGIB := fs.Float64("vram-gib", 0, "declared GPU VRAM in GiB")
	arch := fs.String("arch", "", "CUDA arch tag sm_NNN; defaults to sm_120 when --gpu-model matches an RTX 5090")
	asJSON := fs.Bool("json", false, "print the onboarding summary as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected onboard argument %q", fs.Arg(0))
	}

	id := strings.TrimSpace(*machineID)
	if !machineIDPattern.MatchString(id) {
		return errors.New("--machine-id is required and must match ^[a-z0-9-]+$")
	}
	if *vramGIB < 0 {
		return errors.New("--vram-gib must be non-negative")
	}
	gpu := normalizeHardwareText(*gpuModel)
	a := strings.TrimSpace(*arch)
	if a == "" && isRTX5090(gpu) {
		a = "sm_120" // consumer Blackwell; see make cuda-build-sm120
	}
	if a != "" && !cudaArchPattern.MatchString(a) {
		return fmt.Errorf("--arch %q must look like sm_NNN", *arch)
	}

	host, _ := os.Hostname()
	if host == "" {
		host = "unknown"
	}
	now := time.Now().UTC()

	specsDir := filepath.Join("experiments", "benchmark", "machines", id)
	specsPath := filepath.Join(specsDir, "specs.json")

	prior, err := readExistingSpecs(specsPath)
	if err != nil {
		return fmt.Errorf("%s exists but cannot be parsed; refusing to clobber it: %w", specsPath, err)
	}
	registeredAt := now.Format(time.RFC3339)
	if prior != nil {
		registeredAt = prior.RegisteredAt
		if prior.Hostname != "" {
			host = prior.Hostname
		}
	}

	hw := inventoryHardware()
	specs := MachineSpecs{
		Schema:       machineSpecsSchema,
		MachineID:    id,
		Hostname:     host,
		RegisteredAt: registeredAt,
		Hardware: SpecsHardware{
			CPU: SpecsCPU{
				Model:         hw.CPU,
				CoresPhysical: physicalCoreCount(),
				CoresLogical:  runtime.NumCPU(),
				Architecture:  goArchToISA(),
			},
			RAMGB: round1(float64(hw.MemoryBytes) / (1 << 30)),
		},
		OS:      SpecsOS{Name: runtime.GOOS},
		Runtime: SpecsRuntime{GoVersion: runtime.Version()},
	}
	if gpu != "" {
		entry := SpecsGPU{Model: gpu}
		if *vramGIB > 0 {
			entry.MemoryGB = *vramGIB
		}
		if cc := computeCapability(a); cc != "" {
			entry.ComputeCapability = cc
		}
		specs.Hardware.GPU = []SpecsGPU{entry}
	}
	specs.Tags = onboardTags(a)

	data, err := json.MarshalIndent(specs, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(specsDir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(specsPath, data, 0o644); err != nil {
		return err
	}
	// Round-trip the bytes just written before claiming success: the spec must
	// parse back with the declared facts intact.
	var back MachineSpecs
	if err := json.Unmarshal(data, &back); err != nil {
		return fmt.Errorf("specs.json failed round-trip parse: %w", err)
	}
	if back.MachineID != id || back.Hardware.CPU.Architecture == "" {
		return fmt.Errorf("specs.json round-trip lost machine_id/arch")
	}

	// Smoke receipt: the onboarding invocation itself, sealed with the receipt v1
	// wrapper so `fak bench local verify` accepts it. Written only on success;
	// failures leave prior receipts untouched.
	smoke := fmt.Sprintf("machine=%s os=%s/%s go=%s gpu=%s vram_gib=%g arch=%s\nspecs=%s\n",
		id, runtime.GOOS, runtime.GOARCH, runtime.Version(), displayUnset(gpu), *vramGIB, displayUnset(a), filepath.ToSlash(specsPath))
	started := now
	r := Receipt{
		Schema:     receiptSchema,
		Benchmark:  onboardBenchmark,
		Engine:     engineLabel(a),
		StartedAt:  started.Format(time.RFC3339Nano),
		FinishedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Command:    scrubArgs(onboardCommandLine(id, gpu, *vramGIB, a)),
		ExitStatus: 0,
		OutputSHA256: func() string {
			sum := sha256.Sum256([]byte(smoke))
			return hex.EncodeToString(sum[:])
		}(),
		OutputBytes: int64(len(smoke)),
		Output:      smoke,
		Hardware:    hw,
		Provenance:  inventoryProvenance("fak"),
	}
	r.DurationMS = time.Since(started).Milliseconds()
	if err := seal(&r); err != nil {
		return err
	}
	receiptsDir := filepath.Join(specsDir, "onboard-receipts")
	if err := os.MkdirAll(receiptsDir, 0o755); err != nil {
		return err
	}
	receiptPath := filepath.Join(receiptsDir, "onboard-receipt-"+now.Format("20060102T150405.000000000Z")+".json")
	if err := writeReceipt(receiptPath, r); err != nil {
		return err
	}

	specsRel := filepath.ToSlash(specsPath)
	receiptRel := filepath.ToSlash(receiptPath)
	if *asJSON {
		return writeJSON(stdout, OnboardSummary{
			MachineID: id, Arch: a, GPUModel: gpu, VRAMGiB: *vramGIB,
			SpecsPath: specsRel, ReceiptPath: receiptRel, ReceiptSHA256: r.Integrity.SHA256,
		})
	}
	fmt.Fprintf(stdout, "onboarded %s: arch=%s gpu=%s vram_gib=%g\n", id, displayUnset(a), displayUnset(gpu), *vramGIB)
	fmt.Fprintf(stdout, "specs:   %s\n", specsRel)
	fmt.Fprintf(stdout, "receipt: %s\n", receiptRel)
	fmt.Fprintf(stdout, "verify:  fak bench local verify %s\n", receiptRel)
	return nil
}

// readExistingSpecs returns the prior specs for registered_at/hostname
// preservation. A missing file yields (nil, nil); an unreadable one is an error.
func readExistingSpecs(path string) (*MachineSpecs, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var s MachineSpecs
	if err := json.Unmarshal(bytes.TrimSpace(data), &s); err != nil {
		return nil, err
	}
	return &s, nil
}

func isRTX5090(model string) bool {
	compact := rtx5090Compact.ReplaceAllString(strings.ToLower(model), "")
	return strings.Contains(compact, "rtx5090")
}

// computeCapability renders sm_NNN as CUDA "major.minor" (sm_120 -> 12.0,
// sm_89 -> 8.9); empty arch yields "".
func computeCapability(arch string) string {
	if arch == "" {
		return ""
	}
	d, err := strconv.Atoi(strings.TrimPrefix(arch, "sm_"))
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%d.%d", d/10, d%10)
}

func engineLabel(arch string) string {
	if arch != "" {
		return arch
	}
	return "onboard"
}

func onboardTags(arch string) []string {
	tags := []string{goArchToISA()}
	if arch != "" {
		tags = append(tags, "cuda", arch)
	}
	tags = append(tags, strings.ToLower(runtime.GOOS))
	return tags
}

func goArchToISA() string {
	switch runtime.GOARCH {
	case "amd64":
		return "x86_64"
	case "arm64":
		return "arm64"
	default:
		return runtime.GOARCH
	}
}

// physicalCoreCount probes the physical core count; when the probe is
// unavailable it falls back to the logical count (specs.json stays valid and the
// operator can correct the field by hand).
func physicalCoreCount() int {
	logical := runtime.NumCPU()
	switch runtime.GOOS {
	case "darwin":
		if n := parseUint(commandLine("sysctl", "-n", "hw.physicalcpu")); n > 0 {
			return int(n)
		}
	case "windows":
		if n := parseUint(commandLine("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")); n > 0 {
			return int(n)
		}
	case "linux":
		if n := linuxPhysicalCores(); n > 0 {
			return n
		}
	}
	return logical
}

// linuxPhysicalCores counts distinct (physical id, core id) pairs in /proc/cpuinfo.
func linuxPhysicalCores() int {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return 0
	}
	seen := map[string]bool{}
	phys := ""
	for _, line := range strings.Split(string(data), "\n") {
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key, val = strings.ToLower(strings.TrimSpace(key)), strings.TrimSpace(val)
		switch key {
		case "physical id":
			phys = val
		case "core id":
			seen[phys+"\x00"+val] = true
		}
	}
	return len(seen)
}

func round1(f float64) float64 { return float64(int64(f*10+0.5)) / 10 }

func onboardCommandLine(id, gpu string, vram float64, arch string) []string {
	cmd := []string{"fak", "bench", "local", "onboard", "--machine-id", id}
	if gpu != "" {
		cmd = append(cmd, "--gpu-model", gpu)
	}
	if vram > 0 {
		cmd = append(cmd, "--vram-gib", strconv.FormatFloat(vram, 'f', -1, 64))
	}
	if arch != "" {
		cmd = append(cmd, "--arch", arch)
	}
	return cmd
}
