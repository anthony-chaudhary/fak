package localbench

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
)

const (
	receiptSchema = "fak.local-hardware-benchmark.receipt/v1"
	issueURL      = "https://github.com/anthony-chaudhary/fak/issues/new"
)

type Receipt struct {
	Schema       string     `json:"schema"`
	StartedAt    string     `json:"started_at"`
	FinishedAt   string     `json:"finished_at"`
	DurationMS   int64      `json:"duration_ms"`
	Benchmark    string     `json:"benchmark,omitempty"`
	Engine       string     `json:"engine,omitempty"`
	Command      []string   `json:"command"`
	ExitStatus   int        `json:"exit_status"`
	OutputSHA256 string     `json:"output_sha256"`
	OutputBytes  int64      `json:"output_bytes"`
	Output       string     `json:"output_scrubbed,omitempty"`
	Hardware     Hardware   `json:"hardware"`
	Provenance   Provenance `json:"provenance"`
	Integrity    Integrity  `json:"integrity"`
}

type Hardware struct {
	OS           string            `json:"os"`
	Arch         string            `json:"arch"`
	CPU          string            `json:"cpu"`
	MemoryBytes  uint64            `json:"memory_bytes,omitempty"`
	Accelerators []Accelerator     `json:"accelerators,omitempty"`
	Toolchains   map[string]string `json:"toolchains,omitempty"`
}

type Accelerator struct {
	Vendor  string `json:"vendor"`
	Kind    string `json:"kind"`
	Model   string `json:"model"`
	Backend string `json:"backend,omitempty"`
}

type Provenance struct {
	FakVersion   string `json:"fak_version"`
	FakRevision  string `json:"fak_revision,omitempty"`
	RepoRevision string `json:"repo_revision,omitempty"`
	GoVersion    string `json:"go_version"`
}

type Integrity struct {
	Algorithm string `json:"algorithm"`
	SHA256    string `json:"sha256"`
}

// RunCLI executes the local benchmark receipt workflow without uploading data.
// The caller owns process exit handling so this package can be reused by fak and
// the standalone example.
func RunCLI(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		usage(stderr)
		return errors.New("choose inventory, run, verify, or submit")
	}
	switch args[0] {
	case "run":
		return runCommand(args[1:], stdout, stderr)
	case "inventory":
		fs := flag.NewFlagSet("inventory", flag.ContinueOnError)
		fs.SetOutput(stderr)
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if fs.NArg() != 0 {
			return errors.New("usage: fak bench local inventory")
		}
		return writeJSON(stdout, Inventory{Hardware: inventoryHardware(), Benchmarks: catalogInventory()})
	case "verify":
		fs := flag.NewFlagSet("verify", flag.ContinueOnError)
		fs.SetOutput(stderr)
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if fs.NArg() != 1 {
			return errors.New("usage: fak bench local verify RECEIPT.json")
		}
		r, err := readAndVerify(fs.Arg(0))
		if err != nil {
			return err
		}
		fmt.Fprintf(stdout, "VERIFIED %s %s benchmark=%s engine=%s exit=%d\n", r.Schema, r.Integrity.SHA256, displayUnset(r.Benchmark), displayUnset(r.Engine), r.ExitStatus)
		return nil
	case "submit":
		fs := flag.NewFlagSet("submit", flag.ContinueOnError)
		fs.SetOutput(stderr)
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if fs.NArg() != 1 {
			return errors.New("usage: fak bench local submit RECEIPT.json")
		}
		r, err := readAndVerify(fs.Arg(0))
		if err != nil {
			return err
		}
		body, submissionURL, err := renderSubmission(r)
		if err != nil {
			return err
		}
		fmt.Fprintln(stdout, body)
		fmt.Fprintln(stdout, "\n---\nOpen this URL to review and explicitly submit (this tool never uploads):")
		fmt.Fprintln(stdout, submissionURL)
		return nil
	case "help", "-h", "--help":
		usage(stdout)
		return nil
	default:
		usage(stderr)
		return fmt.Errorf("unknown command %q", args[0])
	}
}

// Inventory is a deterministic local hardware snapshot plus the benchmark
// catalog choices accepted by run.
type Inventory struct {
	Hardware   Hardware       `json:"hardware"`
	Benchmarks []CatalogEntry `json:"benchmarks"`
}

// CatalogEntry is the discovery subset needed to choose a local run. Run is a
// recipe only; the operator must still provide the exact command to execute.
type CatalogEntry struct {
	Name    string `json:"name"`
	Kind    string `json:"kind"`
	Need    string `json:"need"`
	Summary string `json:"summary"`
	Run     string `json:"run"`
	Manual  bool   `json:"manual,omitempty"`
}

func catalogInventory() []CatalogEntry {
	all := benchcatalog.All()
	out := make([]CatalogEntry, 0, len(all))
	for _, b := range all {
		out = append(out, CatalogEntry{Name: b.Name, Kind: string(b.Kind), Need: string(b.Need), Summary: b.Summary, Run: b.Run, Manual: b.Manual})
	}
	return out
}

func usage(w io.Writer) {
	fmt.Fprintln(w, `fak bench local: explicit, no-upload local benchmark receipts

  fak bench local inventory
  fak bench local run --benchmark NAME --engine LABEL --out receipt.json -- COMMAND [ARGS...]
  fak bench local verify receipt.json
  fak bench local submit receipt.json

inventory discovers benchmark names from fak's benchmark catalog. run records the
operator-selected catalog name, engine label, and exact command. It never selects,
switches, or falls back to another engine. submit prints a deterministic review
packet and URL; it never uploads.`)
}

func runCommand(args []string, stdout, stderr io.Writer) error {
	separator := -1
	for i, arg := range args {
		if arg == "--" {
			separator = i
			break
		}
	}
	if separator < 0 {
		return errors.New("an explicit child command is required after --")
	}
	flagArgs, command := args[:separator], args[separator+1:]

	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	benchmark := fs.String("benchmark", "", "benchmark catalog name (see inventory)")
	engine := fs.String("engine", "", "explicit engine label used by the selected command")
	out := fs.String("out", "local-hardware-receipt.json", "receipt output path")
	maxOutput := fs.Int("max-output", 32768, "maximum scrubbed output bytes retained in receipt")
	if err := fs.Parse(flagArgs); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected run argument %q before --", fs.Arg(0))
	}
	if strings.TrimSpace(*benchmark) == "" {
		return errors.New("--benchmark is required; choose a catalog name from `fak bench local inventory`")
	}
	if _, ok := benchcatalog.Get(*benchmark); !ok {
		return fmt.Errorf("unknown benchmark %q; choose a catalog name from `fak bench local inventory`", *benchmark)
	}
	if strings.TrimSpace(*engine) == "" {
		return errors.New("--engine is required and must label the engine selected by the operator")
	}
	if scrub(*engine) != strings.TrimSpace(*engine) {
		return errors.New("--engine must be a non-sensitive stable label")
	}
	if len(command) == 0 {
		return errors.New("an explicit child command is required after --")
	}
	if *maxOutput < 0 {
		return errors.New("--max-output must be non-negative")
	}

	started := time.Now().UTC()
	cmd := exec.Command(command[0], command[1:]...)
	var raw bytes.Buffer
	cmd.Stdout = io.MultiWriter(stdout, &raw)
	cmd.Stderr = io.MultiWriter(stderr, &raw)
	runErr := cmd.Run()
	finished := time.Now().UTC()
	exitStatus := 0
	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			exitStatus = exitErr.ExitCode()
		} else {
			exitStatus = -1
		}
	}
	outputDigest := sha256.Sum256(raw.Bytes())
	scrubbedOutput := scrub(string(raw.Bytes()))
	if len(scrubbedOutput) > *maxOutput {
		scrubbedOutput = scrubbedOutput[:*maxOutput] + "\n[TRUNCATED]"
	}
	r := Receipt{
		Schema: receiptSchema, Benchmark: *benchmark, Engine: strings.TrimSpace(*engine),
		StartedAt: started.Format(time.RFC3339Nano), FinishedAt: finished.Format(time.RFC3339Nano),
		DurationMS: finished.Sub(started).Milliseconds(), Command: scrubArgs(command), ExitStatus: exitStatus,
		OutputSHA256: hex.EncodeToString(outputDigest[:]), OutputBytes: int64(raw.Len()), Output: scrubbedOutput,
		Hardware: inventoryHardware(), Provenance: inventoryProvenance(command[0]),
	}
	if err := seal(&r); err != nil {
		return err
	}
	if err := writeReceipt(*out, r); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "\nreceipt: %s\nverify: fak bench local verify %s\n", *out, *out)
	if runErr != nil {
		return fmt.Errorf("child command exited with status %d; receipt was still written", exitStatus)
	}
	return nil
}

func inventoryHardware() Hardware {
	h := Hardware{OS: runtime.GOOS, Arch: runtime.GOARCH, CPU: "unknown", Toolchains: map[string]string{}}
	switch runtime.GOOS {
	case "linux":
		h.CPU = firstValue(readKeyValue("/proc/cpuinfo", "model name"), readKeyValue("/proc/cpuinfo", "Hardware"), runtime.GOARCH)
		h.MemoryBytes = linuxMemory()
	case "darwin":
		h.CPU = firstValue(commandLine("sysctl", "-n", "machdep.cpu.brand_string"), commandLine("sysctl", "-n", "hw.model"), runtime.GOARCH)
		h.MemoryBytes = parseUint(commandLine("sysctl", "-n", "hw.memsize"))
	case "windows":
		h.CPU = firstValue(commandLine("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"), runtime.GOARCH)
		h.MemoryBytes = parseUint(commandLine("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"))
	default:
		h.CPU = runtime.GOARCH
	}
	h.CPU = normalizeHardwareText(h.CPU)
	h.Accelerators = detectAccelerators()
	for name, probe := range map[string][]string{
		"cuda": {"nvcc", "--version"}, "rocm": {"rocminfo", "--version"}, "vulkan": {"vulkaninfo", "--summary"},
		"metal": {"xcrun", "metal", "--version"}, "oneapi": {"sycl-ls", "--version"},
	} {
		if v := commandLine(probe[0], probe[1:]...); v != "" {
			h.Toolchains[name] = normalizeVersion(v)
		}
	}
	if len(h.Toolchains) == 0 {
		h.Toolchains = nil
	}
	return h
}

func detectAccelerators() []Accelerator {
	var found []Accelerator
	if runtime.GOOS == "darwin" {
		found = append(found, parseAppleAccelerators(commandLine("system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"))...)
	}
	found = append(found, parseNVIDIAAccelerators(commandLine("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"))...)
	found = append(found, parseAMDAccelerators(commandLine("rocminfo"))...)
	found = append(found, parseIntelAccelerators(commandLine("sycl-ls"))...)
	return dedupeAccelerators(found)
}

func parseAppleAccelerators(text string) []Accelerator {
	var found []Accelerator
	for _, model := range valuesAfter(text, "Chipset Model:") {
		vendor := "Apple"
		if strings.Contains(strings.ToLower(model), "amd") {
			vendor = "AMD"
		}
		found = append(found, Accelerator{Vendor: vendor, Kind: "gpu", Model: normalizeHardwareText(model), Backend: "Metal"})
	}
	return dedupeAccelerators(found)
}

func parseNVIDIAAccelerators(text string) []Accelerator {
	var found []Accelerator
	for _, line := range nonemptyLines(text) {
		parts := strings.SplitN(line, ",", 2)
		model := normalizeHardwareText(parts[0])
		if model != "" {
			found = append(found, Accelerator{Vendor: "NVIDIA", Kind: "gpu", Model: model, Backend: "CUDA"})
		}
	}
	return dedupeAccelerators(found)
}

func parseAMDAccelerators(text string) []Accelerator {
	var found []Accelerator
	for _, line := range nonemptyLines(text) {
		if strings.Contains(strings.ToLower(line), "marketing name:") {
			model := normalizeHardwareText(strings.TrimSpace(strings.SplitN(line, ":", 2)[1]))
			if model != "" {
				found = append(found, Accelerator{Vendor: "AMD", Kind: "gpu", Model: model, Backend: "ROCm"})
			}
		}
	}
	return dedupeAccelerators(found)
}

func parseIntelAccelerators(text string) []Accelerator {
	var found []Accelerator
	for _, line := range nonemptyLines(text) {
		if strings.Contains(strings.ToLower(line), "intel") {
			found = append(found, Accelerator{Vendor: "Intel", Kind: "accelerator", Model: normalizeHardwareText(line), Backend: "oneAPI/SYCL"})
		}
	}
	return dedupeAccelerators(found)
}
func dedupeAccelerators(in []Accelerator) []Accelerator {
	seen := map[string]bool{}
	out := make([]Accelerator, 0, len(in))
	for _, a := range in {
		key := a.Vendor + "\x00" + a.Model + "\x00" + a.Backend
		if a.Model == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Vendor+out[i].Model+out[i].Backend < out[j].Vendor+out[j].Model+out[j].Backend
	})
	return out
}

func inventoryProvenance(command string) Provenance {
	p := Provenance{GoVersion: runtime.Version(), RepoRevision: commandLine("git", "rev-parse", "HEAD")}
	candidates := []string{"fak"}
	if base := strings.ToLower(filepath.Base(command)); base == "fak" || base == "fak.exe" {
		candidates = append([]string{command}, candidates...)
	}
	for _, candidate := range candidates {
		text := commandLine(candidate, "version")
		if text == "" {
			continue
		}
		lines := nonemptyLines(text)
		if len(lines) > 0 {
			p.FakVersion = normalizeHardwareText(lines[0])
		}
		re := regexp.MustCompile(`(?i)\b[0-9a-f]{7,40}\b`)
		if m := re.FindString(text); m != "" {
			p.FakRevision = m
		}
		break
	}
	if p.FakVersion == "" {
		p.FakVersion = "unavailable"
	}
	return p
}

func seal(r *Receipt) error {
	r.Integrity = Integrity{}
	payload, err := canonicalReceipt(*r)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(payload)
	r.Integrity = Integrity{Algorithm: "sha256", SHA256: hex.EncodeToString(sum[:])}
	return nil
}

func verify(r Receipt) error {
	if r.Schema != receiptSchema {
		return fmt.Errorf("unsupported schema %q", r.Schema)
	}
	if r.Integrity.Algorithm != "sha256" || len(r.Integrity.SHA256) != 64 {
		return errors.New("invalid integrity block")
	}
	want := r.Integrity.SHA256
	r.Integrity = Integrity{}
	payload, err := canonicalReceipt(r)
	if err != nil {
		return err
	}
	got := sha256.Sum256(payload)
	if !strings.EqualFold(want, hex.EncodeToString(got[:])) {
		return errors.New("receipt integrity mismatch (tampered or corrupted)")
	}
	return nil
}

func canonicalReceipt(r Receipt) ([]byte, error) { return json.Marshal(r) }

func readAndVerify(path string) (Receipt, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Receipt{}, err
	}
	var r Receipt
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return Receipt{}, err
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return Receipt{}, errors.New("receipt contains trailing JSON data")
		}
		return Receipt{}, fmt.Errorf("receipt trailing data: %w", err)
	}
	if err := verify(r); err != nil {
		return Receipt{}, err
	}
	return r, nil
}

func writeReceipt(path string, r Receipt) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}
func writeJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func renderSubmission(r Receipt) (string, string, error) {
	if err := verify(r); err != nil {
		return "", "", err
	}
	accelerators := "none reported (CPU-only or probe unavailable)"
	if len(r.Hardware.Accelerators) > 0 {
		parts := make([]string, 0, len(r.Hardware.Accelerators))
		for _, a := range r.Hardware.Accelerators {
			parts = append(parts, strings.TrimSpace(a.Vendor+" "+a.Model+" ("+a.Backend+")"))
		}
		accelerators = strings.Join(parts, "; ")
	}
	body := fmt.Sprintf("## Local hardware benchmark receipt\n\n"+
		"- Schema: `%s`\n"+
		"- Receipt SHA-256: `%s`\n"+
		"- Benchmark: `%s`\n"+
		"- Engine: `%s`\n"+
		"- Command: `%s`\n"+
		"- Exit status: `%d`\n"+
		"- Output SHA-256: `%s`\n"+
		"- Hardware: `%s/%s`; CPU `%s`; memory `%d` bytes\n"+
		"- Accelerators: %s\n"+
		"- fak: `%s` (`%s`)\n"+
		"- repo revision: `%s`\n\n"+
		"The receipt was generated locally by `fak bench local` (promoted from `examples/local-hardware-benchmark`). The tool did not upload it. I reviewed this packet for privacy before submission.\n\n"+
		"Related: #10421, #10444", r.Schema, r.Integrity.SHA256, displayUnset(r.Benchmark), displayUnset(r.Engine), shellDisplay(r.Command), r.ExitStatus, r.OutputSHA256,
		r.Hardware.OS, r.Hardware.Arch, r.Hardware.CPU, r.Hardware.MemoryBytes, accelerators,
		r.Provenance.FakVersion, r.Provenance.FakRevision, r.Provenance.RepoRevision)
	title := fmt.Sprintf("bench: local hardware receipt %s/%s %s", r.Hardware.OS, r.Hardware.Arch, shortHash(r.Integrity.SHA256))
	q := url.Values{}
	q.Set("title", title)
	q.Set("body", body)
	q.Set("labels", "benchmark")
	return body, issueURL + "?" + q.Encode(), nil
}

func scrubArgs(args []string) []string {
	out := make([]string, len(args))
	secretNext := false
	for i, arg := range args {
		if secretNext {
			out[i] = "[REDACTED]"
			secretNext = false
			continue
		}
		lower := strings.ToLower(arg)
		if key, _, ok := strings.Cut(arg, "="); ok && isSecretFlag(strings.ToLower(key)) {
			out[i] = key + "=[REDACTED]"
			continue
		}
		if isSecretFlag(lower) {
			out[i] = arg
			secretNext = true
			continue
		}
		out[i] = scrub(arg)
	}
	return out
}

func isSecretFlag(s string) bool {
	s = strings.TrimLeft(s, "-")
	for _, word := range []string{"token", "password", "passwd", "secret", "api-key", "apikey", "authorization", "credential"} {
		if s == word || strings.HasSuffix(s, "_"+word) || strings.HasSuffix(s, "-"+word) {
			return true
		}
	}
	return false
}

var (
	secretAssignment = regexp.MustCompile(`(?i)\b([A-Z0-9_]*(?:TOKEN|PASSWORD|PASSWD|SECRET|API_KEY|APIKEY|AUTHORIZATION|CREDENTIAL)[A-Z0-9_]*)\s*[:=]\s*([^\s,;]+)`)
	bearer           = regexp.MustCompile(`(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+`)
	windowsUserPath  = regexp.MustCompile(`(?i)\b[A-Z]:\\Users\\[^\\\s]+`)
	windowsAbsPath   = regexp.MustCompile(`(?i)\b[A-Z]:\\(?:[^\\\r\n]+\\)*[^\\\r\n\s|]+`)
	unixHomePath     = regexp.MustCompile(`/(?:Users|home)/[^/\s]+`)
	uncUserPath      = regexp.MustCompile(`(?i)\\\\[^\\\s]+\\(?:Users|home)\\[^\\\s]+`)
	uncAbsPath       = regexp.MustCompile(`(?i)\\\\[^\\\s]+\\[^\r\n|]+`)
	volatileID       = regexp.MustCompile(`(?i)\b(?:hostname|computername|user(?:name)?|machine[-_ ]?id)\s*[:=]\s*[^\s,;]+`)
)

func scrub(s string) string {
	s = bearer.ReplaceAllString(s, "Bearer-[REDACTED]")
	s = secretAssignment.ReplaceAllString(s, "$1=[REDACTED]")
	s = windowsUserPath.ReplaceAllString(s, `[HOME]`)
	s = windowsAbsPath.ReplaceAllString(s, `[LOCAL_PATH]`)
	s = unixHomePath.ReplaceAllString(s, `[HOME]`)
	s = uncUserPath.ReplaceAllString(s, `[HOME]`)
	s = uncAbsPath.ReplaceAllString(s, `[LOCAL_PATH]`)
	s = volatileID.ReplaceAllStringFunc(s, func(v string) string {
		if i := strings.IndexAny(v, ":="); i >= 0 {
			return v[:i+1] + "[REDACTED]"
		}
		return "[REDACTED]"
	})
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		s = strings.ReplaceAll(s, home, "[HOME]")
		s = strings.ReplaceAll(s, filepath.ToSlash(home), "[HOME]")
	}
	return strings.TrimSpace(s)
}

func normalizeHardwareText(s string) string {
	s = scrub(strings.ReplaceAll(strings.ReplaceAll(s, "\x00", " "), "\r", ""))
	fields := strings.Fields(s)
	s = strings.Join(fields, " ")
	s = regexp.MustCompile(`(?i)\s+@\s+[0-9.]+\s*GHz`).ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

func normalizeVersion(s string) string {
	lines := nonemptyLines(scrub(s))
	if len(lines) == 0 {
		return ""
	}
	if len(lines) > 3 {
		lines = lines[:3]
	}
	return strings.Join(lines, " | ")
}

func commandLine(name string, args ...string) string {
	if _, err := exec.LookPath(name); err != nil {
		return ""
	}
	cmd := exec.Command(name, args...)
	cmd.Env = append(os.Environ(), "LC_ALL=C", "LANG=C")
	out, err := cmd.CombinedOutput()
	if err != nil && len(out) == 0 {
		return ""
	}
	return strings.TrimSpace(scrub(string(out)))
}

func readKeyValue(path, key string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.SplitN(scanner.Text(), ":", 2)
		if len(parts) == 2 && strings.EqualFold(strings.TrimSpace(parts[0]), key) {
			return strings.TrimSpace(parts[1])
		}
	}
	return ""
}
func linuxMemory() uint64 {
	kb := readKeyValue("/proc/meminfo", "MemTotal")
	fields := strings.Fields(kb)
	if len(fields) == 0 {
		return 0
	}
	n, _ := strconv.ParseUint(fields[0], 10, 64)
	return n * 1024
}
func parseUint(s string) uint64 {
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return 0
	}
	n, _ := strconv.ParseUint(fields[0], 10, 64)
	return n
}
func firstValue(v ...string) string {
	for _, x := range v {
		if strings.TrimSpace(x) != "" {
			return strings.TrimSpace(x)
		}
	}
	return "unknown"
}
func nonemptyLines(s string) []string {
	var out []string
	for _, line := range strings.Split(strings.ReplaceAll(s, "\r", ""), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			out = append(out, line)
		}
	}
	return out
}
func valuesAfter(s, prefix string) []string {
	var out []string
	for _, line := range nonemptyLines(s) {
		if i := strings.Index(line, prefix); i >= 0 {
			out = append(out, strings.TrimSpace(line[i+len(prefix):]))
		}
	}
	return out
}
func displayUnset(s string) string {
	if s == "" {
		return "legacy-v1-unspecified"
	}
	return s
}

func shortHash(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}
func shellDisplay(args []string) string { b, _ := json.Marshal(args); return string(b) }
