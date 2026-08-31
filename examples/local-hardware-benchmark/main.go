package main

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

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func runCLI(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		usage(stderr)
		return errors.New("choose run, inventory, verify, or submit")
	}
	switch args[0] {
	case "run":
		return runCommand(args[1:], stdout, stderr)
	case "inventory":
		h := inventoryHardware()
		return writeJSON(stdout, h)
	case "verify":
		fs := flag.NewFlagSet("verify", flag.ContinueOnError)
		fs.SetOutput(stderr)
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if fs.NArg() != 1 {
			return errors.New("usage: verify RECEIPT.json")
		}
		r, err := readAndVerify(fs.Arg(0))
		if err != nil {
			return err
		}
		fmt.Fprintf(stdout, "VERIFIED %s %s exit=%d\n", r.Schema, r.Integrity.SHA256, r.ExitStatus)
		return nil
	case "submit":
		fs := flag.NewFlagSet("submit", flag.ContinueOnError)
		fs.SetOutput(stderr)
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		if fs.NArg() != 1 {
			return errors.New("usage: submit RECEIPT.json")
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

func usage(w io.Writer) {
	fmt.Fprintln(w, `local-hardware-benchmark: explicit local benchmark receipts

  go run . inventory
  go run . run --out receipt.json -- <operator-selected command> [args...]
  go run . verify receipt.json
  go run . submit receipt.json

The command after -- is run exactly as selected. No engine fallback or upload occurs.`)
}

func runCommand(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	out := fs.String("out", "local-hardware-receipt.json", "receipt output path")
	maxOutput := fs.Int("max-output", 32768, "maximum scrubbed output bytes retained in receipt")
	if err := fs.Parse(args); err != nil {
		return err
	}
	command := fs.Args()
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
		Schema:    receiptSchema,
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
	fmt.Fprintf(stdout, "\nreceipt: %s\nverify: go run . verify %s\n", *out, *out)
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
		"- Command: `%s`\n"+
		"- Exit status: `%d`\n"+
		"- Output SHA-256: `%s`\n"+
		"- Hardware: `%s/%s`; CPU `%s`; memory `%d` bytes\n"+
		"- Accelerators: %s\n"+
		"- fak: `%s` (`%s`)\n"+
		"- repo revision: `%s`\n\n"+
		"The receipt was generated locally by `examples/local-hardware-benchmark`. The tool did not upload it. I reviewed this packet for privacy before submission.\n\n"+
		"Related: #10421", r.Schema, r.Integrity.SHA256, shellDisplay(r.Command), r.ExitStatus, r.OutputSHA256,
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
		if isSecretFlag(lower) {
			out[i] = arg
			secretNext = true
			continue
		}
		if key, _, ok := strings.Cut(arg, "="); ok && isSecretFlag(strings.ToLower(key)) {
			out[i] = key + "=[REDACTED]"
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
	unixHomePath     = regexp.MustCompile(`/(?:Users|home)/[^/\s]+`)
	uncUserPath      = regexp.MustCompile(`(?i)\\\\[^\\\s]+\\(?:Users|home)\\[^\\\s]+`)
	volatileID       = regexp.MustCompile(`(?i)\b(?:hostname|computername|user(?:name)?|machine[-_ ]?id)\s*[:=]\s*[^\s,;]+`)
)

func scrub(s string) string {
	s = bearer.ReplaceAllString(s, "Bearer-[REDACTED]")
	s = secretAssignment.ReplaceAllString(s, "$1=[REDACTED]")
	s = windowsUserPath.ReplaceAllString(s, `[HOME]`)
	s = unixHomePath.ReplaceAllString(s, `[HOME]`)
	s = uncUserPath.ReplaceAllString(s, `[HOME]`)
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
func shortHash(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}
func shellDisplay(args []string) string { b, _ := json.Marshal(args); return string(b) }
