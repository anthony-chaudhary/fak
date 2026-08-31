package localbench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchcatalog"
)

func exampleFixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "examples", "local-hardware-benchmark", "testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestHardwareNormalizationFixtures(t *testing.T) {
	tests := []struct {
		name string
		got  []Accelerator
		want []Accelerator
	}{
		{"apple", parseAppleAccelerators(exampleFixture(t, "system-profiler.txt")), []Accelerator{{Vendor: "Apple", Kind: "gpu", Model: "Apple M3 Max", Backend: "Metal"}}},
		{"nvidia", parseNVIDIAAccelerators(exampleFixture(t, "nvidia-smi.txt")), []Accelerator{
			{Vendor: "NVIDIA", Kind: "gpu", Model: "NVIDIA GeForce RTX 4090", Backend: "CUDA"},
			{Vendor: "NVIDIA", Kind: "gpu", Model: "NVIDIA RTX A6000", Backend: "CUDA"},
		}},
		{"amd", parseAMDAccelerators(exampleFixture(t, "rocminfo.txt")), []Accelerator{
			{Vendor: "AMD", Kind: "gpu", Model: "AMD Radeon PRO W6800", Backend: "ROCm"},
			{Vendor: "AMD", Kind: "gpu", Model: "AMD Radeon RX 7900 XTX", Backend: "ROCm"},
		}},
		{"intel", parseIntelAccelerators(exampleFixture(t, "sycl-ls.txt")), []Accelerator{
			{Vendor: "Intel", Kind: "accelerator", Model: "[level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1550 1.6", Backend: "oneAPI/SYCL"},
			{Vendor: "Intel", Kind: "accelerator", Model: "[opencl:gpu:0] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [31.0.101.5590]", Backend: "oneAPI/SYCL"},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !reflect.DeepEqual(tt.got, tt.want) {
				t.Fatalf("got %#v\nwant %#v", tt.got, tt.want)
			}
		})
	}
}

func TestPrivacyScrubbingFixture(t *testing.T) {
	private := strings.ReplaceAll(exampleFixture(t, "private-output.txt"), "<WINDOWS_HOME>", "C:"+`\Users\sample-user`)
	got := scrub(private)
	for _, secret := range []string{"sample-user", "super-secret", "abc.def.ghi", "workstation-77", "deadbeef"} {
		if strings.Contains(strings.ToLower(got), secret) {
			t.Fatalf("scrubbed output leaked %q: %s", secret, got)
		}
	}
	for _, marker := range []string{"[HOME]", "[REDACTED]"} {
		if !strings.Contains(got, marker) {
			t.Fatalf("scrubbed output missing %s: %s", marker, got)
		}
	}
	windowsProbe := `WARNING: loader skipped C:\WINDOWS\System32\DriverStore\FileRepository\nv_dispi.inf_amd64\nv-vk64.json` + "\n" + `Vulkan Instance Version: 1.3.280`
	probeOutput := normalizeVersion(windowsProbe)
	if strings.Contains(strings.ToLower(probeOutput), `c:\windows\`) || strings.Contains(strings.ToLower(probeOutput), "driverstore") {
		t.Fatalf("normalized probe leaked Windows absolute path: %q", probeOutput)
	}
	if !strings.Contains(probeOutput, "Vulkan Instance Version: 1.3.280") || !strings.Contains(probeOutput, "[LOCAL_PATH]") {
		t.Fatalf("normalized probe lost useful version or path marker: %q", probeOutput)
	}
	receipt := sampleReceipt(t)
	receipt.Hardware.Toolchains["vulkan"] = probeOutput
	if err := seal(&receipt); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.ToLower(string(encoded)), `c:\\windows\\`) || strings.Contains(strings.ToLower(string(encoded)), "driverstore") {
		t.Fatalf("sealed receipt leaked Windows absolute path: %s", encoded)
	}

	args := scrubArgs([]string{"runner", "--token", "secret-token", "--api-key=also-secret", "C:" + `\Users\sample-user\weights`, "safe"})
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "secret-token") || strings.Contains(joined, "also-secret") || strings.Contains(joined, "sample-user") {
		t.Fatalf("scrubbed args leaked: %q", joined)
	}
}

func sampleReceipt(t *testing.T) Receipt {
	t.Helper()
	r := Receipt{
		Schema: receiptSchema, Benchmark: "modelbench", Engine: "fak-native",
		StartedAt: "2026-08-20T10:00:00Z", FinishedAt: "2026-08-20T10:00:01Z", DurationMS: 1000,
		Command: []string{"fak", "modelbench", "--model", "qwen"}, ExitStatus: 0,
		OutputSHA256: strings.Repeat("a", 64), OutputBytes: 42, Output: "tokens_per_second=77",
		Hardware: Hardware{OS: "darwin", Arch: "arm64", CPU: "Apple M3 Max", MemoryBytes: 68719476736,
			Accelerators: []Accelerator{{Vendor: "Apple", Kind: "gpu", Model: "Apple M3 Max", Backend: "Metal"}},
			Toolchains:   map[string]string{"metal": "Apple metal version 32023.98"}},
		Provenance: Provenance{FakVersion: "0.45.0", FakRevision: "923309221e5b", RepoRevision: "60b8081aea20", GoVersion: "go1.26.7"},
	}
	if err := seal(&r); err != nil {
		t.Fatal(err)
	}
	return r
}

func TestReceiptVerificationFailClosed(t *testing.T) {
	r := sampleReceipt(t)
	if err := verify(r); err != nil {
		t.Fatalf("verify: %v", err)
	}

	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := writeReceipt(path, r); err != nil {
		t.Fatal(err)
	}
	if _, err := readAndVerify(path); err != nil {
		t.Fatalf("read verify: %v", err)
	}

	tests := map[string]string{
		"tamper":          strings.Replace(readFile(t, path), `"exit_status": 0`, `"exit_status": 1`, 1),
		"unknown field":   strings.Replace(readFile(t, path), `"schema":`, `"surprise": true,\n  "schema":`, 1),
		"unknown version": strings.Replace(readFile(t, path), receiptSchema, "fak.local-hardware-benchmark.receipt/v2", 1),
		"trailing value":  readFile(t, path) + "{}\n",
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			bad := filepath.Join(t.TempDir(), "bad.json")
			if err := os.WriteFile(bad, []byte(body), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := readAndVerify(bad); err == nil {
				t.Fatal("invalid receipt verified")
			}
		})
	}
}

// TestLegacyV1ReceiptCompatibility proves receipts emitted by the #10421 example,
// before benchmark and engine metadata were promoted, retain their exact seal.
func TestLegacyV1ReceiptCompatibility(t *testing.T) {
	type legacyReceipt struct {
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
	legacy := legacyReceipt{Schema: receiptSchema, StartedAt: "2026-08-20T10:00:00Z", FinishedAt: "2026-08-20T10:00:01Z", DurationMS: 1000, Command: []string{"go", "version"}, ExitStatus: 0, OutputSHA256: strings.Repeat("b", 64), OutputBytes: 12, Hardware: Hardware{OS: "linux", Arch: "amd64", CPU: "fixture"}, Provenance: Provenance{FakVersion: "unavailable", GoVersion: "go1.26.7"}}
	payload, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(payload)
	legacy.Integrity = Integrity{Algorithm: "sha256", SHA256: hex.EncodeToString(sum[:])}
	data, err := json.MarshalIndent(legacy, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "legacy-v1.json")
	if err := os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := readAndVerify(path)
	if err != nil {
		t.Fatalf("legacy v1 receipt rejected: %v", err)
	}
	if got.Benchmark != "" || got.Engine != "" {
		t.Fatalf("legacy optional metadata changed: %#v", got)
	}
}

func TestInventoryUsesCatalog(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if err := RunCLI([]string{"inventory"}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	var inventory Inventory
	if err := json.Unmarshal(stdout.Bytes(), &inventory); err != nil {
		t.Fatal(err)
	}
	if len(inventory.Benchmarks) != len(benchcatalog.All()) || len(inventory.Benchmarks) == 0 {
		t.Fatalf("catalog count = %d, want %d", len(inventory.Benchmarks), len(benchcatalog.All()))
	}
	for i, entry := range inventory.Benchmarks {
		if entry.Name != benchcatalog.All()[i].Name {
			t.Fatalf("catalog[%d] = %q, want %q", i, entry.Name, benchcatalog.All()[i].Name)
		}
	}
}

func TestRunRequiresExplicitSelectionAndPreservesLabels(t *testing.T) {
	for name, args := range map[string][]string{
		"benchmark": {"run", "--engine", "fak-native", "--", os.Args[0]},
		"engine":    {"run", "--benchmark", "modelbench", "--", os.Args[0]},
		"command":   {"run", "--benchmark", "modelbench", "--engine", "fak-native"},
		"unknown":   {"run", "--benchmark", "not-in-catalog", "--engine", "fak-native", "--", os.Args[0]},
	} {
		t.Run(name, func(t *testing.T) {
			if err := RunCLI(args, &bytes.Buffer{}, &bytes.Buffer{}); err == nil {
				t.Fatal("missing or invalid explicit selection accepted")
			}
		})
	}

	out := filepath.Join(t.TempDir(), "receipt.json")
	args := []string{"run", "--benchmark", "modelbench", "--engine", "operator-engine", "--out", out, "--", os.Args[0], "-test.run=TestLocalbenchChildHelper", "--", "--token", "secret-value"}
	var stdout, stderr bytes.Buffer
	if err := RunCLI(args, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	r, err := readAndVerify(out)
	if err != nil {
		t.Fatal(err)
	}
	if r.Benchmark != "modelbench" || r.Engine != "operator-engine" || filepath.Base(r.Command[0]) != filepath.Base(os.Args[0]) {
		t.Fatalf("explicit selection not preserved: %#v", r)
	}
	if strings.Contains(strings.Join(r.Command, " "), "secret-value") || strings.Contains(r.Output, "secret-value") {
		t.Fatalf("receipt leaked child secret: %#v", r)
	}
}

func TestChildFailureWritesVerifiableReceipt(t *testing.T) {
	out := filepath.Join(t.TempDir(), "failure.json")
	args := []string{"run", "--benchmark", "modelbench", "--engine", "operator-engine", "--out", out, "--", os.Args[0], "-test.run=TestLocalbenchChildHelper"}
	t.Setenv("FAK_LOCALBENCH_CHILD_FAIL", "1")
	err := RunCLI(args, &bytes.Buffer{}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "status 7") {
		t.Fatalf("child failure error = %v", err)
	}
	r, verifyErr := readAndVerify(out)
	if verifyErr != nil {
		t.Fatalf("failure receipt did not verify: %v", verifyErr)
	}
	if r.ExitStatus != 7 {
		t.Fatalf("exit status = %d, want 7", r.ExitStatus)
	}
}

func TestLocalbenchChildHelper(t *testing.T) {
	if os.Getenv("FAK_LOCALBENCH_CHILD_FAIL") == "1" {
		os.Stderr.WriteString("TOKEN=secret-value\n")
		os.Exit(7)
	}
	if strings.Contains(strings.Join(os.Args, " "), "TestLocalbenchChildHelper") {
		os.Stdout.WriteString("TOKEN=secret-value\nresult=ok\n")
	}
}

func TestDeterministicSubmissionRendering(t *testing.T) {
	r := sampleReceipt(t)
	body1, url1, err := renderSubmission(r)
	if err != nil {
		t.Fatal(err)
	}
	body2, url2, err := renderSubmission(r)
	if err != nil {
		t.Fatal(err)
	}
	if body1 != body2 || url1 != url2 {
		t.Fatal("submission rendering is not deterministic")
	}
	for _, want := range []string{"Related: #10421, #10444", r.Integrity.SHA256, "modelbench", "fak-native"} {
		if !strings.Contains(body1, want) {
			t.Fatalf("body missing %q: %s", want, body1)
		}
	}
	parsed, err := url.Parse(url1)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Host != "github.com" || parsed.Path != "/anthony-chaudhary/fak/issues/new" || parsed.Query().Get("body") != body1 || parsed.Query().Get("labels") != "benchmark" {
		t.Fatalf("unexpected submission URL: %s", url1)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestCommandIsPortableForCurrentPlatform(t *testing.T) {
	if runtime.GOOS == "windows" && filepath.Ext(os.Args[0]) != ".exe" {
		t.Fatalf("unexpected Windows test executable %q", os.Args[0])
	}
}
