package main

import (
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func fixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
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
		{"apple", parseAppleAccelerators(fixture(t, "system-profiler.txt")), []Accelerator{{Vendor: "Apple", Kind: "gpu", Model: "Apple M3 Max", Backend: "Metal"}}},
		{"nvidia", parseNVIDIAAccelerators(fixture(t, "nvidia-smi.txt")), []Accelerator{
			{Vendor: "NVIDIA", Kind: "gpu", Model: "NVIDIA GeForce RTX 4090", Backend: "CUDA"},
			{Vendor: "NVIDIA", Kind: "gpu", Model: "NVIDIA RTX A6000", Backend: "CUDA"},
		}},
		{"amd", parseAMDAccelerators(fixture(t, "rocminfo.txt")), []Accelerator{
			{Vendor: "AMD", Kind: "gpu", Model: "AMD Radeon PRO W6800", Backend: "ROCm"},
			{Vendor: "AMD", Kind: "gpu", Model: "AMD Radeon RX 7900 XTX", Backend: "ROCm"},
		}},
		{"intel", parseIntelAccelerators(fixture(t, "sycl-ls.txt")), []Accelerator{
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
	if got := normalizeHardwareText(" Intel(R)  Core(TM) i9-13900K CPU @ 3.00GHz\r\n"); got != "Intel(R) Core(TM) i9-13900K CPU" {
		t.Fatalf("normalized CPU = %q", got)
	}
}

func TestPrivacyScrubbingFixture(t *testing.T) {
	private := strings.ReplaceAll(fixture(t, "private-output.txt"), "<WINDOWS_HOME>", "C:"+`\Users\sample-user`)
	got := scrub(private)
	forbidden := []string{"sample-user", "super-secret", "abc.def.ghi", `C:\\Users`, "/home/"}
	for _, value := range forbidden {
		if strings.Contains(got, value) {
			t.Fatalf("scrubbed output contains %q: %s", value, got)
		}
	}
	for _, marker := range []string{"[HOME]", "[REDACTED]"} {
		if !strings.Contains(got, marker) {
			t.Fatalf("scrubbed output missing %q: %s", marker, got)
		}
	}
	args := scrubArgs([]string{"fak", "bench", "--api-key=abc", "--token", "xyz", "C:" + `\Users\sample-user\model`})
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "abc") || strings.Contains(joined, "xyz") || strings.Contains(joined, "sample-user") {
		t.Fatalf("scrubbed args leaked: %s", joined)
	}
}

func sampleReceipt(t *testing.T) Receipt {
	t.Helper()
	r := Receipt{
		Schema: receiptSchema, StartedAt: "2026-08-31T12:00:00Z", FinishedAt: "2026-08-31T12:00:02Z", DurationMS: 2000,
		Command: []string{"fak", "bench", "native", "--engine", "metal"}, ExitStatus: 0,
		OutputSHA256: strings.Repeat("a", 64), OutputBytes: 42, Output: "tokens_per_second=12.5",
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

func TestReceiptVerificationAndTamperDetection(t *testing.T) {
	r := sampleReceipt(t)
	if err := verify(r); err != nil {
		t.Fatalf("verify: %v", err)
	}
	r.ExitStatus = 9
	if err := verify(r); err == nil || !strings.Contains(err.Error(), "integrity mismatch") {
		t.Fatalf("tamper error = %v", err)
	}

	good := sampleReceipt(t)
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := writeReceipt(path, good); err != nil {
		t.Fatal(err)
	}
	if _, err := readAndVerify(path); err != nil {
		t.Fatalf("read verify: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	b = []byte(strings.Replace(string(b), `"exit_status": 0`, `"exit_status": 1`, 1))
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readAndVerify(path); err == nil {
		t.Fatal("tampered file verified")
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
	if !strings.Contains(body1, "Related: #10421") || !strings.Contains(body1, r.Integrity.SHA256) {
		t.Fatalf("body missing evidence: %s", body1)
	}
	parsed, err := url.Parse(url1)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Host != "github.com" || parsed.Path != "/anthony-chaudhary/fak/issues/new" {
		t.Fatalf("unexpected URL: %s", url1)
	}
	if parsed.Query().Get("body") != body1 || parsed.Query().Get("labels") != "benchmark" {
		t.Fatalf("URL query mismatch: %s", url1)
	}
}
