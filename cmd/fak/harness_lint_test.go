package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnesslint"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit/lockv2"
)

func createTestLockFile(t *testing.T, content []byte) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "product.lock.json")
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func validV2LockJSON() []byte {
	lock := &lockv2.Lock{
		Schema: lockv2.ProductLockSchemaV2,
		ID:     "sha256:1111111111111111111111111111111111111111111111111111111111111111",
		Platforms: []lockv2.PlatformRequirement{
			{OS: "linux", Arch: "amd64", Contract: "v1"},
			{OS: "darwin", Arch: "arm64", Contract: "v1"},
			{OS: "windows", Arch: "amd64", Contract: "v1"},
		},
		Budget: lockv2.LockBudget{ContextTokens: 2000, MemoryMiB: 512, Workers: 2},
		Components: []lockv2.LockedComponent{
			{
				ID:            "kernel",
				Version:       "1.0.0",
				Digest:        "sha256:2222222222222222222222222222222222222222222222222222222222222222",
				Source:        "registry/kernel",
				Reason:        "root",
				Provider:      "system",
				Provides:      []string{"runtime"},
				Compatibility: lockv2.LockCompatibility{OS: []string{"linux", "darwin", "windows"}, Arch: []string{"amd64", "arm64"}, Contract: "v1"},
				Adapters:      []string{"instruction"},
			},
		},
		Assets: []lockv2.LockedAsset{
			{Kind: "instruction", ID: "guide", Value: "concise", Source: "defaults"},
			{Kind: "secret", ID: "api_key", Ref: "env:FAK_API_KEY", Source: "env"},
		},
	}
	raw, err := json.MarshalIndent(lock, "", "  ")
	if err != nil {
		panic(err)
	}
	return raw
}

func TestHarnessLint(t *testing.T) {
	t.Run("bad lock with plaintext secret fails with code 1 and outputs SECRET_PLAINTEXT_LEAK", func(t *testing.T) {
		lock := &lockv2.Lock{
			Schema: lockv2.ProductLockSchemaV2,
			Budget: lockv2.LockBudget{ContextTokens: 1000},
			Platforms: []lockv2.PlatformRequirement{
				{OS: "linux", Arch: "amd64"},
				{OS: "darwin", Arch: "arm64"},
			},
			Components: []lockv2.LockedComponent{
				{ID: "c1", Version: "1.0.0", Digest: "sha256:abc", Source: "reg"},
			},
			Assets: []lockv2.LockedAsset{
				{Kind: "secret", ID: "leaked_cred", Value: "plaintext-password-12345", Ref: "env:DB_PASS", Source: "env"},
			},
		}
		raw, err := json.MarshalIndent(lock, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		lockPath := createTestLockFile(t, raw)

		var stdout, stderr bytes.Buffer
		code := runHarnessLint(&stdout, &stderr, []string{"--lock", lockPath})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d; stdout: %s; stderr: %s", code, stdout.String(), stderr.String())
		}
		combined := stdout.String() + "\n" + stderr.String()
		if !strings.Contains(combined, "SECRET_PLAINTEXT_LEAK") {
			t.Fatalf("expected output to contain SECRET_PLAINTEXT_LEAK, got:\n%s", combined)
		}

		// Also verify via runHarness dispatch
		stdout.Reset()
		stderr.Reset()
		code = runHarness(&stdout, &stderr, []string{"lint", "--lock", lockPath})
		if code != 1 {
			t.Fatalf("runHarness dispatch: expected exit code 1, got %d", code)
		}
		if !strings.Contains(stdout.String()+"\n"+stderr.String(), "SECRET_PLAINTEXT_LEAK") {
			t.Fatalf("runHarness dispatch: expected SECRET_PLAINTEXT_LEAK in output")
		}
	})

	t.Run("bad lock with CRLF fails with code 1 and outputs HL002_CRLF_LINE_ENDINGS", func(t *testing.T) {
		raw := validV2LockJSON()
		rawCRLF := []byte(strings.ReplaceAll(string(raw), "\n", "\r\n"))
		lockPath := createTestLockFile(t, rawCRLF)

		var stdout, stderr bytes.Buffer
		code := runHarnessLint(&stdout, &stderr, []string{"--lock", lockPath})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d; stdout: %s; stderr: %s", code, stdout.String(), stderr.String())
		}
		combined := stdout.String() + "\n" + stderr.String()
		if !strings.Contains(combined, string(harnesslint.HL002_CRLF_LINE_ENDINGS)) {
			t.Fatalf("expected output to contain %s, got:\n%s", harnesslint.HL002_CRLF_LINE_ENDINGS, combined)
		}
	})

	t.Run("bad lock with unknown fields fails with code 1 and outputs HL005_UNKNOWN_FIELDS", func(t *testing.T) {
		raw := []byte(`{
			"schema": "fak.harness-product-lock/v2",
			"id": "sha256:test",
			"disallowed_unknown_key": "bad_field",
			"budget": {"context_tokens": 1000}
		}`)
		lockPath := createTestLockFile(t, raw)

		var stdout, stderr bytes.Buffer
		code := runHarnessLint(&stdout, &stderr, []string{"--lock", lockPath})
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d; stdout: %s; stderr: %s", code, stdout.String(), stderr.String())
		}
		combined := stdout.String() + "\n" + stderr.String()
		if !strings.Contains(combined, string(harnesslint.HL005_UNKNOWN_FIELDS)) {
			t.Fatalf("expected output to contain %s, got:\n%s", harnesslint.HL005_UNKNOWN_FIELDS, combined)
		}
	})

	t.Run("valid v2 lock passes with exit code 0 and <50ms execution", func(t *testing.T) {
		raw := validV2LockJSON()
		lockPath := createTestLockFile(t, raw)

		var stdout, stderr bytes.Buffer
		start := time.Now()
		code := runHarnessLint(&stdout, &stderr, []string{"--lock", lockPath})
		elapsed := time.Since(start)

		if code != 0 {
			t.Fatalf("expected exit code 0 for valid lock, got %d; stderr: %s", code, stderr.String())
		}
		if elapsed > 50*time.Millisecond {
			t.Fatalf("execution took %v, required <50ms", elapsed)
		}
		if !strings.Contains(stdout.String(), "OK") {
			t.Fatalf("expected stdout to report OK, got: %s", stdout.String())
		}
	})

	t.Run("json mode outputs serialized Report", func(t *testing.T) {
		raw := validV2LockJSON()
		lockPath := createTestLockFile(t, raw)

		var stdout, stderr bytes.Buffer
		code := runHarnessLint(&stdout, &stderr, []string{"--lock", lockPath, "--json"})
		if code != 0 {
			t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
		}

		var report harnesslint.Report
		if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
			t.Fatalf("failed to unmarshal JSON report: %v; raw: %s", err, stdout.String())
		}
		if !report.Valid {
			t.Fatalf("expected valid report in JSON, got %+v", report)
		}
		if report.LockPath != lockPath {
			t.Fatalf("expected lock path %q, got %q", lockPath, report.LockPath)
		}
	})

	t.Run("manifest flag is supported as alias for lock", func(t *testing.T) {
		raw := validV2LockJSON()
		manifestPath := createTestLockFile(t, raw)

		var stdout, stderr bytes.Buffer
		code := runHarnessLint(&stdout, &stderr, []string{"--manifest", manifestPath})
		if code != 0 {
			t.Fatalf("expected exit code 0 with --manifest, got %d; stderr: %s", code, stderr.String())
		}
	})

	t.Run("allow single platform suppresses warning", func(t *testing.T) {
		lock := &lockv2.Lock{
			Schema: lockv2.ProductLockSchemaV2,
			Platforms: []lockv2.PlatformRequirement{
				{OS: "linux", Arch: "amd64"},
			},
			Budget: lockv2.LockBudget{ContextTokens: 1000},
			Components: []lockv2.LockedComponent{
				{ID: "c1", Version: "1.0.0", Digest: "sha256:2222222222222222222222222222222222222222222222222222222222222222", Source: "reg"},
			},
		}
		raw, err := json.MarshalIndent(lock, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		lockPath := createTestLockFile(t, raw)

		// Without flag -> reports warning
		var stdout, stderr bytes.Buffer
		code := runHarnessLint(&stdout, &stderr, []string{"--lock", lockPath})
		if code != 0 {
			t.Fatalf("warnings should exit with 0, got %d", code)
		}
		if !strings.Contains(stdout.String(), "SINGLE_PLATFORM_WARNING") {
			t.Fatalf("expected SINGLE_PLATFORM_WARNING, got: %s", stdout.String())
		}

		// With flag -> warning suppressed
		stdout.Reset()
		stderr.Reset()
		code = runHarnessLint(&stdout, &stderr, []string{"--lock", lockPath, "--allow-single-platform"})
		if code != 0 {
			t.Fatalf("exit %d", code)
		}
		if strings.Contains(stdout.String(), "SINGLE_PLATFORM_WARNING") {
			t.Fatalf("did not expect SINGLE_PLATFORM_WARNING with --allow-single-platform: %s", stdout.String())
		}
	})

	t.Run("missing flag exits with 2", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runHarnessLint(&stdout, &stderr, []string{})
		if code != 2 {
			t.Fatalf("expected exit code 2 when no flags passed, got %d", code)
		}
	})
}
