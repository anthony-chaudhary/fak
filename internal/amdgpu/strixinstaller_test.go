package amdgpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestGenerateStrixInstallerPackage_Platforms(t *testing.T) {
	tests := []struct {
		platform     StrixHaloPlatform
		wantPlatform string
		wantTTMPages string
		wantGTTSize  string
	}{
		{
			platform:     StrixHalo128GB,
			wantPlatform: "strix-halo-128",
			wantTTMPages: "31457280",
			wantGTTSize:  "131072",
		},
		{
			platform:     StrixHalo64GB,
			wantPlatform: "strix-halo-64",
			wantTTMPages: "14680064",
			wantGTTSize:  "65536",
		},
	}

	for _, tc := range tests {
		t.Run(string(tc.platform), func(t *testing.T) {
			cfg := DefaultStrixInstallerConfig()
			cfg.Platform = tc.platform

			pkg, err := GenerateStrixInstallerPackage(cfg)
			if err != nil {
				t.Fatalf("unexpected error generating package: %v", err)
			}

			if pkg.Manifest.Platform != tc.wantPlatform {
				t.Errorf("manifest platform = %q, want %q", pkg.Manifest.Platform, tc.wantPlatform)
			}

			govService := string(pkg.Files["conf/fak-strix-governor.service"])
			if !strings.Contains(govService, tc.wantTTMPages) {
				t.Errorf("governor service missing TTM pages %q:\n%s", tc.wantTTMPages, govService)
			}

			grubScript := string(pkg.Files["scripts/setup-grub.sh"])
			if !strings.Contains(grubScript, "ttm.pages_limit="+tc.wantTTMPages) {
				t.Errorf("grub script missing ttm.pages_limit=%s:\n%s", tc.wantTTMPages, grubScript)
			}
			if !strings.Contains(grubScript, "amdgpu.gttsize="+tc.wantGTTSize) {
				t.Errorf("grub script missing amdgpu.gttsize=%s:\n%s", tc.wantGTTSize, grubScript)
			}
		})
	}

	// Verify unknown platform returns error
	cfgInvalid := DefaultStrixInstallerConfig()
	cfgInvalid.Platform = StrixHaloPlatform("invalid-platform-preset")
	if _, err := GenerateStrixInstallerPackage(cfgInvalid); err == nil {
		t.Fatal("expected error for invalid platform, got nil")
	}
}

func TestGenerateStrixInstallerPackage_ExpectedFiles(t *testing.T) {
	cfg := DefaultStrixInstallerConfig()
	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedFiles := []string{
		"manifest.json",
		"install.sh",
		"uninstall.sh",
		"verify.sh",
		"conf/strix-halo.env",
		"conf/fak-serve.service",
		"conf/fak-strix-governor.service",
		"conf/policy.json",
		"scripts/setup-grub.sh",
		"scripts/setup-firewall.sh",
		"clients/lan-agent.env",
		"clients/lan-agent.ps1",
		"clients/.mcp.json",
		"clients/README.md",
	}

	if len(pkg.Files) != len(expectedFiles) {
		t.Errorf("got %d files in package, want %d", len(pkg.Files), len(expectedFiles))
	}

	for _, file := range expectedFiles {
		content, ok := pkg.Files[file]
		if !ok {
			t.Errorf("expected file %q missing from package", file)
			continue
		}
		if len(content) == 0 {
			t.Errorf("file %q is empty", file)
		}
	}
}

func TestGenerateStrixInstallerPackage_ManifestChecksums(t *testing.T) {
	cfg := DefaultStrixInstallerConfig()
	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if pkg.Manifest.Schema != "fak.strix-installer-manifest/1" {
		t.Errorf("manifest schema = %q, want 'fak.strix-installer-manifest/1'", pkg.Manifest.Schema)
	}
	if pkg.Manifest.GatewayKey != pkg.Config.GatewayKey {
		t.Errorf("manifest gateway key mismatch: %q != %q", pkg.Manifest.GatewayKey, pkg.Config.GatewayKey)
	}

	for relPath, expectedChecksum := range pkg.Manifest.Files {
		content, ok := pkg.Files[relPath]
		if !ok {
			t.Errorf("file %q listed in manifest is not present in package.Files", relPath)
			continue
		}
		h := sha256.Sum256(content)
		actualChecksum := hex.EncodeToString(h[:])
		if actualChecksum != expectedChecksum {
			t.Errorf("checksum mismatch for %q: actual=%s, expected=%s", relPath, actualChecksum, expectedChecksum)
		}
	}

	// Verify manifest.json on disk unmarshals cleanly and matches pkg.Manifest
	var manifestFromJSON StrixPackageManifest
	if err := json.Unmarshal(pkg.Files["manifest.json"], &manifestFromJSON); err != nil {
		t.Fatalf("failed to parse manifest.json: %v", err)
	}
	if manifestFromJSON.Schema != pkg.Manifest.Schema {
		t.Errorf("unmarshaled schema = %q, want %q", manifestFromJSON.Schema, pkg.Manifest.Schema)
	}
	if manifestFromJSON.GatewayURL != pkg.Manifest.GatewayURL {
		t.Errorf("unmarshaled gateway URL = %q, want %q", manifestFromJSON.GatewayURL, pkg.Manifest.GatewayURL)
	}
}

func TestGenerateStrixInstallerPackage_LANCommunications(t *testing.T) {
	cfg := StrixInstallerConfig{
		Platform:             StrixHalo128GB,
		LANIP:                "192.168.1.150",
		Port:                 9090,
		ModelPort:            9131,
		ModelID:              "qwen3.6-27b",
		GatewayKey:           "testkey0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		EnableFleetSpine:     true,
		FleetSpineGroup:      "239.255.70.70",
		FleetSpinePort:       5765,
		FleetSpineAdvertiseS: 20.0,
		EnableCORS:           true,
		AllowedOrigins:       "http://localhost:3000,http://192.168.1.*",
		SpecDraftUBatch:      512,
		PrefillChunkTokens:   1024,
		KVBufferGiB:          4,
		FakBinaryPath:        "/opt/fak/bin/fak",
		OutputDir:            "test-output",
	}

	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 1. Check conf/strix-halo.env
	env := string(pkg.Files["conf/strix-halo.env"])
	wantEnvItems := []string{
		"FAK_GATEWAY_KEY=" + cfg.GatewayKey,
		"FLEET_SPINE_ENABLED=true",
		"FLEET_SPINE_GROUP=239.255.70.70",
		"FLEET_SPINE_PORT=5765",
		"FLEET_SPINE_ADVERTISE_S=20.0",
		"FAK_CORS_ALLOWED_ORIGINS=http://localhost:3000,http://192.168.1.*",
		"PREFILL_CHUNK_TOKENS=1024",
	}
	for _, item := range wantEnvItems {
		if !strings.Contains(env, item) {
			t.Errorf("conf/strix-halo.env missing %q:\n%s", item, env)
		}
	}

	// 2. Check conf/fak-serve.service
	serve := string(pkg.Files["conf/fak-serve.service"])
	wantServe := []string{
		"--addr 0.0.0.0:9090",
		"--base-url http://127.0.0.1:9131/v1",
		"--model qwen3.6-27b",
		"/opt/fak/bin/fak serve",
		"--require-key-env FAK_GATEWAY_KEY",
		"UnsetEnvironment=GGML_CUDA_ENABLE_UNIFIED_MEMORY HSA_OVERRIDE_GFX_VERSION",
	}
	for _, item := range wantServe {
		if !strings.Contains(serve, item) {
			t.Errorf("conf/fak-serve.service missing %q:\n%s", item, serve)
		}
	}

	// 3. Check scripts/setup-firewall.sh
	firewall := string(pkg.Files["scripts/setup-firewall.sh"])
	wantFirewall := []string{
		"GW_PORT=9090",
		"SPINE_PORT=5765",
	}
	for _, item := range wantFirewall {
		if !strings.Contains(firewall, item) {
			t.Errorf("scripts/setup-firewall.sh missing %q:\n%s", item, firewall)
		}
	}
	// Verify that MODEL_PORT is NOT exposed externally via ufw/firewall-cmd/iptables
	if strings.Contains(firewall, `"${MODEL_PORT}/tcp"`) || strings.Contains(firewall, `--dport "${MODEL_PORT}"`) {
		t.Errorf("scripts/setup-firewall.sh exposes model port to LAN (must remain loopback-only):\n%s", firewall)
	}

	// 4. Check clients/lan-agent.env
	clientEnv := string(pkg.Files["clients/lan-agent.env"])
	wantClientEnv := []string{
		`export FAK_BASE_URL="http://192.168.1.150:9090"`,
		`export FAK_GATEWAY_KEY="` + cfg.GatewayKey + `"`,
		`export OPENAI_BASE_URL="http://192.168.1.150:9090/v1"`,
		`export OPENAI_API_KEY="` + cfg.GatewayKey + `"`,
		`export FLEET_SPINE_GROUP="239.255.70.70"`,
		`export FLEET_SPINE_PORT="5765"`,
	}
	for _, item := range wantClientEnv {
		if !strings.Contains(clientEnv, item) {
			t.Errorf("clients/lan-agent.env missing %q:\n%s", item, clientEnv)
		}
	}

	// 5. Check clients/lan-agent.ps1
	clientPs1 := string(pkg.Files["clients/lan-agent.ps1"])
	wantClientPs1 := []string{
		`$env:FAK_BASE_URL = "http://192.168.1.150:9090"`,
		`$env:FAK_GATEWAY_KEY = "` + cfg.GatewayKey + `"`,
		`$env:OPENAI_BASE_URL = "http://192.168.1.150:9090/v1"`,
		`$env:OPENAI_API_KEY = "` + cfg.GatewayKey + `"`,
		`$env:FLEET_SPINE_GROUP = "239.255.70.70"`,
		`$env:FLEET_SPINE_PORT = "5765"`,
	}
	for _, item := range wantClientPs1 {
		if !strings.Contains(clientPs1, item) {
			t.Errorf("clients/lan-agent.ps1 missing %q:\n%s", item, clientPs1)
		}
	}

	// 6. Check clients/.mcp.json
	mcp := string(pkg.Files["clients/.mcp.json"])
	wantMCP := []string{
		`"url": "http://192.168.1.150:9090/mcp"`,
		`"Authorization": "Bearer ` + cfg.GatewayKey + `"`,
	}
	for _, item := range wantMCP {
		if !strings.Contains(mcp, item) {
			t.Errorf("clients/.mcp.json missing %q:\n%s", item, mcp)
		}
	}
}

func TestGenerateStrixInstallerPackage_GotchaSettings(t *testing.T) {
	cfg := DefaultStrixInstallerConfig()
	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 1. Verify amdgpu.lockup_timeout=-1 in setup-grub.sh and verify.sh
	grub := string(pkg.Files["scripts/setup-grub.sh"])
	if !strings.Contains(grub, "amdgpu.lockup_timeout=-1") {
		t.Errorf("setup-grub.sh missing amdgpu.lockup_timeout=-1:\n%s", grub)
	}
	verify := string(pkg.Files["verify.sh"])
	if !strings.Contains(verify, "amdgpu.lockup_timeout=-1") {
		t.Errorf("verify.sh missing amdgpu.lockup_timeout=-1:\n%s", verify)
	}

	// 2. Verify ttm.pages_limit in setup-grub.sh and verify.sh
	if !strings.Contains(grub, "ttm.pages_limit=31457280") {
		t.Errorf("setup-grub.sh missing ttm.pages_limit=31457280:\n%s", grub)
	}
	if !strings.Contains(verify, "ttm.pages_limit") {
		t.Errorf("verify.sh missing ttm.pages_limit check:\n%s", verify)
	}

	// 3. Verify Ollama & PyTorch gotcha env vars in conf/strix-halo.env
	env := string(pkg.Files["conf/strix-halo.env"])
	gotchaEnvs := []string{
		"OLLAMA_VULKAN=1",
		"OLLAMA_IGPU_ENABLE=1",
		"HIP_VISIBLE_DEVICES=-1",
		"TORCH_BLAS_PREFER_HIPBLASLT=1",
		"KV_CACHE_DIRTY_RING_BUFFER_GIB=4",
		"SPEC_DRAFT_UBATCH_SIZE=512",
		"FAK_PLANNER_TIMEOUT_S=1800",
		"FAK_HTTP_WRITE_TIMEOUT_S=1800",
	}
	for _, ge := range gotchaEnvs {
		if !strings.Contains(env, ge) {
			t.Errorf("conf/strix-halo.env missing gotcha setting %q:\n%s", ge, env)
		}
	}

	// 4. Verify DPM governor locking in fak-strix-governor.service and verify.sh
	gov := string(pkg.Files["conf/fak-strix-governor.service"])
	if !strings.Contains(gov, "echo high >") || !strings.Contains(gov, "power_dpm_force_performance_level") {
		t.Errorf("fak-strix-governor.service missing DPM high lock:\n%s", gov)
	}
	if !strings.Contains(verify, "power_dpm_force_performance_level") {
		t.Errorf("verify.sh missing DPM governor check:\n%s", verify)
	}

	// 5. Verify toxic gotchas #9 and #5 are checked in verify.sh and documented in strix-halo.env
	if !strings.Contains(verify, "GGML_CUDA_ENABLE_UNIFIED_MEMORY") || !strings.Contains(verify, "HSA_OVERRIDE_GFX_VERSION") {
		t.Errorf("verify.sh missing check for toxic Gotchas #9 and #5:\n%s", verify)
	}
	if !strings.Contains(env, "GGML_CUDA_ENABLE_UNIFIED_MEMORY") || !strings.Contains(env, "HSA_OVERRIDE_GFX_VERSION") {
		t.Errorf("conf/strix-halo.env missing documentation for Gotchas #9 and #5:\n%s", env)
	}
}

func TestStrixPackage_WriteToDir(t *testing.T) {
	cfg := DefaultStrixInstallerConfig()
	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tmpDir := t.TempDir()
	targetDir := filepath.Join(tmpDir, "test-pkg")

	if err := pkg.WriteToDir(targetDir); err != nil {
		t.Fatalf("WriteToDir failed: %v", err)
	}

	for relPath, expectedContent := range pkg.Files {
		fullPath := filepath.Join(targetDir, filepath.FromSlash(relPath))
		info, err := os.Stat(fullPath)
		if err != nil {
			t.Errorf("file %q not found on disk: %v", relPath, err)
			continue
		}

		diskContent, err := os.ReadFile(fullPath)
		if err != nil {
			t.Errorf("failed to read file %q on disk: %v", relPath, err)
			continue
		}

		if !bytes.Equal(diskContent, expectedContent) {
			t.Errorf("content mismatch for %q on disk", relPath)
		}

		// On non-Windows platforms, check exact permissions
		if runtime.GOOS != "windows" {
			perm := info.Mode().Perm()
			if strings.HasSuffix(relPath, ".sh") && perm&0111 == 0 {
				t.Errorf("script %q should be executable, got perm: %o", relPath, perm)
			}
			if (strings.HasSuffix(relPath, ".env") || strings.HasSuffix(relPath, ".ps1") || (strings.HasSuffix(relPath, ".json") && strings.Contains(relPath, "clients"))) && perm != 0600 {
				t.Errorf("secret file %q should have 0600 perm, got perm: %o", relPath, perm)
			}
		}
	}
}

func TestRunStrixInstallerCLI(t *testing.T) {
	t.Run("HelpFlag", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := RunStrixInstallerCLI(&out, &errOut, []string{"--help"})
		if code != 2 {
			t.Errorf("expected exit code 2 for --help, got %d", code)
		}
		if !strings.Contains(errOut.String(), "Usage of amd-strix-package") {
			t.Errorf("stderr missing usage: %s", errOut.String())
		}
	})

	t.Run("JSONFlag", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := RunStrixInstallerCLI(&out, &errOut, []string{"--json", "--platform", "strix-halo-64"})
		if code != 0 {
			t.Fatalf("expected exit code 0 for --json, got %d (err: %s)", code, errOut.String())
		}
		var manifest StrixPackageManifest
		if err := json.Unmarshal(out.Bytes(), &manifest); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v\noutput: %s", err, out.String())
		}
		if manifest.Platform != "strix-halo-64" {
			t.Errorf("manifest platform = %q, want 'strix-halo-64'", manifest.Platform)
		}
		if len(manifest.Files) == 0 {
			t.Error("manifest files map is empty")
		}
	})

	t.Run("DirAndApplyFlag", func(t *testing.T) {
		tmpDir := t.TempDir()
		outDir := filepath.Join(tmpDir, "strix-installer-output")

		var out, errOut bytes.Buffer
		code := RunStrixInstallerCLI(&out, &errOut, []string{
			"--dir", outDir,
			"--apply",
			"--platform", "strix-halo-128",
			"--port", "8888",
			"--model-port", "8889",
			"--model", "qwen3.6-27b",
			"--lan-ip", "10.0.0.42",
		})
		if code != 0 {
			t.Fatalf("expected exit code 0, got %d (err: %s)", code, errOut.String())
		}

		if !strings.Contains(out.String(), "successfully written") {
			t.Errorf("stdout missing confirmation message: %s", out.String())
		}

		manifestPath := filepath.Join(outDir, "manifest.json")
		if _, err := os.Stat(manifestPath); err != nil {
			t.Fatalf("manifest.json not written to disk: %v", err)
		}

		installPath := filepath.Join(outDir, "install.sh")
		if _, err := os.Stat(installPath); err != nil {
			t.Fatalf("install.sh not written to disk: %v", err)
		}
	})

	t.Run("DryRunFlag", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := RunStrixInstallerCLI(&out, &errOut, []string{"--dir", "my-pkg-dir"})
		if code != 0 {
			t.Fatalf("expected exit code 0, got %d (err: %s)", code, errOut.String())
		}
		if !strings.Contains(out.String(), "dry-run") {
			t.Errorf("stdout missing dry-run note: %s", out.String())
		}
		if !strings.Contains(out.String(), "--apply") {
			t.Errorf("stdout missing --apply instruction: %s", out.String())
		}
	})

	t.Run("InvalidPlatform", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := RunStrixInstallerCLI(&out, &errOut, []string{"--platform", "nonexistent"})
		if code != 2 {
			t.Errorf("expected exit code 2 for invalid platform, got %d", code)
		}
	})

	t.Run("UnexpectedArgs", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := RunStrixInstallerCLI(&out, &errOut, []string{"extra", "argument"})
		if code != 2 {
			t.Errorf("expected exit code 2 for unexpected args, got %d", code)
		}
	})
}
