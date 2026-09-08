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

func TestGenerateStrixInstallerPackage_DualTP2(t *testing.T) {
	cfg := DefaultStrixInstallerConfig()
	cfg.ClusterMode = ClusterModeDualTP2
	cfg.ClusterPeerIP = "169.254.1.2"

	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		t.Fatalf("unexpected error generating dual_tp2 package: %v", err)
	}

	// Dual TP2 mode should have 15 files (including scripts/setup-usb4-rdma.sh)
	if len(pkg.Files) != 15 {
		t.Errorf("got %d files in dual_tp2 package, want 15", len(pkg.Files))
	}

	rdmaScript, ok := pkg.Files["scripts/setup-usb4-rdma.sh"]
	if !ok {
		t.Fatalf("scripts/setup-usb4-rdma.sh missing from dual_tp2 package")
	}
	rdmaStr := string(rdmaScript)

	for _, token := range []string{
		"0x38c00",
		"8 us",
		"value 32",
		"169.254.1.2",
		"169.254.1.1",
		"roce_",
		"thunderbolt0",
		"peer daemon discovery",
	} {
		if !strings.Contains(rdmaStr, token) {
			t.Errorf("scripts/setup-usb4-rdma.sh missing required token %q:\n%s", token, rdmaStr)
		}
	}

	// Verify conf/fak-serve.service contains cluster flags
	serveService := string(pkg.Files["conf/fak-serve.service"])
	if !strings.Contains(serveService, "--tp 2 --cluster-peer 169.254.1.2") {
		t.Errorf("conf/fak-serve.service missing dual_tp2 flags:\n%s", serveService)
	}

	// Verify verify.sh contains USB4 RoCEv2 and peer checks in dual_tp2 mode
	verifySh := string(pkg.Files["verify.sh"])
	if !strings.Contains(verifySh, "USB4 RoCEv2") || !strings.Contains(verifySh, "169.254.1.2") {
		t.Errorf("verify.sh missing dual_tp2 USB4 RoCEv2 peer checks:\n%s", verifySh)
	}

	// Test peer IP toggle: when peer is 169.254.1.1, local is 169.254.1.2
	cfg2 := DefaultStrixInstallerConfig()
	cfg2.ClusterMode = ClusterModeDualTP2
	cfg2.ClusterPeerIP = "169.254.1.1"
	pkg2, err := GenerateStrixInstallerPackage(cfg2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rdmaStr2 := string(pkg2.Files["scripts/setup-usb4-rdma.sh"])
	if !strings.Contains(rdmaStr2, `LOCAL_IP="169.254.1.2"`) || !strings.Contains(rdmaStr2, `PEER_IP="169.254.1.1"`) {
		t.Errorf("setup-usb4-rdma.sh did not invert local/peer IPs:\n%s", rdmaStr2)
	}

	// Verify invalid cluster mode error
	cfgInvalid := DefaultStrixInstallerConfig()
	cfgInvalid.ClusterMode = "invalid_cluster_mode"
	if _, err := GenerateStrixInstallerPackage(cfgInvalid); err == nil {
		t.Errorf("expected error for invalid cluster mode, got nil")
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
	if strings.Contains(env, "FAK_GPU_LEASE=") {
		t.Errorf("proxy-only conf/strix-halo.env must not claim the GPU lease:\n%s", env)
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
	if strings.Contains(serve, "FAK_GPU_LEASE=") {
		t.Errorf("proxy-only conf/fak-serve.service must not claim the GPU lease:\n%s", serve)
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

	// 7. Check install.sh does NOT leak plaintext gateway key
	installSh := string(pkg.Files["install.sh"])
	if strings.Contains(installSh, cfg.GatewayKey) {
		t.Errorf("install.sh contains raw gateway key: %s", cfg.GatewayKey)
	}
	wantInstallRedaction := `Gateway Key:     [configured in /etc/fak/strix-halo.env]`
	if !strings.Contains(installSh, wantInstallRedaction) {
		t.Errorf("install.sh missing redacted gateway key note:\n%s", installSh)
	}

	// 8. Check clients/README.md does NOT leak plaintext gateway key and uses ${FAK_GATEWAY_KEY}
	readme := string(pkg.Files["clients/README.md"])
	if strings.Contains(readme, cfg.GatewayKey) {
		t.Errorf("clients/README.md contains raw gateway key: %s", cfg.GatewayKey)
	}
	if !strings.Contains(readme, "${FAK_GATEWAY_KEY}") {
		t.Errorf("clients/README.md missing ${FAK_GATEWAY_KEY} placeholder:\n%s", readme)
	}
	if !strings.Contains(readme, "clients/lan-agent.env") {
		t.Errorf("clients/README.md missing reference to clients/lan-agent.env:\n%s", readme)
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

	// 6. Verify Vulkan SPIR-V acceleration asset defaults and shader compilation
	if !strings.Contains(env, "FAK_VULKAN_SPIRV=/var/lib/fak/spirv") {
		t.Errorf("conf/strix-halo.env missing FAK_VULKAN_SPIRV=/var/lib/fak/spirv:\n%s", env)
	}
	installSh := string(pkg.Files["install.sh"])
	if !strings.Contains(installSh, "/var/lib/fak/spirv") {
		t.Errorf("install.sh missing /var/lib/fak/spirv directory creation:\n%s", installSh)
	}
	if !strings.Contains(installSh, "glslc -O --target-env=vulkan1.2 -fshader-stage=comp") {
		t.Errorf("install.sh missing glslc SPIR-V shader compilation step:\n%s", installSh)
	}
	if !strings.Contains(installSh, "1002:1586") || !strings.Contains(installSh, "gfx1151") || !strings.Contains(installSh, "Radeon 8060S") {
		t.Errorf("install.sh missing AMD Strix Halo GPU (Radeon 8060S / 1002:1586 / gfx1151) auto-detection:\n%s", installSh)
	}
	if !strings.Contains(installSh, "--engine inkernel --backend vulkan") {
		t.Errorf("install.sh missing --engine inkernel --backend vulkan injection:\n%s", installSh)
	}
	if !strings.Contains(verify, "/var/lib/fak/spirv") || !strings.Contains(verify, "Vulkan SPIR-V") {
		t.Errorf("verify.sh missing Vulkan SPIR-V verification:\n%s", verify)
	}
	if !strings.Contains(verify, "runtime-capabilities --backend vulkan") || !strings.Contains(verify, "status: available") {
		t.Errorf("verify.sh missing fak runtime-capabilities --backend vulkan status: available check:\n%s", verify)
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
			if isSecretBearingFile(relPath) && perm != 0600 {
				t.Errorf("secret file %q should have 0600 perm, got perm: %o", relPath, perm)
			}
		}
	}

	if runtime.GOOS != "windows" {
		targetInfo, err := os.Stat(targetDir)
		if err != nil {
			t.Fatalf("failed to stat targetDir: %v", err)
		}
		if targetInfo.Mode().Perm() != 0700 {
			t.Errorf("targetDir perm = %o, want 0700", targetInfo.Mode().Perm())
		}

		for _, sub := range []string{"conf", "clients"} {
			subPath := filepath.Join(targetDir, sub)
			subInfo, err := os.Stat(subPath)
			if err != nil {
				t.Fatalf("failed to stat subDir %q: %v", sub, err)
			}
			if subInfo.Mode().Perm() != 0700 {
				t.Errorf("subDir %q perm = %o, want 0700", sub, subInfo.Mode().Perm())
			}
		}

		// Test preexisting permissive file and directory get restricted to 0600 / 0700
		preexistingDir := filepath.Join(tmpDir, "preexisting-pkg")
		if err := os.MkdirAll(filepath.Join(preexistingDir, "conf"), 0777); err != nil {
			t.Fatalf("failed to create preexisting dir: %v", err)
		}
		if err := os.Chmod(filepath.Join(preexistingDir, "conf"), 0777); err != nil {
			t.Fatalf("failed to chmod preexisting dir: %v", err)
		}
		preexistingFile := filepath.Join(preexistingDir, "manifest.json")
		if err := os.WriteFile(preexistingFile, []byte("{}"), 0666); err != nil {
			t.Fatalf("failed to write preexisting file: %v", err)
		}
		if err := os.Chmod(preexistingFile, 0666); err != nil {
			t.Fatalf("failed to chmod preexisting file: %v", err)
		}

		if err := pkg.WriteToDir(preexistingDir); err != nil {
			t.Fatalf("WriteToDir on preexisting dir failed: %v", err)
		}

		confInfo, err := os.Stat(filepath.Join(preexistingDir, "conf"))
		if err != nil {
			t.Fatalf("failed to stat conf: %v", err)
		}
		if confInfo.Mode().Perm() != 0700 {
			t.Errorf("preexisting conf dir perm = %o, want 0700", confInfo.Mode().Perm())
		}

		manifestInfo, err := os.Stat(preexistingFile)
		if err != nil {
			t.Fatalf("failed to stat manifest: %v", err)
		}
		if manifestInfo.Mode().Perm() != 0600 {
			t.Errorf("preexisting manifest file perm = %o, want 0600", manifestInfo.Mode().Perm())
		}
	}
}

func TestStrixInstaller_CredentialRedaction(t *testing.T) {
	cfg := DefaultStrixInstallerConfig()
	cfg.GatewayKey = "testkey0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// install.sh must not contain plaintext GatewayKey
	installSh := string(pkg.Files["install.sh"])
	if strings.Contains(installSh, cfg.GatewayKey) {
		t.Errorf("install.sh leaks plaintext GatewayKey: %s", cfg.GatewayKey)
	}
	wantInstallNote := `Gateway Key:     [configured in /etc/fak/strix-halo.env]`
	if !strings.Contains(installSh, wantInstallNote) {
		t.Errorf("install.sh missing expected note %q:\n%s", wantInstallNote, installSh)
	}

	// clients/README.md must not contain plaintext GatewayKey
	readme := string(pkg.Files["clients/README.md"])
	if strings.Contains(readme, cfg.GatewayKey) {
		t.Errorf("clients/README.md leaks plaintext GatewayKey: %s", cfg.GatewayKey)
	}
	if !strings.Contains(readme, "${FAK_GATEWAY_KEY}") {
		t.Errorf("clients/README.md missing ${FAK_GATEWAY_KEY} placeholder:\n%s", readme)
	}
	if !strings.Contains(readme, "clients/lan-agent.env") {
		t.Errorf("clients/README.md missing reference to clients/lan-agent.env:\n%s", readme)
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

	t.Run("DualTP2Flag", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := RunStrixInstallerCLI(&out, &errOut, []string{
			"--tp2",
			"--cluster-peer", "10.0.0.99",
			"--json",
		})
		if code != 0 {
			t.Fatalf("expected exit code 0 for --tp2 --json, got %d (err: %s)", code, errOut.String())
		}
		var manifest StrixPackageManifest
		if err := json.Unmarshal(out.Bytes(), &manifest); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v\noutput: %s", err, out.String())
		}
		if _, ok := manifest.Files["scripts/setup-usb4-rdma.sh"]; !ok {
			t.Errorf("manifest missing scripts/setup-usb4-rdma.sh in dual_tp2 mode")
		}
	})

	t.Run("InvalidClusterMode", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := RunStrixInstallerCLI(&out, &errOut, []string{"--cluster-mode", "invalid_cluster"})
		if code == 0 {
			t.Errorf("expected non-zero exit code for invalid cluster mode, got 0")
		}
	})
}

func TestStrixInstaller_Issue12003_SPIRVAndVulkanDiscovery(t *testing.T) {
	cfg := DefaultStrixInstallerConfig()
	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		t.Fatalf("unexpected error generating package: %v", err)
	}

	// 1. SPIR-V compilation step with glslc and bundled SPIR-V fallback in install.sh
	installSh := string(pkg.Files["install.sh"])
	if !strings.Contains(installSh, "glslc -O --target-env=vulkan1.2 -fshader-stage=comp") {
		t.Errorf("install.sh missing glslc SPIR-V compilation step:\n%s", installSh)
	}
	if !strings.Contains(installSh, `cp "${SCRIPT_DIR}/spirv"/*.spv /var/lib/fak/spirv/`) {
		t.Errorf("install.sh missing bundled SPIR-V copy fallback:\n%s", installSh)
	}

	// 2. Auto-detection of Radeon 8060S / gfx1151 / 1002:1586 setting --engine inkernel --backend vulkan
	for _, expectedToken := range []string{"1002:1586", "gfx1151", "Radeon 8060S", "--engine inkernel --backend vulkan"} {
		if !strings.Contains(installSh, expectedToken) {
			t.Errorf("install.sh missing expected token %q for Strix Halo GPU auto-detection:\n%s", expectedToken, installSh)
		}
	}

	// 3. Vulkan SPIR-V directory and runtime-capabilities checks in verify.sh
	verifySh := string(pkg.Files["verify.sh"])
	if !strings.Contains(verifySh, "/var/lib/fak/spirv") {
		t.Errorf("verify.sh missing /var/lib/fak/spirv check:\n%s", verifySh)
	}
	if !strings.Contains(verifySh, "runtime-capabilities --backend vulkan") || !strings.Contains(verifySh, "status: available") {
		t.Errorf("verify.sh missing fak runtime-capabilities --backend vulkan check:\n%s", verifySh)
	}
}

func TestStrixInstaller_Issue12004_DualNodeTP2(t *testing.T) {
	// 1. ClusterMode standalone default: 14 files, no scripts/setup-usb4-rdma.sh
	defaultCfg := DefaultStrixInstallerConfig()
	if defaultCfg.ClusterMode != ClusterModeStandalone {
		t.Errorf("default ClusterMode = %q, want %q", defaultCfg.ClusterMode, ClusterModeStandalone)
	}
	standalonePkg, err := GenerateStrixInstallerPackage(defaultCfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := standalonePkg.Files["scripts/setup-usb4-rdma.sh"]; ok {
		t.Errorf("standalone package should not generate scripts/setup-usb4-rdma.sh")
	}

	// 2. ClusterMode dual_tp2: 15 files, generates scripts/setup-usb4-rdma.sh
	tp2Cfg := DefaultStrixInstallerConfig()
	tp2Cfg.ClusterMode = ClusterModeDualTP2
	tp2Cfg.ClusterPeerIP = "169.254.1.2"
	tp2Pkg, err := GenerateStrixInstallerPackage(tp2Cfg)
	if err != nil {
		t.Fatalf("unexpected error generating dual_tp2 package: %v", err)
	}

	rdmaScript, ok := tp2Pkg.Files["scripts/setup-usb4-rdma.sh"]
	if !ok {
		t.Fatalf("scripts/setup-usb4-rdma.sh missing from dual_tp2 package")
	}
	rdmaStr := string(rdmaScript)

	// Check 8 us interrupt moderation (0x38c00, value 32)
	for _, token := range []string{"0x38c00", "8 us", "value 32", "nhi_interrupt_moderation"} {
		if !strings.Contains(rdmaStr, token) {
			t.Errorf("scripts/setup-usb4-rdma.sh missing interrupt moderation token %q:\n%s", token, rdmaStr)
		}
	}

	// Check static link-local IPv4 routing
	for _, token := range []string{`LOCAL_IP="169.254.1.1"`, `PEER_IP="169.254.1.2"`, `"${LOCAL_IP}/30"`, `"${PEER_IP}/32"`} {
		if !strings.Contains(rdmaStr, token) {
			t.Errorf("scripts/setup-usb4-rdma.sh missing link-local routing token %q:\n%s", token, rdmaStr)
		}
	}

	// Check peer daemon discovery
	for _, token := range []string{"peer daemon discovery", "169.254.1.2", "/healthz"} {
		if !strings.Contains(strings.ToLower(rdmaStr), token) {
			t.Errorf("scripts/setup-usb4-rdma.sh missing peer discovery token %q:\n%s", token, rdmaStr)
		}
	}

	// 3. conf/fak-serve.service --tp 2 and --cluster-peer flags
	serveService := string(tp2Pkg.Files["conf/fak-serve.service"])
	if !strings.Contains(serveService, "--tp 2 --cluster-peer 169.254.1.2") {
		t.Errorf("conf/fak-serve.service missing dual_tp2 flags:\n%s", serveService)
	}

	// 4. verify.sh checks for USB4 RoCEv2 and peer daemon in dual_tp2 mode
	verifySh := string(tp2Pkg.Files["verify.sh"])
	for _, token := range []string{"usb4 rocev2", "169.254.1.2", "peer daemon discovery"} {
		if !strings.Contains(strings.ToLower(verifySh), token) {
			t.Errorf("verify.sh missing dual_tp2 verification token %q:\n%s", token, verifySh)
		}
	}

	// 5. Unsupported cluster mode returns error
	invalidCfg := DefaultStrixInstallerConfig()
	invalidCfg.ClusterMode = "quad_tp4"
	if _, err := GenerateStrixInstallerPackage(invalidCfg); err == nil {
		t.Errorf("expected error for unsupported cluster mode, got nil")
	}
}
