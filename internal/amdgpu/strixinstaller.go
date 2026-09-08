// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// Strix Halo APU operational serving profiles, direct AQL/PM4 packet dispatch,
// native HSACO code-object emission, and Strix Halo installer package generation.
package amdgpu

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Strix Halo cluster modes for single-node standalone vs dual-node TP=2 over USB4 RoCEv2.
const (
	ClusterModeStandalone = "standalone"
	ClusterModeDualTP2    = "dual_tp2"
)

// StrixInstallerConfig configures the AMD Strix Halo installer package generation,
// including LAN communication coordinates, fleet spine discovery parameters, and
// Linux kernel gotcha mitigation settings.
type StrixInstallerConfig struct {
	Platform             StrixHaloPlatform `json:"platform"`
	LANIP                string            `json:"lan_ip"`
	Port                 int               `json:"port"`
	ModelPort            int               `json:"model_port"`
	ModelID              string            `json:"model_id"`
	GatewayKey           string            `json:"key"`
	EnableFleetSpine     bool              `json:"enable_fleet_spine"`
	FleetSpineGroup      string            `json:"fleet_spine_group"`
	FleetSpinePort       int               `json:"fleet_spine_port"`
	FleetSpineAdvertiseS float64           `json:"fleet_spine_advertise_s"`
	EnableCORS           bool              `json:"enable_cors"`
	AllowedOrigins       string            `json:"allowed_origins"`
	SpecDraftUBatch      int               `json:"spec_draft_ubatch"`
	PrefillChunkTokens   int               `json:"prefill_chunk_tokens"`
	KVBufferGiB          int               `json:"kv_buffer_gib"`
	FakBinaryPath        string            `json:"fak_binary_path"`
	OutputDir            string            `json:"output_dir"`
	ClusterMode          string            `json:"cluster_mode"`
	ClusterPeerIP        string            `json:"cluster_peer_ip,omitempty"`
}

// StrixPackageManifest records metadata, connectivity parameters, and cryptographic
// SHA-256 integrity checksums for all files generated in the installer package.
type StrixPackageManifest struct {
	Schema          string            `json:"schema"`
	Platform        string            `json:"platform"`
	GeneratedAt     string            `json:"generated_at"`
	GatewayURL      string            `json:"gateway_url"`
	GatewayKey      string            `json:"key"`
	FleetSpineGroup string            `json:"fleet_spine_group"`
	FleetSpinePort  int               `json:"fleet_spine_port"`
	Files           map[string]string `json:"files"`
}

// StrixPackage contains the in-memory representation of an AMD Strix Halo installer package,
// including configuration, manifest, and file payloads.
type StrixPackage struct {
	Config   StrixInstallerConfig
	Manifest StrixPackageManifest
	Files    map[string][]byte
}

// probeOutboundIP attempts to discover the host's primary outbound IPv4 address.
// Falls back to scanning network interfaces, and ultimately to "0.0.0.0" if no LAN IP is found.
func probeOutboundIP() string {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err == nil {
		defer conn.Close()
		if udpAddr, ok := conn.LocalAddr().(*net.UDPAddr); ok && udpAddr.IP != nil && !udpAddr.IP.IsUnspecified() {
			return udpAddr.IP.String()
		}
	}
	ifaces, err := net.Interfaces()
	if err == nil {
		for _, iface := range ifaces {
			if iface.Flags&net.FlagUp == 0 {
				continue
			}
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}
			for _, addr := range addrs {
				if ipNet, ok := addr.(*net.IPNet); ok && !ipNet.IP.IsLoopback() && ipNet.IP.To4() != nil {
					return ipNet.IP.String()
				}
			}
		}
	}
	return "0.0.0.0"
}

// DefaultStrixInstallerConfig returns a StrixInstallerConfig populated with production
// defaults for AMD Strix Halo (128 GiB preset, probed outbound LAN IP, and default ports).
func DefaultStrixInstallerConfig() StrixInstallerConfig {
	return StrixInstallerConfig{
		Platform:             StrixHalo128GB,
		LANIP:                probeOutboundIP(),
		Port:                 8080,
		ModelPort:            8131,
		ModelID:              "qwen3.6-27b",
		EnableFleetSpine:     true,
		FleetSpineGroup:      "239.255.70.65",
		FleetSpinePort:       4765,
		FleetSpineAdvertiseS: 15.0,
		EnableCORS:           true,
		AllowedOrigins:       "*",
		SpecDraftUBatch:      512,
		PrefillChunkTokens:   1024,
		KVBufferGiB:          4,
		FakBinaryPath:        "/usr/local/bin/fak",
		OutputDir:            "fak-strix-halo-pkg",
		ClusterMode:          ClusterModeStandalone,
		ClusterPeerIP:        "",
	}
}

// GenerateStrixInstallerPackage constructs all components of the Strix Halo installer package,
// including systemd service units, kernel cmdline scripts, firewall rules, client environments,
// and cryptographic package manifest.
func GenerateStrixInstallerPackage(cfg StrixInstallerConfig) (*StrixPackage, error) {
	if cfg.Platform == "" {
		cfg.Platform = StrixHalo128GB
	} else {
		parsed, err := ParseStrixHaloPlatform(string(cfg.Platform))
		if err != nil {
			return nil, err
		}
		cfg.Platform = parsed
	}

	if cfg.LANIP == "" {
		cfg.LANIP = probeOutboundIP()
	}
	if cfg.Port <= 0 {
		cfg.Port = 8080
	}
	if cfg.ModelPort <= 0 {
		cfg.ModelPort = 8131
	}
	if cfg.ModelID == "" {
		cfg.ModelID = "qwen3.6-27b"
	}
	if cfg.GatewayKey == "" {
		b := make([]byte, 32)
		if _, err := rand.Read(b); err != nil {
			return nil, fmt.Errorf("amdgpu: failed to generate gateway key: %w", err)
		}
		cfg.GatewayKey = hex.EncodeToString(b)
	}
	if cfg.FleetSpineGroup == "" {
		cfg.FleetSpineGroup = "239.255.70.65"
		cfg.EnableFleetSpine = true
	}
	if cfg.FleetSpinePort <= 0 {
		cfg.FleetSpinePort = 4765
	}
	if cfg.FleetSpineAdvertiseS <= 0 {
		cfg.FleetSpineAdvertiseS = 15.0
	}
	if cfg.AllowedOrigins == "" {
		cfg.AllowedOrigins = "*"
		cfg.EnableCORS = true
	}
	if cfg.SpecDraftUBatch <= 0 {
		cfg.SpecDraftUBatch = 512
	}
	if cfg.PrefillChunkTokens <= 0 {
		cfg.PrefillChunkTokens = 1024
	}
	if cfg.KVBufferGiB <= 0 {
		cfg.KVBufferGiB = 4
	}
	if cfg.FakBinaryPath == "" {
		cfg.FakBinaryPath = "/usr/local/bin/fak"
	}
	if cfg.OutputDir == "" {
		cfg.OutputDir = "fak-strix-halo-pkg"
	}
	if cfg.ClusterMode == "" {
		cfg.ClusterMode = ClusterModeStandalone
	} else if cfg.ClusterMode != ClusterModeStandalone && cfg.ClusterMode != ClusterModeDualTP2 {
		return nil, fmt.Errorf("amdgpu: unsupported cluster mode %q (must be %q or %q)", cfg.ClusterMode, ClusterModeStandalone, ClusterModeDualTP2)
	}
	if cfg.ClusterMode == ClusterModeDualTP2 && cfg.ClusterPeerIP == "" {
		cfg.ClusterPeerIP = "169.254.1.2"
	}

	var (
		ttmPages string
		gttSize  int
	)
	switch cfg.Platform {
	case StrixHalo64GB:
		ttmPages = "14680064"
		gttSize = 65536
	case StrixHalo128GB:
		ttmPages = "31457280"
		gttSize = 131072
	default:
		ttmPages = "31457280"
		gttSize = 131072
	}

	files := make(map[string][]byte)

	// 1. conf/strix-halo.env
	envContent := fmt.Sprintf(`# AMD Strix Halo (Ryzen AI MAX+ 395) Operational Environment
FAK_GATEWAY_KEY=%s
FLEET_SPINE_ENABLED=%t
FLEET_SPINE_GROUP=%s
FLEET_SPINE_PORT=%d
FLEET_SPINE_ADVERTISE_S=%.1f
FAK_CORS_ALLOWED_ORIGINS=%s
FAK_VULKAN_SPIRV=/var/lib/fak/spirv
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
OLLAMA_VULKAN=1
OLLAMA_IGPU_ENABLE=1
HIP_VISIBLE_DEVICES=-1
TORCH_BLAS_PREFER_HIPBLASLT=1
KV_CACHE_DIRTY_RING_BUFFER_GIB=%d
SPEC_DRAFT_UBATCH_SIZE=%d
PREFILL_CHUNK_TOKENS=%d
FAK_PLANNER_TIMEOUT_S=1800
FAK_HTTP_WRITE_TIMEOUT_S=1800
# Gotchas #9 & #5: ensure toxic overrides remain unset on APU:
# GGML_CUDA_ENABLE_UNIFIED_MEMORY must NOT be set (corrupts ROCm APU output)
# HSA_OVERRIDE_GFX_VERSION=11.0.0 must NOT be set (causes SIGSEGV in libamdhip64)
`,
		cfg.GatewayKey,
		cfg.EnableFleetSpine,
		cfg.FleetSpineGroup,
		cfg.FleetSpinePort,
		cfg.FleetSpineAdvertiseS,
		cfg.AllowedOrigins,
		cfg.KVBufferGiB,
		cfg.SpecDraftUBatch,
		cfg.PrefillChunkTokens,
	)
	files["conf/strix-halo.env"] = []byte(envContent)

	// 2. conf/fak-serve.service
	clusterFlags := ""
	if cfg.ClusterMode == ClusterModeDualTP2 {
		clusterFlags = fmt.Sprintf(" --tp 2 --cluster-peer %s", cfg.ClusterPeerIP)
	}
	serveService := fmt.Sprintf(`[Unit]
Description=fak Agent Kernel Gateway & Inference Service (Strix Halo)
After=network.target fak-strix-governor.service
Wants=fak-strix-governor.service

[Service]
Type=simple
EnvironmentFile=/etc/fak/strix-halo.env
UnsetEnvironment=GGML_CUDA_ENABLE_UNIFIED_MEMORY HSA_OVERRIDE_GFX_VERSION
ExecStart=%s serve%s --provider openai --base-url http://127.0.0.1:%d/v1 --model %s --addr 0.0.0.0:%d --policy /etc/fak/policy.json --require-key-env FAK_GATEWAY_KEY
Restart=always
RestartSec=5s
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
`,
		cfg.FakBinaryPath,
		clusterFlags,
		cfg.ModelPort,
		cfg.ModelID,
		cfg.Port,
	)
	files["conf/fak-serve.service"] = []byte(serveService)

	// 3. conf/fak-strix-governor.service
	governorService := fmt.Sprintf(`[Unit]
Description=AMD Strix Halo APU Performance Governor & TTM Aperture Setup
DefaultDependencies=no
After=sysinit.target local-fs.target
Before=fak-serve.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '\
  if [ -w /sys/module/ttm/parameters/pages_limit ]; then echo %s > /sys/module/ttm/parameters/pages_limit; fi; \
  for card in /sys/class/drm/card*/device/power_dpm_force_performance_level; do \
    if [ -w "$card" ]; then echo high > "$card"; fi; \
  done'

[Install]
WantedBy=multi-user.target
`,
		ttmPages,
	)
	files["conf/fak-strix-governor.service"] = []byte(governorService)

	// 4. conf/policy.json
	defaultPolicy := `{
  "version": "fak-policy/v1",
  "allow": [
    "search_web",
    "fetch_url",
    "run_query",
    "read_file",
    "list_dir"
  ],
  "allow_prefix": [
    "read_",
    "get_",
    "search_",
    "list_",
    "lookup_",
    "find_"
  ],
  "deny": {
    "delete_account": "POLICY_BLOCK",
    "rotate_credentials": "POLICY_BLOCK",
    "exfiltrate": "POLICY_BLOCK",
    "transfer_funds": "POLICY_BLOCK"
  },
  "self_modify_globs": [
    ".git/",
    ".dos/",
    "policy.json",
    "/etc/",
    "id_rsa"
  ],
  "redact_fields": [
    "password",
    "secret",
    "api_key",
    "token",
    "authorization"
  ]
}
`
	files["conf/policy.json"] = []byte(defaultPolicy)

	// 5. scripts/setup-grub.sh
	setupGrub := fmt.Sprintf(`#!/usr/bin/env bash
# Configure Linux GRUB kernel cmdline for AMD Strix Halo APU
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Error: setup-grub.sh must be run as root" >&2
  exit 1
fi

GRUB_FILE="/etc/default/grub"
BACKUP_FILE="/etc/default/grub.fak-backup"

if [[ ! -f "$GRUB_FILE" ]]; then
  echo "Warning: $GRUB_FILE not found, skipping GRUB configuration"
  exit 0
fi

if [[ ! -f "$BACKUP_FILE" ]]; then
  echo "Creating backup of $GRUB_FILE -> $BACKUP_FILE"
  cp "$GRUB_FILE" "$BACKUP_FILE"
fi

PARAMS="amdgpu.lockup_timeout=-1 ttm.pages_limit=%s amdgpu.gttsize=%d"

if grep -q "GRUB_CMDLINE_LINUX_DEFAULT=" "$GRUB_FILE"; then
  # Strip existing parameters to preserve idempotence
  sed -i -E 's/(amdgpu\.lockup_timeout=[^ "]+|ttm\.pages_limit=[^ "]+|amdgpu\.gttsize=[^ "]+)//g' "$GRUB_FILE"
  sed -i 's/  */ /g' "$GRUB_FILE"
  sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"${PARAMS} /" "$GRUB_FILE"
  echo "Updated $GRUB_FILE with Strix Halo parameters: ${PARAMS}"
fi

if command -v update-grub >/dev/null 2>&1; then
  echo "Running update-grub..."
  update-grub
elif command -v grub2-mkconfig >/dev/null 2>&1; then
  echo "Running grub2-mkconfig..."
  if [[ -f /boot/grub2/grub.cfg ]]; then
    grub2-mkconfig -o /boot/grub2/grub.cfg
  elif [[ -f /boot/grub/grub.cfg ]]; then
    grub2-mkconfig -o /boot/grub/grub.cfg
  else
    grub2-mkconfig -o /boot/grub2/grub.cfg
  fi
else
  echo "Warning: neither update-grub nor grub2-mkconfig found in PATH"
fi
`,
		ttmPages,
		gttSize,
	)
	files["scripts/setup-grub.sh"] = []byte(setupGrub)

	// 6. scripts/setup-firewall.sh
	setupFirewall := fmt.Sprintf(`#!/usr/bin/env bash
# Configure firewall rules for fak Strix Halo gateway and fleet spine
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Error: setup-firewall.sh must be run as root" >&2
  exit 1
fi

	GW_PORT=%d
	SPINE_PORT=%d

# Security note: MODEL_PORT (%d) is strictly loopback (127.0.0.1) so all LAN agent
# traffic must transit the authenticated, policy-gated fak serve gateway on GW_PORT.
echo "Configuring firewall for gateway (${GW_PORT}/tcp) and fleet spine (${SPINE_PORT}/udp)..."

if command -v ufw >/dev/null 2>&1 && ufw status | grep -qw "active"; then
  echo "Configuring ufw..."
  ufw allow "${GW_PORT}/tcp" comment "fak gateway" || true
  ufw allow "${SPINE_PORT}/udp" comment "fak fleet spine" || true
elif command -v firewall-cmd >/dev/null 2>&1 && systemctl is-active --quiet firewalld; then
  echo "Configuring firewalld..."
  firewall-cmd --permanent --add-port="${GW_PORT}/tcp" || true
  firewall-cmd --permanent --add-port="${SPINE_PORT}/udp" || true
  firewall-cmd --reload || true
elif command -v iptables >/dev/null 2>&1; then
  echo "Configuring iptables..."
  iptables -C INPUT -p tcp --dport "${GW_PORT}" -j ACCEPT 2>/dev/null || iptables -A INPUT -p tcp --dport "${GW_PORT}" -j ACCEPT
  iptables -C INPUT -p udp --dport "${SPINE_PORT}" -j ACCEPT 2>/dev/null || iptables -A INPUT -p udp --dport "${SPINE_PORT}" -j ACCEPT
else
  echo "No active ufw, firewalld, or iptables detected; skipping firewall rules."
fi
`,
		cfg.Port,
		cfg.FleetSpinePort,
		cfg.ModelPort,
	)
	files["scripts/setup-firewall.sh"] = []byte(setupFirewall)

	// 6.5 scripts/setup-usb4-rdma.sh (dual_tp2 mode)
	if cfg.ClusterMode == ClusterModeDualTP2 {
		localIP := "169.254.1.1"
		if cfg.ClusterPeerIP == "169.254.1.1" {
			localIP = "169.254.1.2"
		}
		setupUsb4Rdma := fmt.Sprintf(`#!/usr/bin/env bash
# USB4 RoCEv2 and NHI interrupt moderation setup for dual Strix Halo TP=2
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Error: setup-usb4-rdma.sh must be run as root" >&2
  exit 1
fi

PEER_IP="%s"
LOCAL_IP="%s"
IFACE="${USB4_IFACE:-thunderbolt0}"

echo "=== Configuring USB4 RoCEv2 for Strix Halo TP=2 (Peer: ${PEER_IP}) ==="

# 1. Configure USB4 NHI Interrupt Moderation to 8 us (register 0x38c00, value 32)
echo "--> Programming USB4 NHI interrupt moderation (8 us, register 0x38c00, value 32)..."
for pci in /sys/bus/pci/devices/*; do
  if [[ -f "$pci/class" ]]; then
    pci_class=$(cat "$pci/class")
    # USB4 / Thunderbolt NHI class code is 0x0c0340
    if [[ "$pci_class" == "0x0c0340"* ]]; then
      echo "    Setting 8 us interrupt moderation (value 32) on USB4 controller at $pci (register 0x38c00)"
      if [[ -f "$pci/nhi_interrupt_moderation" ]]; then
        echo "32" > "$pci/nhi_interrupt_moderation"
      fi
    fi
  done
fi

# 2. Configure Point-to-Point IPv4 Link-Local Interface
if ip link show "$IFACE" >/dev/null 2>&1; then
  echo "--> Configuring point-to-point IP on $IFACE (Local: ${LOCAL_IP}, Peer: ${PEER_IP})..."
  ip link set "$IFACE" up
  ip addr flush dev "$IFACE" || true
  ip addr add "${LOCAL_IP}/30" dev "$IFACE"
  ip route replace "${PEER_IP}/32" dev "$IFACE"
else
  echo "--> Interface $IFACE not yet present; udev/hotplug will assign ${LOCAL_IP}/30 when connected"
fi

# 3. Soft-RoCE (RXE) device setup if hardware RoCE not native
if command -v rdma >/dev/null 2>&1; then
  echo "--> Verifying RDMA link on $IFACE..."
  if ! rdma link show 2>/dev/null | grep -q "$IFACE"; then
    rdma link add "roce_${IFACE}" type rxe netdev "$IFACE" 2>/dev/null || true
  fi
fi

# 4. Peer daemon discovery
echo "--> Probing peer daemon discovery at ${PEER_IP}:%d..."
if ping -c 1 -W 2 "${PEER_IP}" >/dev/null 2>&1; then
  echo "    Peer node at ${PEER_IP} reachable via ICMP"
  if command -v curl >/dev/null 2>&1; then
    if curl -s -f -m 2 "http://${PEER_IP}:%d/healthz" >/dev/null 2>&1; then
      echo "    Peer daemon discovered at http://${PEER_IP}:%d/healthz [READY]"
    else
      echo "    Peer node online; peer daemon at http://${PEER_IP}:%d not yet listening"
    fi
  fi
else
  echo "    Peer node at ${PEER_IP} not yet reachable (standby until peer cable connected)"
fi

echo "--> USB4 RoCEv2 dual-node network configuration applied."
`,
			cfg.ClusterPeerIP,
			localIP,
			cfg.Port,
			cfg.Port,
			cfg.Port,
			cfg.Port,
		)
		files["scripts/setup-usb4-rdma.sh"] = []byte(setupUsb4Rdma)
	}

	// 7. install.sh
	installSh := fmt.Sprintf(`#!/usr/bin/env bash
# Idempotent installation script for fak AMD Strix Halo APU node
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Error: install.sh must be run as root (use sudo)" >&2
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Warning: /etc/os-release not found. This installer is designed for Linux systems." >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== Installing fak Strix Halo Node from ${SCRIPT_DIR} ==="

# 1. Setup GRUB for Strix Halo gotcha parameters (lockup timeout, ttm pages, gttsize)
echo "--> Configuring kernel parameters..."
bash "${SCRIPT_DIR}/scripts/setup-grub.sh"

# 2. Setup /etc/fak configuration and state directories
echo "--> Installing configuration in /etc/fak..."
mkdir -p /etc/fak /var/lib/fak/spirv /var/lib/fak/models
cp "${SCRIPT_DIR}/conf/strix-halo.env" /etc/fak/strix-halo.env
chmod 0600 /etc/fak/strix-halo.env

cp "${SCRIPT_DIR}/conf/policy.json" /etc/fak/policy.json
chmod 0644 /etc/fak/policy.json

# 2.5 Compile Vulkan SPIR-V compute shaders
echo "--> Compiling Vulkan SPIR-V compute shaders..."
mkdir -p /var/lib/fak/spirv
SHADER_SRC=""
for cand in "${SCRIPT_DIR}/shaders" "${SCRIPT_DIR}/internal/compute/shaders" "/var/lib/fak/repo/internal/compute/shaders" "$(pwd)/internal/compute/shaders" "${SCRIPT_DIR}/../internal/compute/shaders"; do
  if [[ -d "$cand" && -n "$(ls "$cand"/*.comp 2>/dev/null)" ]]; then
    SHADER_SRC="$cand"
    break
  fi
done

if [[ -n "$SHADER_SRC" ]] && command -v glslc >/dev/null 2>&1; then
  for shader in "$SHADER_SRC"/*.comp; do
    sname=$(basename "$shader" .comp)
    glslc -O --target-env=vulkan1.2 -fshader-stage=comp "$shader" -o "/var/lib/fak/spirv/${sname}.spv"
  done
  echo "    Compiled compute shaders into /var/lib/fak/spirv"
elif [[ -d "${SCRIPT_DIR}/spirv" ]]; then
  cp "${SCRIPT_DIR}/spirv"/*.spv /var/lib/fak/spirv/ 2>/dev/null || true
  echo "    Copied bundled SPIR-V shaders into /var/lib/fak/spirv"
else
  echo "    Note: glslc not found or shader sources absent; shaders can be placed in /var/lib/fak/spirv later"
fi

# 2.6 Auto-detect AMD Strix Halo GPU (Radeon 8060S / 1002:1586 / gfx1151)
HAS_STRIX=0
if [[ -d /sys/bus/pci/devices ]]; then
  for dev in /sys/bus/pci/devices/*; do
    if [[ -f "$dev/vendor" && -f "$dev/device" ]]; then
      v=$(cat "$dev/vendor" 2>/dev/null || true)
      d=$(cat "$dev/device" 2>/dev/null || true)
      if [[ "$v" == "0x1002" && "$d" == "0x1586" ]]; then
        HAS_STRIX=1
        break
      fi
    fi
  done
fi
if [[ $HAS_STRIX -eq 0 ]] && command -v lspci >/dev/null 2>&1; then
  if lspci -nn 2>/dev/null | grep -Ei "1002:1586|gfx1151|8060s|Radeon 8060S" >/dev/null 2>&1; then
    HAS_STRIX=1
  elif lspci 2>/dev/null | grep -Ei "Radeon 8060S|8060s|gfx1151" >/dev/null 2>&1; then
    HAS_STRIX=1
  fi
fi

# 2.7 USB4 RoCEv2 dual-node TP=2 network setup if present
if [[ -f "${SCRIPT_DIR}/scripts/setup-usb4-rdma.sh" ]]; then
  echo "--> Configuring USB4 RoCEv2 peer networking..."
  bash "${SCRIPT_DIR}/scripts/setup-usb4-rdma.sh"
fi

# 3. Setup firewall
echo "--> Configuring firewall rules..."
bash "${SCRIPT_DIR}/scripts/setup-firewall.sh"

# 4. Install systemd service units
echo "--> Installing systemd services..."
cp "${SCRIPT_DIR}/conf/fak-strix-governor.service" /etc/systemd/system/fak-strix-governor.service
cp "${SCRIPT_DIR}/conf/fak-serve.service" /etc/systemd/system/fak-serve.service
if [[ $HAS_STRIX -eq 1 ]]; then
  echo "--> Detected AMD Strix Halo GPU (Radeon 8060S / 1002:1586 / gfx1151); configuring --engine inkernel --backend vulkan"
  sed -i 's/serve /serve --engine inkernel --backend vulkan /' /etc/systemd/system/fak-serve.service
fi
chmod 0644 /etc/systemd/system/fak-strix-governor.service
chmod 0644 /etc/systemd/system/fak-serve.service

# 5. Enable and start services
if command -v systemctl >/dev/null 2>&1; then
  echo "--> Enabling and starting services..."
  systemctl daemon-reload
  systemctl enable --now fak-strix-governor.service
  systemctl enable --now fak-serve.service
fi

echo ""
echo "================================================================================"
echo "          fak AMD Strix Halo Node Installed Successfully!                       "
echo "================================================================================"
echo "Gateway URL:     http://%s:%d"
echo "Gateway Key:     [configured in /etc/fak/strix-halo.env]"
echo "Model Upstream:  http://127.0.0.1:%d/v1 (model: %s)"
echo "Fleet Spine:     %s:%d"
echo ""
echo "Client Connection Instructions:"
echo "  1. Linux / macOS agents:"
echo "     source ${SCRIPT_DIR}/clients/lan-agent.env"
echo "  2. Windows PowerShell agents:"
echo "     . ${SCRIPT_DIR}/clients/lan-agent.ps1"
echo "  3. Claude Code / MCP clients:"
echo "     Copy ${SCRIPT_DIR}/clients/.mcp.json to your project root"
echo ""
echo "Verify installation:"
echo "  bash ${SCRIPT_DIR}/verify.sh"
echo "================================================================================"
`,
		cfg.LANIP,
		cfg.Port,
		cfg.ModelPort,
		cfg.ModelID,
		cfg.FleetSpineGroup,
		cfg.FleetSpinePort,
	)
	files["install.sh"] = []byte(installSh)

	// 8. uninstall.sh
	uninstallSh := fmt.Sprintf(`#!/usr/bin/env bash
# Rollback / uninstall script for fak AMD Strix Halo APU node
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Error: uninstall.sh must be run as root (use sudo)" >&2
  exit 1
fi

echo "=== Uninstalling fak Strix Halo Node ==="

# 1. Stop and disable services
if command -v systemctl >/dev/null 2>&1; then
  echo "--> Stopping and disabling services..."
  systemctl stop fak-serve.service 2>/dev/null || true
  systemctl disable fak-serve.service 2>/dev/null || true
  systemctl disable fak-strix-governor.service 2>/dev/null || true
  rm -f /etc/systemd/system/fak-serve.service /etc/systemd/system/fak-strix-governor.service
  systemctl daemon-reload
fi

# 2. Revert GRUB configuration
if [[ -f /etc/default/grub.fak-backup ]]; then
  echo "--> Restoring /etc/default/grub from backup..."
  cp /etc/default/grub.fak-backup /etc/default/grub
  rm -f /etc/default/grub.fak-backup
  if command -v update-grub >/dev/null 2>&1; then
    update-grub
  elif command -v grub2-mkconfig >/dev/null 2>&1; then
    if [[ -f /boot/grub2/grub.cfg ]]; then
      grub2-mkconfig -o /boot/grub2/grub.cfg
    elif [[ -f /boot/grub/grub.cfg ]]; then
      grub2-mkconfig -o /boot/grub/grub.cfg
    fi
  fi
fi

# 3. Restore firewall rules
GW_PORT=%d
SPINE_PORT=%d

echo "--> Removing firewall rules..."
if command -v ufw >/dev/null 2>&1 && ufw status | grep -qw "active"; then
  ufw delete allow "${GW_PORT}/tcp" 2>/dev/null || true
  ufw delete allow "${SPINE_PORT}/udp" 2>/dev/null || true
elif command -v firewall-cmd >/dev/null 2>&1 && systemctl is-active --quiet firewalld; then
  firewall-cmd --permanent --remove-port="${GW_PORT}/tcp" 2>/dev/null || true
  firewall-cmd --permanent --remove-port="${SPINE_PORT}/udp" 2>/dev/null || true
  firewall-cmd --reload 2>/dev/null || true
elif command -v iptables >/dev/null 2>&1; then
  iptables -D INPUT -p tcp --dport "${GW_PORT}" -j ACCEPT 2>/dev/null || true
  iptables -D INPUT -p udp --dport "${SPINE_PORT}" -j ACCEPT 2>/dev/null || true
fi

# 4. Remove /etc/fak configuration
echo "--> Removing configuration in /etc/fak..."
rm -f /etc/fak/strix-halo.env /etc/fak/policy.json
rmdir /etc/fak 2>/dev/null || true

echo "=== Uninstallation complete ==="
`,
		cfg.Port,
		cfg.FleetSpinePort,
	)
	files["uninstall.sh"] = []byte(uninstallSh)

	// 9. verify.sh
	dualTP2Verify := ""
	if cfg.ClusterMode == ClusterModeDualTP2 {
		dualTP2Verify = fmt.Sprintf(`
# 4. Check Dual-Node USB4 RoCEv2 & Peer Daemon (ClusterMode: dual_tp2)
echo "Checking Dual-Node USB4 RoCEv2 & Peer Daemon..."
USB4_DEV="${USB4_IFACE:-thunderbolt0}"
echo -n "  - USB4 point-to-point interface (${USB4_DEV}): "
if ip link show "$USB4_DEV" >/dev/null 2>&1; then
  echo "[PASS]"
else
  echo "[WARN] (interface ${USB4_DEV} not detected)"
fi

echo -n "  - Peer node reachability (%s): "
if ping -c 1 -W 2 "%s" >/dev/null 2>&1; then
  echo "[PASS]"
else
  echo "[WARN] (peer node %s not reachable over USB4 link)"
fi

echo -n "  - Peer daemon discovery (http://%s:%d/healthz): "
if command -v curl >/dev/null 2>&1; then
  if curl -s -f -m 2 "http://%s:%d/healthz" >/dev/null 2>&1; then
    echo "[PASS]"
  else
    echo "[WARN] (peer daemon at http://%s:%d/healthz not reachable)"
  fi
else
  echo "[SKIP] (curl not installed)"
fi
`,
			cfg.ClusterPeerIP,
			cfg.ClusterPeerIP,
			cfg.ClusterPeerIP,
			cfg.ClusterPeerIP,
			cfg.Port,
			cfg.ClusterPeerIP,
			cfg.Port,
			cfg.ClusterPeerIP,
			cfg.Port,
		)
	}

	verifySh := fmt.Sprintf(`#!/usr/bin/env bash
# Verification and health-check script for fak AMD Strix Halo APU node
set -euo pipefail

GW_PORT=%d
HEALTHZ_URL="http://127.0.0.1:${GW_PORT}/healthz"

echo "=== Verifying fak Strix Halo Node Health ==="
failures=0

# 1. Check Gateway Healthz
echo -n "Checking gateway healthz (${HEALTHZ_URL})... "
if command -v curl >/dev/null 2>&1; then
  if curl -s -f -m 5 "$HEALTHZ_URL" >/dev/null; then
    echo "[PASS]"
  else
    echo "[FAIL] (gateway endpoint unreachable or returned error)"
    failures=$((failures + 1))
  fi
else
  echo "[SKIP] (curl not installed)"
fi

# 2. Check Strix Halo Gotchas
echo "Checking Strix Halo gotcha mitigations..."

# Gotcha: amdgpu.lockup_timeout=-1
echo -n "  - Kernel cmdline amdgpu.lockup_timeout=-1: "
if [[ -f /proc/cmdline ]] && grep -q "amdgpu.lockup_timeout=-1" /proc/cmdline; then
  echo "[PASS]"
else
  echo "[WARN] (amdgpu.lockup_timeout=-1 not found in /proc/cmdline; reboot may be pending)"
fi

# Gotcha: ttm.pages_limit
echo -n "  - TTM pages_limit configured: "
if [[ -f /sys/module/ttm/parameters/pages_limit ]]; then
  limit=$(cat /sys/module/ttm/parameters/pages_limit)
  if [[ "$limit" -ge 14000000 ]]; then
    echo "[PASS] (${limit} pages)"
  else
    echo "[WARN] (${limit} pages; kernel default 50%% ceiling may be active)"
  fi
elif [[ -f /proc/cmdline ]] && grep -q "ttm.pages_limit=" /proc/cmdline; then
  echo "[PASS] (present in /proc/cmdline; reboot may be pending)"
else
  echo "[WARN] (/sys/module/ttm/parameters/pages_limit not readable)"
fi

# Gotcha: DPM governor
echo -n "  - AMD DPM performance level: "
dpm_ok=0
for card in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  if [[ -f "$card" ]]; then
    level=$(cat "$card" 2>/dev/null || true)
    if [[ "$level" == "high" ]]; then
      dpm_ok=1
    fi
  fi
done
if [[ $dpm_ok -eq 1 ]]; then
  echo "[PASS] (locked to high)"
else
  echo "[WARN] (DPM not locked to high; ensure fak-strix-governor.service is active)"
fi

# Gotchas #9 & #5: toxic overrides unset
echo -n "  - Safe ROCm APU environment (toxic overrides unset): "
if [[ -n "${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-}" ]]; then
  echo "[WARN] (GGML_CUDA_ENABLE_UNIFIED_MEMORY is set; must be unset to avoid corrupting APU output)"
elif [[ "${HSA_OVERRIDE_GFX_VERSION:-}" == "11.0.0" ]]; then
  echo "[WARN] (HSA_OVERRIDE_GFX_VERSION=11.0.0 is set; must be unset to avoid SIGSEGV in libamdhip64)"
else
  echo "[PASS]"
fi

# Vulkan runtime capabilities and SPIR-V acceleration assets
echo -n "  - Vulkan runtime capabilities: "
FAK_BIN="%s"
if command -v "$FAK_BIN" >/dev/null 2>&1; then
  cap_out=$("$FAK_BIN" runtime-capabilities --backend vulkan 2>&1 || true)
  if echo "$cap_out" | grep -q '"status":\s*"available"'; then
    echo "[PASS] (backend: vulkan status: available)"
  else
    echo "[WARN] (fak runtime-capabilities does not report status: available for vulkan)"
  fi
else
  echo "[SKIP] ($FAK_BIN not found)"
fi

echo -n "  - Vulkan SPIR-V acceleration assets: "
if [[ -d /var/lib/fak/spirv ]]; then
  echo "[PASS]"
else
  echo "[WARN] (/var/lib/fak/spirv not found; GPU acceleration requires compiled SPIR-V shaders)"
fi

# 3. Check systemd services
if command -v systemctl >/dev/null 2>&1; then
  echo -n "  - fak-serve.service status: "
  if systemctl is-active --quiet fak-serve.service; then
    echo "[PASS]"
  else
    echo "[WARN] (service is $(systemctl is-active fak-serve.service 2>/dev/null || echo 'inactive'))"
  fi
fi
%s
if [[ $failures -eq 0 ]]; then
  echo "=== All primary checks completed successfully! ==="
  exit 0
else
  echo "=== Completed with ${failures} failure(s) ==="
  exit 1
fi
`,
		cfg.Port,
		cfg.FakBinaryPath,
		dualTP2Verify,
	)
	files["verify.sh"] = []byte(verifySh)

	// 10. clients/lan-agent.env
	lanAgentEnv := fmt.Sprintf(`#!/usr/bin/env bash
# Environment configuration for LAN agents connecting to Strix Halo fak server
export FAK_BASE_URL="http://%s:%d"
export FAK_GATEWAY_KEY="%s"
export OPENAI_BASE_URL="http://%s:%d/v1"
export OPENAI_API_KEY="%s"
export FLEET_SPINE_GROUP="%s"
export FLEET_SPINE_PORT="%d"
`,
		cfg.LANIP,
		cfg.Port,
		cfg.GatewayKey,
		cfg.LANIP,
		cfg.Port,
		cfg.GatewayKey,
		cfg.FleetSpineGroup,
		cfg.FleetSpinePort,
	)
	files["clients/lan-agent.env"] = []byte(lanAgentEnv)

	// 11. clients/lan-agent.ps1
	lanAgentPs1 := fmt.Sprintf(`# Environment configuration for LAN agents connecting to Strix Halo fak server
$env:FAK_BASE_URL = "http://%s:%d"
$env:FAK_GATEWAY_KEY = "%s"
$env:OPENAI_BASE_URL = "http://%s:%d/v1"
$env:OPENAI_API_KEY = "%s"
$env:FLEET_SPINE_GROUP = "%s"
$env:FLEET_SPINE_PORT = "%d"
`,
		cfg.LANIP,
		cfg.Port,
		cfg.GatewayKey,
		cfg.LANIP,
		cfg.Port,
		cfg.GatewayKey,
		cfg.FleetSpineGroup,
		cfg.FleetSpinePort,
	)
	files["clients/lan-agent.ps1"] = []byte(lanAgentPs1)

	// 12. clients/.mcp.json
	mcpJSON := fmt.Sprintf(`{
  "mcpServers": {
    "fak-strix-halo": {
      "url": "http://%s:%d/mcp",
      "headers": {
        "Authorization": "Bearer %s"
      }
    }
  }
}
`,
		cfg.LANIP,
		cfg.Port,
		cfg.GatewayKey,
	)
	files["clients/.mcp.json"] = []byte(mcpJSON)

	// 13. clients/README.md
	clientReadme := fmt.Sprintf(`# AMD Strix Halo LAN Agent Connection Guide

This directory provides environment configurations for local network agents connecting to the
AMD Strix Halo (Ryzen AI MAX+ 395) unified memory inference server.

## Node Coordinates
- **Gateway Base URL:** http://%s:%d
- **OpenAI Compatible Endpoint:** http://%s:%d/v1
- **Model ID:** %s
- **Fleet Spine Multicast:** %s:%d

## Quickstart

### 1. Linux / macOS Bash Agents
Run:
`+"```bash"+`
source clients/lan-agent.env
`+"```"+`
This sets `+"`FAK_BASE_URL`"+`, `+"`FAK_GATEWAY_KEY`"+`, `+"`OPENAI_BASE_URL`"+`, and `+"`OPENAI_API_KEY`"+`.

### 2. Windows PowerShell Agents
Run:
`+"```powershell"+`
. .\clients\lan-agent.ps1
`+"```"+`

### 3. Claude Code / Cursor / MCP Agents
Copy `+"`clients/.mcp.json`"+` to your project root or configure your MCP client with:
- **Server URL:** `+"`http://%s:%d/mcp`"+`
- **Header:** `+"`Authorization: Bearer ${FAK_GATEWAY_KEY}` (refer to `clients/lan-agent.env`)"+`

### 4. Health Check
Verify network reachability from your client machine:
`+"```bash"+`
curl -H "Authorization: Bearer ${FAK_GATEWAY_KEY}" http://%s:%d/healthz
`+"```"+`
`,
		cfg.LANIP, cfg.Port,
		cfg.LANIP, cfg.Port,
		cfg.ModelID,
		cfg.FleetSpineGroup, cfg.FleetSpinePort,
		cfg.LANIP, cfg.Port,
		cfg.LANIP, cfg.Port,
	)
	files["clients/README.md"] = []byte(clientReadme)

	// Calculate cryptographic checksums for all files to populate manifest
	manifestFiles := make(map[string]string)
	for path, content := range files {
		h := sha256.Sum256(content)
		manifestFiles[path] = hex.EncodeToString(h[:])
	}

	manifest := StrixPackageManifest{
		Schema:          "fak.strix-installer-manifest/1",
		Platform:        string(cfg.Platform),
		GeneratedAt:     time.Now().UTC().Format(time.RFC3339),
		GatewayURL:      fmt.Sprintf("http://%s:%d", cfg.LANIP, cfg.Port),
		GatewayKey:      cfg.GatewayKey,
		FleetSpineGroup: cfg.FleetSpineGroup,
		FleetSpinePort:  cfg.FleetSpinePort,
		Files:           manifestFiles,
	}

	manifestJSON, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("amdgpu: failed to marshal package manifest: %w", err)
	}
	manifestJSON = append(manifestJSON, '\n')
	files["manifest.json"] = manifestJSON

	return &StrixPackage{
		Config:   cfg,
		Manifest: manifest,
		Files:    files,
	}, nil
}

// isSecretBearingFile reports whether relPath contains sensitive gateway credentials or secrets.
func isSecretBearingFile(relPath string) bool {
	clean := filepath.ToSlash(relPath)
	return strings.HasSuffix(clean, ".env") ||
		strings.HasSuffix(clean, ".ps1") ||
		clean == "manifest.json" ||
		strings.HasSuffix(clean, "/manifest.json") ||
		(strings.HasSuffix(clean, ".json") && strings.Contains(clean, "clients"))
}

// WriteToDir writes all installer package files to the target directory on disk,
// creating subdirectories and applying appropriate execution and read permissions.
func (p *StrixPackage) WriteToDir(dir string) error {
	if dir == "" {
		dir = p.Config.OutputDir
	}
	if dir == "" {
		dir = "fak-strix-halo-pkg"
	}

	sensitiveDirs := make(map[string]bool)
	sensitiveDirs["."] = true
	for relPath := range p.Files {
		if isSecretBearingFile(relPath) {
			cleanRel := filepath.ToSlash(relPath)
			for d := path.Dir(cleanRel); d != "." && d != "/"; d = path.Dir(d) {
				sensitiveDirs[d] = true
			}
		}
	}

	dirPerm := os.FileMode(0755)
	if sensitiveDirs["."] {
		dirPerm = 0700
	}
	if err := os.MkdirAll(dir, dirPerm); err != nil {
		return fmt.Errorf("amdgpu: failed to create output directory %q: %w", dir, err)
	}
	if err := os.Chmod(dir, dirPerm); err != nil {
		return fmt.Errorf("amdgpu: failed to chmod output directory %q: %w", dir, err)
	}

	for relPath, content := range p.Files {
		cleanRel := filepath.ToSlash(relPath)
		parentRel := path.Dir(cleanRel)
		fullPath := filepath.Join(dir, filepath.FromSlash(relPath))
		parentDir := filepath.Dir(fullPath)

		subDirPerm := os.FileMode(0755)
		if sensitiveDirs[parentRel] {
			subDirPerm = 0700
		}
		if err := os.MkdirAll(parentDir, subDirPerm); err != nil {
			return fmt.Errorf("amdgpu: failed to create parent directory for %q: %w", relPath, err)
		}
		if err := os.Chmod(parentDir, subDirPerm); err != nil {
			return fmt.Errorf("amdgpu: failed to chmod directory %q: %w", parentDir, err)
		}

		perm := os.FileMode(0644)
		if strings.HasSuffix(relPath, ".sh") {
			perm = 0755
		} else if isSecretBearingFile(relPath) {
			perm = 0600
		}
		if err := os.WriteFile(fullPath, content, perm); err != nil {
			return fmt.Errorf("amdgpu: failed to write file %q: %w", relPath, err)
		}
		if err := os.Chmod(fullPath, perm); err != nil {
			return fmt.Errorf("amdgpu: failed to chmod file %q: %w", relPath, err)
		}
	}
	return nil
}

// RunStrixInstallerCLI provides the command-line interface for generating Strix Halo
// installer packages with LAN communications and gotcha settings.
func RunStrixInstallerCLI(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-strix-package", flag.ContinueOnError)
	fs.SetOutput(stderr)

	dir := fs.String("dir", "fak-strix-halo-pkg", "output directory for generated installer package")
	lanIP := fs.String("lan-ip", "", "LAN IP address for agent access (default: probed outbound IP)")
	port := fs.Int("port", 8080, "gateway HTTP listener port")
	modelPort := fs.Int("model-port", 8131, "upstream model server port")
	model := fs.String("model", "qwen3.6-27b", "upstream model ID")
	platform := fs.String("platform", "strix-halo-128", "hardware platform preset (strix-halo-128, strix-halo-64)")
	key := fs.String("key", "", "gateway authentication key (32-byte hex; generated if empty)")
	clusterMode := fs.String("cluster-mode", "standalone", "cluster mode (standalone, dual_tp2)")
	tp2 := fs.Bool("tp2", false, "enable dual-node Tensor Parallelism (TP=2) over USB4 RoCEv2")
	clusterPeer := fs.String("cluster-peer", "", "peer IPv4 address for dual-node TP=2 (default: 169.254.1.2)")
	apply := fs.Bool("apply", false, "write installer package files to disk")
	jsonOut := fs.Bool("json", false, "output package manifest as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "amd-strix-package: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}

	parsedPlatform, err := ParseStrixHaloPlatform(*platform)
	if err != nil {
		fmt.Fprintf(stderr, "amd-strix-package error: %v\n", err)
		return 2
	}

	cfg := DefaultStrixInstallerConfig()
	cfg.Platform = parsedPlatform
	if *tp2 {
		cfg.ClusterMode = ClusterModeDualTP2
	} else if *clusterMode != "" {
		cfg.ClusterMode = *clusterMode
	}
	if *clusterPeer != "" {
		cfg.ClusterPeerIP = *clusterPeer
	}
	if *lanIP != "" {
		cfg.LANIP = *lanIP
	}
	if *port > 0 {
		cfg.Port = *port
	}
	if *modelPort > 0 {
		cfg.ModelPort = *modelPort
	}
	if *model != "" {
		cfg.ModelID = *model
	}
	if *key != "" {
		cfg.GatewayKey = *key
	}
	if *dir != "" {
		cfg.OutputDir = *dir
	}

	pkg, err := GenerateStrixInstallerPackage(cfg)
	if err != nil {
		fmt.Fprintf(stderr, "amd-strix-package generation error: %v\n", err)
		return 1
	}

	if *apply {
		if err := pkg.WriteToDir(cfg.OutputDir); err != nil {
			fmt.Fprintf(stderr, "amd-strix-package write error: %v\n", err)
			return 1
		}
	}

	if *jsonOut {
		raw, err := json.MarshalIndent(pkg.Manifest, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "amd-strix-package json error: %v\n", err)
			return 1
		}
		stdout.Write(raw)
		stdout.Write([]byte("\n"))
		return 0
	}

	if *apply {
		fmt.Fprintf(stdout, "AMD Strix Halo (%s) installer package successfully written to %q\n\n", cfg.Platform, cfg.OutputDir)
	} else {
		fmt.Fprintf(stdout, "AMD Strix Halo (%s) installer package plan (dry-run):\n\n", cfg.Platform)
	}
	fmt.Fprintf(stdout, "Target Directory:   %s\n", cfg.OutputDir)
	fmt.Fprintf(stdout, "Gateway URL:        http://%s:%d\n", cfg.LANIP, cfg.Port)
	fmt.Fprintf(stdout, "Gateway Auth Key:   %s\n", cfg.GatewayKey)
	fmt.Fprintf(stdout, "Model Upstream:     http://127.0.0.1:%d/v1 (model: %s)\n", cfg.ModelPort, cfg.ModelID)
	fmt.Fprintf(stdout, "Fleet Spine:        %s:%d\n\n", cfg.FleetSpineGroup, cfg.FleetSpinePort)

	var fileList []string
	for f := range pkg.Files {
		fileList = append(fileList, f)
	}
	sort.Strings(fileList)

	fmt.Fprintf(stdout, "Package Files (%d files):\n", len(fileList))
	for _, f := range fileList {
		fmt.Fprintf(stdout, "  - %s\n", f)
	}
	if !*apply {
		fmt.Fprintln(stdout, "\nTo write files to disk, run with --apply:")
		fmt.Fprintf(stdout, "  fak-dev amd-strix-package --dir %s --apply\n", cfg.OutputDir)
	} else {
		fmt.Fprintln(stdout, "\nTo install on host:")
		fmt.Fprintf(stdout, "  cd %s && sudo ./install.sh\n", cfg.OutputDir)
	}

	return 0
}
