// Package systemservice renders service-manager definitions that keep fak's
// control plane outside terminal, compositor, and login-session process trees.
package systemservice

import (
	"fmt"
	"strconv"
	"strings"
)

const SystemdUnitName = "fak-guard-control.service"

type SystemdConfig struct {
	Executable string
	StateDir   string
}

// RenderSystemdSystemUnit renders a PID-1-owned system service. systemd allocates
// a transient, unprivileged identity while PID 1 retains lifecycle and durable-
// directory ownership; no login account or user manager is in the dependency chain.
func RenderSystemdSystemUnit(c SystemdConfig) (string, error) {
	for name, value := range map[string]string{"executable": c.Executable, "state directory": c.StateDir} {
		if strings.TrimSpace(value) == "" || strings.ContainsAny(value, "\x00\r\n") {
			return "", fmt.Errorf("invalid %s", name)
		}
	}
	exec := systemdQuote(c.Executable)
	state := systemdQuote(c.StateDir)
	return `[Unit]
Description=fak Guard OS-owned control plane
Documentation=https://github.com/anthony-chaudhary/fak
After=local-fs.target network-online.target
Wants=network-online.target

[Service]
Type=notify
NotifyAccess=main
WatchdogSec=30s
DynamicUser=yes
RuntimeDirectory=fak
RuntimeDirectoryMode=0700
ExecStart=` + exec + ` service run --interval 15s --notify systemd
Restart=always
RestartSec=3s
KillMode=control-group
UMask=0077
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ProtectControlGroups=yes
ProtectKernelModules=yes
ProtectKernelTunables=yes
ProtectKernelLogs=yes
ProtectClock=yes
ProtectHostname=yes
RestrictSUIDSGID=yes
LockPersonality=yes
MemoryDenyWriteExecute=yes
RestrictRealtime=yes
SystemCallArchitectures=native
ReadWritePaths=` + state + `
MemoryMax=1G
TasksMax=256
CPUQuota=50%
Environment=FAK_SERVICE_MANAGER=systemd-system
Environment=FLEET_REG_DIR=` + state + `

[Install]
WantedBy=multi-user.target
`, nil
}

// RenderSystemdUserUnit remains as a source-compatible alias, but deliberately
// renders the system-owned contract. New callers should use the explicit name.
func RenderSystemdUserUnit(c SystemdConfig) (string, error) { return RenderSystemdSystemUnit(c) }

func systemdQuote(s string) string { return strconv.Quote(s) }
