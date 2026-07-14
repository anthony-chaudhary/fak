// Package systemservice renders service-manager definitions that keep fak's
// control plane outside terminal and compositor process trees.
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

func RenderSystemdUserUnit(c SystemdConfig) (string, error) {
	if strings.TrimSpace(c.Executable) == "" || strings.ContainsAny(c.Executable, "\r\n") {
		return "", fmt.Errorf("invalid executable")
	}
	if strings.TrimSpace(c.StateDir) == "" || strings.ContainsAny(c.StateDir, "\r\n") {
		return "", fmt.Errorf("invalid state directory")
	}
	exec := systemdQuote(c.Executable)
	state := systemdQuote(c.StateDir)
	return `[Unit]
Description=fak Guard OS-owned control plane
Documentation=https://github.com/anthony-chaudhary/fak
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=` + exec + ` service run --interval 15s
Restart=always
RestartSec=3s
KillMode=control-group
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=full
ProtectControlGroups=yes
ProtectKernelModules=yes
ProtectKernelTunables=yes
RestrictSUIDSGID=yes
LockPersonality=yes
MemoryMax=1G
TasksMax=256
CPUQuota=50%
Environment=FAK_SERVICE_MANAGER=systemd
Environment=FLEET_REG_DIR=` + state + `

[Install]
WantedBy=default.target
`, nil
}

func systemdQuote(s string) string { return strconv.Quote(s) }
