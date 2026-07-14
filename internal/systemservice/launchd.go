package systemservice

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"strings"
)

const LaunchdLabel = "com.fak.guard-control"

type LaunchdConfig struct {
	Executable string
	StateDir   string
	StdoutPath string
	StderrPath string
	UserName   string
}

// RenderLaunchDaemon renders a system-domain launchd service. It is loaded by
// PID 1 from /Library/LaunchDaemons and survives Terminal, WindowServer, and
// GUI-login teardown. UserName may drop data-plane privileges while launchd
// retains lifecycle ownership.
func RenderLaunchDaemon(c LaunchdConfig) (string, error) {
	for name, v := range map[string]string{"executable": c.Executable, "state directory": c.StateDir, "stdout path": c.StdoutPath, "stderr path": c.StderrPath, "user": c.UserName} {
		if strings.TrimSpace(v) == "" || strings.ContainsAny(v, "\x00\r\n") {
			return "", fmt.Errorf("invalid launchd %s", name)
		}
	}
	var b bytes.Buffer
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>`)
	_ = xml.EscapeText(&b, []byte(LaunchdLabel))
	b.WriteString(`</string>
<key>ProgramArguments</key><array><string>`)
	_ = xml.EscapeText(&b, []byte(c.Executable))
	b.WriteString(`</string><string>service</string><string>run</string><string>--interval</string><string>15s</string></array>
<key>UserName</key><string>`)
	_ = xml.EscapeText(&b, []byte(c.UserName))
	b.WriteString(`</string>
<key>EnvironmentVariables</key><dict><key>FAK_SERVICE_MANAGER</key><string>launchd-system</string><key>FLEET_REG_DIR</key><string>`)
	_ = xml.EscapeText(&b, []byte(c.StateDir))
	b.WriteString(`</string></dict>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ThrottleInterval</key><integer>3</integer>
<key>ProcessType</key><string>Background</string>
<key>EnableTransactions</key><true/>
<key>SoftResourceLimits</key><dict><key>NumberOfFiles</key><integer>4096</integer><key>NumberOfProcesses</key><integer>256</integer></dict>
<key>StandardOutPath</key><string>`)
	_ = xml.EscapeText(&b, []byte(c.StdoutPath))
	b.WriteString(`</string>
<key>StandardErrorPath</key><string>`)
	_ = xml.EscapeText(&b, []byte(c.StderrPath))
	b.WriteString(`</string>
</dict></plist>
`)
	return b.String(), nil
}

// RenderLaunchAgent remains as a source-compatible alias, but deliberately
// renders the system-domain contract. New callers should use RenderLaunchDaemon.
func RenderLaunchAgent(c LaunchdConfig) (string, error) { return RenderLaunchDaemon(c) }
