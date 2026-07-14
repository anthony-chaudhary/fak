package systemservice

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"strings"
)

const LaunchdLabel = "com.fak.guard-control"

type LaunchdConfig struct{ Executable, StateDir, StdoutPath, StderrPath string }

type launchdPlist struct {
	XMLName xml.Name    `xml:"plist"`
	Version string      `xml:"version,attr"`
	Dict    launchdDict `xml:"dict"`
}
type launchdDict struct {
	Items []launchdItem `xml:",any"`
}
type launchdItem struct {
	XMLName  xml.Name
	Value    string        `xml:",chardata"`
	Children []launchdItem `xml:",any"`
}

func RenderLaunchAgent(c LaunchdConfig) (string, error) {
	for _, v := range []string{c.Executable, c.StateDir, c.StdoutPath, c.StderrPath} {
		if strings.TrimSpace(v) == "" || strings.ContainsAny(v, "\x00\r\n") {
			return "", fmt.Errorf("invalid launchd path")
		}
	}
	// Hand-render the tiny plist so ProgramArguments remains an array and every
	// operator-supplied path is XML-escaped by encoding/xml.
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
<key>EnvironmentVariables</key><dict><key>FAK_SERVICE_MANAGER</key><string>launchd</string><key>FLEET_REG_DIR</key><string>`)
	_ = xml.EscapeText(&b, []byte(c.StateDir))
	b.WriteString(`</string></dict>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><true/>
<key>ThrottleInterval</key><integer>3</integer>
<key>ProcessType</key><string>Background</string>
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
