package systemservice

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
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

// LaunchdProjectionSchemaV1 names the versioned launchd projection document.
const LaunchdProjectionSchemaV1 = "fak.service.launchd.v1"

// LaunchdType distinguishes system-wide daemons from user-session agents.
type LaunchdType string

const (
	LaunchDaemon LaunchdType = "LaunchDaemon"
	LaunchAgent  LaunchdType = "LaunchAgent"
)

// LaunchdKeepAlive models the launchd.plist KeepAlive configuration.
// It can be unconditional (Always: true), or bounded by condition flags.
type LaunchdKeepAlive struct {
	Always         bool  `json:"always,omitempty"`
	SuccessfulExit *bool `json:"successful_exit,omitempty"`
	Crashed        *bool `json:"crashed,omitempty"`
}

// LaunchdCommands carries modern launchctl command strings.
type LaunchdCommands struct {
	Bootstrap string `json:"bootstrap"`
	Bootout   string `json:"bootout"`
	Kickstart string `json:"kickstart"`
	Print     string `json:"print"`
	Enable    string `json:"enable,omitempty"`
	Disable   string `json:"disable,omitempty"`
}

// LaunchdInput carries host-specific facts for projecting a portable spec
// into macOS launchd.
type LaunchdInput struct {
	Type              LaunchdType
	Domain            string
	UID               int
	PlistDir          string
	User              string
	GroupName         string
	StandardOutPath   string
	StandardErrorPath string
	ThrottleInterval  int
	EnvironmentFiles  []string
	KeepAlive         *LaunchdKeepAlive
}

// LaunchdProjection is the desired launchd form of one workload.
type LaunchdProjection struct {
	Schema         string               `json:"schema"`
	Identity       servicespec.Identity `json:"identity"`
	Kind           servicespec.Kind     `json:"kind"`
	Type           LaunchdType          `json:"type"`
	Label          string               `json:"label"`
	Domain         string               `json:"domain"`
	Target         string               `json:"target"`
	PlistPath      string               `json:"plist_path"`
	PlistXML       string               `json:"plist_xml"`
	KeepAlive      LaunchdKeepAlive     `json:"keep_alive"`
	Commands       LaunchdCommands      `json:"commands"`
	Enabled        bool                 `json:"enabled"`
	DesiredStopped bool                 `json:"desired_stopped"`
}

func (p *LaunchdProjection) BootstrapCommand() string { return p.Commands.Bootstrap }
func (p *LaunchdProjection) BootoutCommand() string   { return p.Commands.Bootout }
func (p *LaunchdProjection) KickstartCommand() string { return p.Commands.Kickstart }
func (p *LaunchdProjection) PrintCommand() string     { return p.Commands.Print }
func (p *LaunchdProjection) EnableCommand() string    { return p.Commands.Enable }
func (p *LaunchdProjection) DisableCommand() string   { return p.Commands.Disable }
func (p *LaunchdProjection) PlistText() string        { return p.PlistXML }
func (p *LaunchdProjection) UnitText() string         { return p.PlistXML }

// LaunchdStatus represents the parsed runtime status from launchctl print.
// Program, Arguments, Runs, and Properties describe the job definition
// launchd currently has loaded, so a caller can compare it against the plist
// on disk and detect a stale loaded definition.
type LaunchdStatus struct {
	Target         string               `json:"target,omitempty"`
	Label          string               `json:"label"`
	Domain         string               `json:"domain,omitempty"`
	Type           string               `json:"type,omitempty"`
	State          string               `json:"state"`
	PID            int                  `json:"pid,omitempty"`
	PlistPath      string               `json:"plist_path,omitempty"`
	Program        string               `json:"program,omitempty"`
	Arguments      []string             `json:"arguments,omitempty"`
	Runs           int                  `json:"runs,omitempty"`
	Properties     []string             `json:"properties,omitempty"`
	LastExitCode   *int                 `json:"last_exit_code,omitempty"`
	ExitReason     string               `json:"exit_reason,omitempty"`
	Phase          servicespec.Phase    `json:"phase"`
	Observed       servicespec.Observed `json:"observed"`
	ObservedStatus servicespec.Observed `json:"observed_status"`
}

func (s *LaunchdStatus) ExitCode() (int, bool) {
	if s.LastExitCode != nil {
		return *s.LastExitCode, true
	}
	return 0, false
}

func (s *LaunchdStatus) ObservedRecord() servicespec.Observed {
	return s.Observed
}

func (s *LaunchdStatus) IsRunning() bool {
	return s.State == "running" || s.PID > 0
}

// Errors returned by launchd projection and parsing.
var (
	ErrUnknownLaunchdType       = errors.New("systemservice: unknown launchd type")
	ErrBadLabelToken            = errors.New("systemservice: identity/dependency is not a valid launchd label token")
	ErrEmptyLaunchctlPrint      = errors.New("systemservice: empty launchctl print output")
	ErrMalformedLaunchctlOutput = errors.New("systemservice: malformed launchctl print output")
	ErrLaunchctlServiceNotFound = errors.New("systemservice: launchctl service not found")
)

// LaunchdLabelFor derives the deterministic launchd service label for a workload.
func LaunchdLabelFor(workload string) string {
	if strings.HasPrefix(workload, "com.fak.") {
		return workload
	}
	return "com.fak." + workload
}

func normalizeLaunchdType(t LaunchdType) (LaunchdType, error) {
	switch strings.ToLower(string(t)) {
	case "launchdaemon", "daemon", "system":
		return LaunchDaemon, nil
	case "launchagent", "agent", "gui", "user":
		return LaunchAgent, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrUnknownLaunchdType, t)
	}
}

// ProjectLaunchd derives the desired launchd form of the spec.
func ProjectLaunchd(spec *servicespec.Spec, in LaunchdInput) (*LaunchdProjection, error) {
	if spec == nil {
		return nil, ErrNilSpec
	}
	s := *spec
	s.Normalize()
	if err := s.Validate(); err != nil {
		return nil, err
	}
	normType, err := normalizeLaunchdType(in.Type)
	if err != nil {
		return nil, err
	}
	if !unitToken.MatchString(s.Identity.Workload) {
		return nil, fmt.Errorf("%w: workload %q", ErrBadLabelToken, s.Identity.Workload)
	}
	for _, d := range s.DependsOn {
		if !unitToken.MatchString(d) {
			return nil, fmt.Errorf("%w: dependency %q", ErrBadLabelToken, d)
		}
	}
	for name, v := range map[string]string{
		"identity.node": s.Identity.Node, "identity.service": s.Identity.Service,
		"dir": s.Dir, "checkpoint_dir": s.CheckpointDir, "user": in.User, "group": in.GroupName,
		"stdout path": in.StandardOutPath, "stderr path": in.StandardErrorPath,
		"plist dir": in.PlistDir, "domain": in.Domain,
	} {
		if strings.ContainsAny(v, "\x00\r\n") {
			return nil, fmt.Errorf("systemservice: %s contains a control character", name)
		}
	}
	for _, a := range s.Command {
		if strings.ContainsAny(a, "\x00\r\n") {
			return nil, errors.New("systemservice: command argument contains a control character")
		}
	}
	for _, f := range in.EnvironmentFiles {
		if strings.TrimSpace(f) == "" || strings.ContainsAny(f, "\x00\r\n") {
			return nil, errors.New("systemservice: environment file reference is empty or contains a control character")
		}
	}
	for _, e := range s.Env {
		if strings.ContainsAny(e.Name+e.Value, "\x00\r\n") {
			return nil, fmt.Errorf("systemservice: env %q contains a control character", e.Name)
		}
		if e.SecretRef != "" && len(in.EnvironmentFiles) == 0 {
			return nil, fmt.Errorf("%w (env %q)", ErrSecretNeedsFile, e.Name)
		}
	}

	domain := in.Domain
	if domain == "" {
		if normType == LaunchDaemon {
			domain = "system"
		} else {
			uid := in.UID
			if uid <= 0 {
				if u := os.Getuid(); u > 0 {
					uid = u
				} else {
					uid = 501
				}
			}
			domain = fmt.Sprintf("gui/%d", uid)
		}
	}

	label := LaunchdLabelFor(s.Identity.Workload)
	target := domain + "/" + label

	plistDir := in.PlistDir
	if plistDir == "" {
		if normType == LaunchDaemon {
			plistDir = "/Library/LaunchDaemons"
		} else {
			if domain == "system" {
				plistDir = "/Library/LaunchAgents"
			} else if home := os.Getenv("HOME"); home != "" && in.User == "" {
				plistDir = filepath.Join(home, "Library", "LaunchAgents")
			} else {
				plistDir = "~/Library/LaunchAgents"
			}
		}
	}
	plistPath := filepath.Join(plistDir, label+".plist")

	commands := LaunchdCommands{
		Bootstrap: fmt.Sprintf("launchctl bootstrap %s %s", domain, plistPath),
		Bootout:   fmt.Sprintf("launchctl bootout %s", target),
		Kickstart: fmt.Sprintf("launchctl kickstart -k %s", target),
		Print:     fmt.Sprintf("launchctl print %s", target),
		Enable:    fmt.Sprintf("launchctl enable %s", target),
		Disable:   fmt.Sprintf("launchctl disable %s", target),
	}

	var keepAlive LaunchdKeepAlive
	if in.KeepAlive != nil {
		keepAlive = *in.KeepAlive
	} else if s.Kind == servicespec.KindJob {
		bFalse := false
		keepAlive = LaunchdKeepAlive{SuccessfulExit: &bFalse}
	} else {
		keepAlive = LaunchdKeepAlive{Always: true}
	}

	throttle := in.ThrottleInterval
	if throttle <= 0 {
		if s.Restart.InitialBackoffMS > 0 {
			throttle = int(s.Restart.InitialBackoffMS / 1000)
			if throttle <= 0 {
				throttle = 1
			}
		} else {
			throttle = 3
		}
	}

	plistXML, err := buildPlistXML(&s, in, normType, label, keepAlive, throttle)
	if err != nil {
		return nil, err
	}

	p := &LaunchdProjection{
		Schema:         LaunchdProjectionSchemaV1,
		Identity:       s.Identity,
		Kind:           s.Kind,
		Type:           normType,
		Label:          label,
		Domain:         domain,
		Target:         target,
		PlistPath:      plistPath,
		PlistXML:       plistXML,
		KeepAlive:      keepAlive,
		Commands:       commands,
		Enabled:        s.Desired != servicespec.DesiredStopped,
		DesiredStopped: s.Desired == servicespec.DesiredStopped,
	}
	return p, nil
}

func buildPlistXML(s *servicespec.Spec, in LaunchdInput, launchdType LaunchdType, label string, keepAlive LaunchdKeepAlive, throttle int) (string, error) {
	var b bytes.Buffer
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	b.WriteString(`<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">` + "\n")
	b.WriteString(`<plist version="1.0"><dict>` + "\n")

	b.WriteString(`<key>Label</key><string>`)
	_ = xml.EscapeText(&b, []byte(label))
	b.WriteString("</string>\n")

	b.WriteString(`<key>ProgramArguments</key><array>`)
	for _, arg := range s.Command {
		b.WriteString("<string>")
		_ = xml.EscapeText(&b, []byte(arg))
		b.WriteString("</string>")
	}
	b.WriteString("</array>\n")

	if s.Dir != "" {
		b.WriteString(`<key>WorkingDirectory</key><string>`)
		_ = xml.EscapeText(&b, []byte(s.Dir))
		b.WriteString("</string>\n")
	}

	if launchdType == LaunchDaemon {
		if in.User != "" {
			b.WriteString(`<key>UserName</key><string>`)
			_ = xml.EscapeText(&b, []byte(in.User))
			b.WriteString("</string>\n")
		}
		if in.GroupName != "" {
			b.WriteString(`<key>GroupName</key><string>`)
			_ = xml.EscapeText(&b, []byte(in.GroupName))
			b.WriteString("</string>\n")
		}
	}

	envMap := make(map[string]string)
	for _, e := range s.Env {
		if e.SecretRef != "" {
			continue // referenced via environment file, never serialized
		}
		envMap[e.Name] = e.Value
	}
	if launchdType == LaunchDaemon {
		envMap["FAK_SERVICE_MANAGER"] = "launchd-system"
	} else {
		envMap["FAK_SERVICE_MANAGER"] = "launchd-agent"
	}
	envMap["FAK_SERVICE_NODE"] = s.Identity.Node
	envMap["FAK_SERVICE_NAME"] = s.Identity.Service
	envMap["FAK_SERVICE_WORKLOAD"] = s.Identity.Workload
	if s.CheckpointDir != "" {
		envMap["FLEET_REG_DIR"] = s.CheckpointDir
	}

	envKeys := make([]string, 0, len(envMap))
	for k := range envMap {
		envKeys = append(envKeys, k)
	}
	sort.Strings(envKeys)

	b.WriteString(`<key>EnvironmentVariables</key><dict>`)
	for _, k := range envKeys {
		b.WriteString("<key>")
		_ = xml.EscapeText(&b, []byte(k))
		b.WriteString("</key><string>")
		_ = xml.EscapeText(&b, []byte(envMap[k]))
		b.WriteString("</string>")
	}
	b.WriteString("</dict>\n")

	b.WriteString(`<key>RunAtLoad</key><true/>` + "\n")

	buildKeepAliveXML(&b, keepAlive)

	if throttle > 0 {
		b.WriteString(fmt.Sprintf("<key>ThrottleInterval</key><integer>%d</integer>\n", throttle))
	}

	b.WriteString(`<key>ProcessType</key><string>Background</string>` + "\n")
	b.WriteString(`<key>EnableTransactions</key><true/>` + "\n")
	b.WriteString(`<key>SoftResourceLimits</key><dict><key>NumberOfFiles</key><integer>4096</integer><key>NumberOfProcesses</key><integer>256</integer></dict>` + "\n")

	if in.StandardOutPath != "" {
		b.WriteString(`<key>StandardOutPath</key><string>`)
		_ = xml.EscapeText(&b, []byte(in.StandardOutPath))
		b.WriteString("</string>\n")
	}
	if in.StandardErrorPath != "" {
		b.WriteString(`<key>StandardErrorPath</key><string>`)
		_ = xml.EscapeText(&b, []byte(in.StandardErrorPath))
		b.WriteString("</string>\n")
	}

	b.WriteString("</dict></plist>\n")
	return b.String(), nil
}

func buildKeepAliveXML(b *bytes.Buffer, k LaunchdKeepAlive) {
	if k.Always {
		b.WriteString("<key>KeepAlive</key><true/>\n")
		return
	}
	if k.Crashed == nil && k.SuccessfulExit == nil {
		b.WriteString("<key>KeepAlive</key><false/>\n")
		return
	}
	b.WriteString("<key>KeepAlive</key><dict>")
	if k.Crashed != nil {
		if *k.Crashed {
			b.WriteString("<key>Crashed</key><true/>")
		} else {
			b.WriteString("<key>Crashed</key><false/>")
		}
	}
	if k.SuccessfulExit != nil {
		if *k.SuccessfulExit {
			b.WriteString("<key>SuccessfulExit</key><true/>")
		} else {
			b.WriteString("<key>SuccessfulExit</key><false/>")
		}
	}
	b.WriteString("</dict>\n")
}

// ParseLaunchctlPrint parses `launchctl print <target>` output into a typed LaunchdStatus.
//
// The output is a tab-indented dump: a column-0 `<target> = {` header whose
// directly indented keys describe the job, plus nested blocks (`arguments = {`,
// `environment = {`, `resource coalition = {`, `jetsam coalition = {`,
// `event triggers = {`, ...) indented one level deeper. Nested blocks reuse
// key names such as `state` and `type` (a coalition is `type = jetsam`,
// `state = active`), and launchctl sometimes prints a block's closing brace at
// the end of a content line, so top-level membership is decided by
// indentation, not by counting braces: a key belongs to the job exactly when
// its leading whitespace equals the first body line's. Without a header (a
// caller that pre-strips it) the first line's indentation is the top level.
// The top-level `arguments = {` block is copied verbatim, one argv entry per
// line with exactly one extra indent stripped. The XPC_SERVICE_NAME label
// fallback is honored at any depth because it lives inside `environment = {`.
func ParseLaunchctlPrint(output string) (*LaunchdStatus, error) {
	if strings.TrimSpace(output) == "" {
		return nil, ErrEmptyLaunchctlPrint
	}

	lines := strings.Split(strings.ReplaceAll(output, "\r\n", "\n"), "\n")
	var target, label, domain, launchdType, state, plistPath, exitReason, program, termSignal string
	var pid, runs int
	var lastExitCode *int
	var arguments, properties []string

	// The header is the first column-0 "<target> = {" line; anything before it
	// (a merged stderr warning) is ignored.
	bodyStart, header := 0, false
	for i, l := range lines {
		if l == "" || l[0] == ' ' || l[0] == '\t' {
			continue
		}
		tl := strings.TrimRight(l, " \t")
		if !strings.HasSuffix(tl, " = {") {
			continue
		}
		t := strings.TrimSpace(strings.TrimSuffix(tl, " = {"))
		// Check if this is a domain print rather than service
		if t == "system" || (strings.HasPrefix(t, "gui/") || strings.HasPrefix(t, "user/") || strings.HasPrefix(t, "login/") || strings.HasPrefix(t, "pid/")) && !strings.Contains(strings.TrimPrefix(strings.TrimPrefix(strings.TrimPrefix(strings.TrimPrefix(t, "gui/"), "user/"), "login/"), "pid/"), "/") {
			return nil, fmt.Errorf("%w: target %q is a domain, not a service", ErrMalformedLaunchctlOutput, t)
		}
		target = t
		if lastSlash := strings.LastIndex(target, "/"); lastSlash >= 0 {
			domain = target[:lastSlash]
			label = target[lastSlash+1:]
		} else {
			label = target
		}
		bodyStart, header = i+1, true
		break
	}
	if !header {
		// launchctl's refusals are short header-less texts ("Bad request." /
		// "Could not find service ..."). Matching them only here keeps a
		// loaded job whose argv or exit text happens to contain one of these
		// phrases from reading as not found.
		lower := strings.ToLower(output)
		for _, sentinel := range []string{"could not find service", "could not find domain", "service is not loaded", "no such process", "invalid domain"} {
			if strings.Contains(lower, sentinel) {
				return nil, fmt.Errorf("%w: %s", ErrLaunchctlServiceNotFound, strings.TrimSpace(output))
			}
		}
	}

	topIndent, indentKnown := "", false
	closed, inArgs := !header, false
	for _, line := range lines[bodyStart:] {
		trimmed := strings.TrimSpace(line)
		indent := line[:len(line)-len(strings.TrimLeft(line, " \t"))]
		if inArgs {
			if trimmed == "}" && indent == topIndent {
				inArgs = false
				continue
			}
			if entry, ok := strings.CutPrefix(line, topIndent+"\t"); ok {
				arguments = append(arguments, entry)
			} else {
				arguments = append(arguments, trimmed)
			}
			continue
		}
		if trimmed == "" {
			continue
		}
		if !indentKnown {
			topIndent, indentKnown = indent, true
		}
		if header && trimmed == "}" && len(indent) < len(topIndent) {
			closed = true
			break
		}
		if strings.HasPrefix(trimmed, "XPC_SERVICE_NAME => ") {
			if label == "" {
				label = strings.TrimSpace(strings.TrimPrefix(trimmed, "XPC_SERVICE_NAME => "))
			}
			continue
		}
		if indent != topIndent {
			continue
		}
		if trimmed == "arguments = {" {
			inArgs = true
			arguments = arguments[:0]
			continue
		}
		if strings.HasSuffix(trimmed, "{") || trimmed == "}" {
			continue // a nested block opener or closer at the top indent
		}
		if strings.HasPrefix(trimmed, "path = ") {
			plistPath = strings.TrimSpace(strings.TrimPrefix(trimmed, "path = "))
		} else if strings.HasPrefix(trimmed, "type = ") {
			launchdType = strings.TrimSpace(strings.TrimPrefix(trimmed, "type = "))
		} else if strings.HasPrefix(trimmed, "state = ") {
			state = strings.TrimSpace(strings.TrimPrefix(trimmed, "state = "))
		} else if strings.HasPrefix(trimmed, "job state = ") && state == "" {
			state = strings.TrimSpace(strings.TrimPrefix(trimmed, "job state = "))
		} else if strings.HasPrefix(trimmed, "program = ") {
			program = strings.TrimSpace(strings.TrimPrefix(trimmed, "program = "))
		} else if strings.HasPrefix(trimmed, "runs = ") {
			if n, ok := parseLeadingInt(strings.TrimPrefix(trimmed, "runs = ")); ok && n >= 0 {
				runs = n
			}
		} else if strings.HasPrefix(trimmed, "properties = ") {
			properties = properties[:0]
			for _, p := range strings.Split(strings.TrimPrefix(trimmed, "properties = "), "|") {
				if p = strings.TrimSpace(p); p != "" {
					properties = append(properties, p)
				}
			}
		} else if strings.HasPrefix(trimmed, "pid = ") {
			pidStr := strings.TrimSpace(strings.TrimPrefix(trimmed, "pid = "))
			val, err := strconv.Atoi(pidStr)
			if err != nil || val < 0 {
				return nil, fmt.Errorf("%w: invalid pid %q", ErrMalformedLaunchctlOutput, pidStr)
			}
			pid = val
		} else if strings.HasPrefix(trimmed, "domain = ") && domain == "" {
			d := strings.TrimSpace(strings.TrimPrefix(trimmed, "domain = "))
			if f := strings.Fields(d); len(f) > 0 {
				domain = f[0]
			}
		} else if strings.HasPrefix(trimmed, "last exit code = ") {
			// Forms: "0", "3", "1: Operation not permitted", "(never exited)".
			if c, ok := parseLeadingInt(strings.TrimPrefix(trimmed, "last exit code = ")); ok {
				lastExitCode = &c
			}
		} else if strings.HasPrefix(trimmed, "last exit reason = ") {
			exitReason = strings.TrimSpace(strings.TrimPrefix(trimmed, "last exit reason = "))
		} else if strings.HasPrefix(trimmed, "exit reason = ") && exitReason == "" {
			exitReason = strings.TrimSpace(strings.TrimPrefix(trimmed, "exit reason = "))
		} else if strings.HasPrefix(trimmed, "immediate reason = ") && exitReason == "" {
			exitReason = strings.TrimSpace(strings.TrimPrefix(trimmed, "immediate reason = "))
		} else if strings.HasPrefix(trimmed, "last terminating signal = ") {
			termSignal = strings.TrimSpace(strings.TrimPrefix(trimmed, "last terminating signal = "))
		}
	}
	if inArgs || !closed {
		return nil, fmt.Errorf("%w: truncated output (an unclosed block)", ErrMalformedLaunchctlOutput)
	}
	if termSignal != "" && lastExitCode == nil && exitReason == "" {
		exitReason = "terminated by signal: " + termSignal
	}
	if len(arguments) == 0 {
		arguments = nil
	}
	if len(properties) == 0 {
		properties = nil
	}

	if label == "" {
		return nil, fmt.Errorf("%w: could not extract service label", ErrMalformedLaunchctlOutput)
	}

	if state == "" {
		if pid > 0 {
			state = "running"
		} else {
			state = "not running"
		}
	}

	if target == "" {
		if domain != "" {
			target = domain + "/" + label
		} else {
			target = label
		}
	}

	var phase servicespec.Phase
	switch strings.ToLower(state) {
	case "running":
		phase = servicespec.PhaseReady
	case "spawn scheduled", "waiting", "waiting to run":
		phase = servicespec.PhaseStarting
	case "not running", "stopped":
		if lastExitCode != nil && *lastExitCode != 0 {
			phase = servicespec.PhaseFailed
		} else {
			phase = servicespec.PhaseStopped
		}
	default:
		if pid > 0 {
			phase = servicespec.PhaseReady
		} else if lastExitCode != nil && *lastExitCode != 0 {
			phase = servicespec.PhaseFailed
		} else {
			phase = servicespec.PhaseUnknown
		}
	}

	var lastExit *servicespec.ExitRecord
	if lastExitCode != nil {
		exitClass := servicespec.ExitClean
		if *lastExitCode != 0 {
			exitClass = servicespec.ExitCrash
		}
		if strings.Contains(strings.ToLower(exitReason), "watchdog") {
			exitClass = servicespec.ExitWatchdog
		}
		lastExit = &servicespec.ExitRecord{
			Class: exitClass,
			Code:  *lastExitCode,
		}
	}

	observed := servicespec.Observed{
		Schema: servicespec.ObservedSchemaV1,
		Identity: servicespec.Identity{
			Service:  label,
			Workload: label,
		},
		Phase:    phase,
		LastExit: lastExit,
	}

	return &LaunchdStatus{
		Target:         target,
		Label:          label,
		Domain:         domain,
		Type:           launchdType,
		State:          state,
		PID:            pid,
		PlistPath:      plistPath,
		Program:        program,
		Arguments:      arguments,
		Runs:           runs,
		Properties:     properties,
		LastExitCode:   lastExitCode,
		ExitReason:     exitReason,
		Phase:          phase,
		Observed:       observed,
		ObservedStatus: observed,
	}, nil
}

// parseLeadingInt parses the leading, optionally signed, decimal integer of s
// and ignores any trailing text, so "1: Operation not permitted" yields 1.
// It reports false when s does not start with a number, e.g. "(never exited)".
func parseLeadingInt(s string) (int, bool) {
	s = strings.TrimSpace(s)
	end := 0
	if end < len(s) && (s[end] == '-' || s[end] == '+') {
		end++
	}
	digits := end
	for end < len(s) && s[end] >= '0' && s[end] <= '9' {
		end++
	}
	if end == digits {
		return 0, false
	}
	v, err := strconv.Atoi(s[:end])
	if err != nil {
		return 0, false
	}
	return v, true
}

// ParseLaunchctlPrintDisabled reports whether label is disabled in
// `launchctl print-disabled <domain>` output. Rows look like
//
//	"com.fak.up" => enabled
//	"com.fak.up" => disabled
//
// and older macOS prints `=> false` (enabled) / `=> true` (disabled). The
// quoted label must match exactly. found is false when no row names label
// (launchd then applies the plist's own Disabled key, which defaults to
// enabled).
func ParseLaunchctlPrintDisabled(output, label string) (disabled bool, found bool) {
	for _, line := range strings.Split(output, "\n") {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, `"`) {
			continue
		}
		end := strings.Index(trimmed[1:], `"`)
		if end < 0 || trimmed[1:1+end] != label {
			continue
		}
		rest := strings.TrimSpace(trimmed[1+end+1:])
		if !strings.HasPrefix(rest, "=>") {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(strings.TrimPrefix(rest, "=>"))) {
		case "disabled", "true":
			return true, true
		case "enabled", "false":
			return false, true
		}
	}
	return false, false
}
