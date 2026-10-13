package validate

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

var validateWSLCommandSurface = []struct {
	phase    string
	commands []string
}{
	{phase: "launcher", commands: []string{"wsl.exe", "bash"}},
	{phase: "test_materialize", commands: []string{"rm", "mkdir", "tar", "go"}},
	{phase: "extract_tip", commands: []string{"rm", "mkdir", "git", "tar", "mv", "ls", "tail", "xargs"}},
	{phase: "cleanup", commands: []string{"rm"}},
	{phase: "overlay", commands: []string{"pwd", "mkdir", "cp", "rm"}},
	{phase: "go_checks", commands: []string{"go", "gofmt"}},
}

func validateWSLRequiredCommandNames() []string {
	seen := make(map[string]bool)
	for _, surface := range validateWSLCommandSurface {
		for _, command := range surface.commands {
			seen[command] = true
		}
	}
	commands := make([]string, 0, len(seen))
	for command := range seen {
		commands = append(commands, command)
	}
	sort.Strings(commands)
	return commands
}

func runValidateWSLCapabilityCommand(ctx context.Context, args ...string) ([]byte, error) {
	cmd := windowgate.CommandContext(ctx, "wsl.exe", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if exitErr, ok := err.(*exec.ExitError); ok && len(exitErr.Stderr) > 0 {
		out = append(out, exitErr.Stderr...)
	}
	return out, err
}

// validateWSLPreflightBounds is the per-attempt probe deadline: a bare wsl.exe
// launch takes about a second idle but can exceed the first bound under host
// load, so one longer retry precedes the typed unavailable verdict.
func validateWSLPreflightBounds() []time.Duration {
	first := validateWSLPreflightTimeout
	if raw := strings.TrimSpace(os.Getenv(validateWSLPreflightTimeoutEnv)); raw != "" {
		if d, err := time.ParseDuration(raw); err == nil && d > 0 {
			first = d
		}
	}
	retry := time.Duration(math.MaxInt64)
	if first <= retry/validateWSLPreflightRetryFactor {
		retry = validateWSLPreflightRetryFactor * first
	}
	return []time.Duration{first, retry}
}

func preflightValidateWSLCapabilitiesWithin(ctx context.Context) validateWSLCapabilityVerdict {
	required := validateWSLRequiredCommandNames()
	verdict := validateWSLCapabilityVerdict{Status: "ready", Required: required, Missing: []string{}}
	wslPath, err := validateWSLLookPath("wsl.exe")
	if err != nil {
		verdict.Status = "missing"
		verdict.Missing = []string{"wsl.exe"}
		verdict.Detail = "WSL capability preflight: host command wsl.exe is unavailable; repair WSL and rerun; no fallback selected"
		return verdict
	}
	validateWSLCapabilities.Lock()
	if identityKey, ok := validateWSLCapabilities.identityByLauncher[wslPath]; ok {
		if cached, found := validateWSLCapabilities.byIdentity[wslPath+"\x00"+identityKey]; found {
			validateWSLCapabilities.Unlock()
			cached.Cached = true
			cached.Required = append([]string(nil), cached.Required...)
			cached.Missing = append([]string(nil), cached.Missing...)
			cached.Attempts = append([]validateWSLProbeAttempt(nil), cached.Attempts...)
			return cached
		}
	}
	validateWSLCapabilities.Unlock()

	bounds := validateWSLPreflightBounds()
	var out []byte
	timedOut, serviceFailed := false, false
	for _, bound := range bounds {
		probeCtx, cancel := context.WithTimeout(ctx, bound)
		started := validateNow()
		// -e execs bash directly; without it wsl.exe re-parses the command line
		// through the distro's default shell, which measured 5-20s per probe
		// against under 1.5s for -e on the same host.
		out, err = validateWSLCommand(probeCtx, "-e", "bash", "-lc", validateWSLCapabilityScript(required))
		attempt := validateWSLProbeAttempt{BoundMS: bound.Milliseconds(), ElapsedMS: validateNow().Sub(started).Milliseconds(), Outcome: validateWSLProbeOK}
		timedOut = err != nil && errors.Is(probeCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil
		serviceFailed = err != nil && !timedOut && ctx.Err() == nil && validateWSLServiceFailure(out)
		cancel()
		switch {
		case timedOut:
			attempt.Outcome = validateWSLProbeTimeout
		case err != nil:
			attempt.Outcome = validateWSLProbeError
		}
		verdict.Attempts = append(verdict.Attempts, attempt)
		if !timedOut && !serviceFailed {
			break
		}
	}
	if err != nil {
		switch {
		case timedOut:
			verdict.Status = "unavailable"
			verdict.Detail = fmt.Sprintf("WSL capability preflight exceeded %s across %d attempts before workspace allocation (raise %s under host load); repair WSL and rerun; no fallback selected", bounds, len(bounds), validateWSLPreflightTimeoutEnv)
		case serviceFailed:
			verdict.Status = "unavailable"
			verdict.Detail = fmt.Sprintf("WSL capability preflight: the WSL service failed to start the distribution across %d attempts: %s; repair WSL and rerun; no fallback selected", len(verdict.Attempts), validateCommandDetail(bytes.ReplaceAll(out, []byte{0}, nil), err))
		default:
			verdict.Status = "missing"
			verdict.Missing = []string{"bash"}
			verdict.Detail = fmt.Sprintf("WSL capability preflight: bash is unavailable in the selected WSL environment: %s; repair WSL and rerun; no fallback selected", validateCommandDetail(out, err))
		}
		return verdict
	}
	identityOut, missingOut, err := parseValidateWSLCapabilityOutput(out)
	if err != nil {
		verdict.Status = "unavailable"
		verdict.Detail = fmt.Sprintf("WSL capability preflight returned an invalid verdict: %v; repair WSL and rerun; no fallback selected", err)
		return verdict
	}
	identity, identityKey := validateWSLEnvironmentIdentity(identityOut)
	verdict.Identity = identity
	cacheKey := wslPath + "\x00" + identityKey
	seen := make(map[string]bool)
	for _, line := range strings.Split(string(missingOut), "\n") {
		if raw, ok := strings.CutPrefix(strings.TrimSpace(line), validateWSLCheckUSPrefix); ok {
			if us, err := strconv.ParseInt(raw, 10, 64); err == nil && us >= 0 {
				last := &verdict.Attempts[len(verdict.Attempts)-1]
				last.CheckMS = us / 1000
				last.LaunchMS = max(last.ElapsedMS-last.CheckMS, 0)
			}
			continue
		}
		if command := strings.TrimSpace(line); command != "" && !seen[command] {
			seen[command] = true
			verdict.Missing = append(verdict.Missing, command)
		}
	}
	sort.Strings(verdict.Missing)
	if len(verdict.Missing) > 0 {
		verdict.Status = "missing"
		verdict.Detail = fmt.Sprintf("WSL capability preflight: environment %q is missing required command(s): %s; repair the selected WSL distribution and rerun; no fallback selected", verdict.Identity, strings.Join(verdict.Missing, ", "))
	}
	validateWSLCapabilities.Lock()
	validateWSLCapabilities.byIdentity[cacheKey] = verdict
	validateWSLCapabilities.identityByLauncher[wslPath] = identityKey
	validateWSLCapabilities.Unlock()
	return verdict
}

func validateWSLServiceFailure(out []byte) bool {
	return strings.Contains(strings.ReplaceAll(string(out), "\x00", ""), "Wsl/Service/")
}

const (
	validateWSLDistroPrefix  = "__FAK_VALIDATE_DISTRO__="
	validateWSLPathPrefix    = "__FAK_VALIDATE_PATH__="
	validateWSLCheckUSPrefix = "__FAK_VALIDATE_CHECK_US__="
)

func parseValidateWSLCapabilityOutput(out []byte) (identity, missing []byte, err error) {
	lines := bytes.SplitN(bytes.ReplaceAll(out, []byte("\r\n"), []byte("\n")), []byte("\n"), 3)
	if len(lines) < 2 || !bytes.HasPrefix(lines[0], []byte(validateWSLDistroPrefix)) || !bytes.HasPrefix(lines[1], []byte(validateWSLPathPrefix)) {
		return nil, nil, fmt.Errorf("missing environment identity header")
	}
	distro := bytes.TrimPrefix(lines[0], []byte(validateWSLDistroPrefix))
	path := bytes.TrimPrefix(lines[1], []byte(validateWSLPathPrefix))
	identity = append(append(append([]byte(nil), distro...), '\n'), path...)
	if len(lines) == 3 {
		missing = lines[2]
	}
	return identity, missing, nil
}

func validateWSLEnvironmentIdentity(out []byte) (display, key string) {
	parts := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)
	distro := strings.TrimSpace(parts[0])
	path := ""
	if len(parts) == 2 {
		path = strings.TrimSpace(parts[1])
	}
	digest := sha256.Sum256([]byte(path))
	return fmt.Sprintf("%s@path:%x", distro, digest[:6]), distro + "\x00" + fmt.Sprintf("%x", digest[:])
}

func validateWSLCapabilityScript(required []string) string {
	var checked []string
	gofmtCheck := ""
	for _, command := range required {
		switch command {
		case "wsl.exe":
		case "gofmt":
			gofmtCheck = "g=$(go env GOROOT 2>/dev/null) || g=''; { [ -n \"$g\" ] && [ -x \"$g/bin/gofmt\" ]; } || printf '%s\\n' gofmt; "
		default:
			checked = append(checked, posixQuote(command))
		}
	}
	return "set -u; t0=${EPOCHREALTIME:-}; printf '" + validateWSLDistroPrefix + "%s\\n" + validateWSLPathPrefix + "%s\\n' \"${WSL_DISTRO_NAME:-unknown}\" \"${PATH:-}\"; " +
		"for cmd in " + strings.Join(checked, " ") + "; do " +
		"command -v \"$cmd\" >/dev/null 2>&1 || printf '%s\\n' \"$cmd\"; done; " + gofmtCheck +
		"t1=${EPOCHREALTIME:-}; if [ -n \"$t0\" ] && [ -n \"$t1\" ]; then printf '" + validateWSLCheckUSPrefix + "%s\\n' \"$(( 10#${t1//[.,]/} - 10#${t0//[.,]/} ))\"; fi"
}
