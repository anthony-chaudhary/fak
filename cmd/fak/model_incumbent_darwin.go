//go:build darwin

package main

// The Darwin half of `fak model incumbent` (#9714): read-only live observation
// for preflight and the fail-closed bootstrap for install. Every lifecycle gate
// matches the preserved campaign contract: the expected launchd job must bind the
// exact port-8090 listener PID before anything may be signaled, and a foreign
// supervisor's incumbent is reported, never touched.

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processstart"
)

type incumbentObservationDeps struct {
	lookupUID  func() (int, error)
	runLsof    func(ctx context.Context, port int) ([]byte, error)
	runLaunchd func(ctx context.Context, args ...string) ([]byte, error)
	readPS     func(ctx context.Context, pid int) (modelCanaryProcessIdentity, error)
	probeHTTP  func(ctx context.Context, url string) (int, error)
	// probeHTTPBody exists for /v1/models, whose alias inventory the verdict reads.
	probeHTTPBody func(ctx context.Context, url string) (int, []byte, error)
	now           func() time.Time
}

func liveIncumbentDeps() incumbentObservationDeps {
	client := &http.Client{
		Timeout:       5 * time.Second,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse },
	}
	probe := func(ctx context.Context, url string) (int, []byte, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return 0, nil, err
		}
		resp, err := client.Do(req)
		if err != nil {
			return 0, nil, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if err != nil {
			return resp.StatusCode, nil, err
		}
		return resp.StatusCode, body, nil
	}
	return incumbentObservationDeps{
		lookupUID: func() (int, error) { return os.Getuid(), nil },
		runLsof: func(ctx context.Context, port int) ([]byte, error) {
			// Byte-identical observation shape with the canary runtime: the same
			// argv means the same -Fpcftn record stream and the same parser
			// contract, with no P-descriptor lines.
			cmd := exec.CommandContext(ctx, "/usr/sbin/lsof", "-nP", "-iTCP:"+strconv.Itoa(port), "-sTCP:LISTEN", "-Fpcftn")
			var stdout, stderr strings.Builder
			cmd.Stdout, cmd.Stderr = &stdout, &stderr
			runErr := cmd.Run()
			exitCode := 0
			var exitErr *exec.ExitError
			if errors.As(runErr, &exitErr) {
				exitCode = exitErr.ExitCode()
			} else if runErr != nil {
				return nil, fmt.Errorf("lsof: %w", runErr)
			}
			present, classifyErr := classifyModelCanaryLsofExit([]byte(stdout.String()), []byte(stderr.String()), exitCode)
			if classifyErr != nil {
				return nil, classifyErr
			}
			if !present {
				return nil, nil
			}
			return []byte(stdout.String()), nil
		},
		runLaunchd: func(ctx context.Context, args ...string) ([]byte, error) {
			cmd := exec.CommandContext(ctx, "/bin/launchctl", args...)
			var stdout, stderr strings.Builder
			cmd.Stdout, cmd.Stderr = &stdout, &stderr
			if err := cmd.Run(); err != nil {
				return nil, fmt.Errorf("launchctl %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
			}
			return []byte(stdout.String()), nil
		},
		readPS: incumbentProcessIdentity,
		probeHTTP: func(ctx context.Context, url string) (int, error) {
			status, _, err := probe(ctx, url)
			return status, err
		},
		probeHTTPBody: probe,
		now:           time.Now,
	}
}

// incumbentProcessIdentity binds a live PID to its BSD ps command identity the
// same way the canary runtime does, including the kernel process-start
// cross-check that guards against PID reuse.
func incumbentProcessIdentity(ctx context.Context, pid int) (modelCanaryProcessIdentity, error) {
	started, ok := processstart.Start(pid)
	if !ok {
		return modelCanaryProcessIdentity{}, fmt.Errorf("kernel start identity unavailable for PID %d", pid)
	}
	raw, err := exec.CommandContext(ctx, "ps", "-ww", "-p", strconv.Itoa(pid), "-o", "pid=,lstart=,command=").Output()
	if err != nil {
		return modelCanaryProcessIdentity{}, fmt.Errorf("ps identity for PID %d: %w", pid, err)
	}
	parsed, _, err := parseModelCanaryPS(raw, time.Local)
	if err != nil {
		return modelCanaryProcessIdentity{}, err
	}
	if parsed.PID != pid {
		return modelCanaryProcessIdentity{}, errors.New("BSD ps returned the wrong PID")
	}
	parsedStart, _ := time.Parse(time.RFC3339Nano, parsed.StartedAt)
	if !parsedStart.Equal(started.Truncate(time.Second)) {
		return modelCanaryProcessIdentity{}, errors.New("BSD ps and kernel process-start identities disagree")
	}
	parsed.StartedAt = started.UTC().Format(time.RFC3339Nano)
	return parsed, nil
}

func runModelIncumbentPreflight(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model incumbent preflight", flag.ContinueOnError)
	fs.SetOutput(stderr)
	port := fs.Int("port", incumbentDefaultPort, "incumbent listener port")
	label := fs.String("label", incumbentDefaultLabel, "expected launchd service label")
	jsonOut := fs.Bool("json", false, "print the machine-readable receipt")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak model incumbent preflight: unexpected positional arguments")
		return 2
	}
	receipt, err := observeIncumbentPreflight(context.Background(), liveIncumbentDeps(), *port, *label)
	if err != nil {
		fmt.Fprintf(stderr, "fak model incumbent preflight: %v\n", err)
		return 1
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(receipt)
	} else {
		fmt.Fprint(stdout, renderIncumbentPreflightText(receipt))
	}
	return incumbentVerdictExit(receipt.Verdict)
}

func renderIncumbentPreflightText(r incumbentPreflightReceipt) string {
	var b strings.Builder
	fmt.Fprintf(&b, "MODEL INCUMBENT PREFLIGHT (%s)\n", r.Verdict)
	fmt.Fprintf(&b, "  launchd target: %s\n", r.LaunchdTarget)
	fmt.Fprintf(&b, "  expected job: present=%v pid=%d plist=%s\n", r.ExpectedOwner.JobPresent, r.ExpectedOwner.JobPID, r.ExpectedOwner.PlistPath)
	fmt.Fprintf(&b, "  listener: present=%v pid=%d port=%d\n", r.Incumbent.ListenerPresent, r.Incumbent.ListenerPID, r.ListenerPort)
	fmt.Fprintf(&b, "  command: %s matches_preserved=%v\n", r.Incumbent.CommandSHA256, r.Incumbent.CommandMatchesPreserved)
	fmt.Fprintf(&b, "  owner label: %s resolved=%v\n", r.Incumbent.OwnerLabelSHA256, r.Incumbent.OwnerResolved)
	fmt.Fprintf(&b, "  endpoints: health=%d models=%d alias=%q alias_matches=%v\n", r.Incumbent.HealthStatus, r.Incumbent.ModelsStatus, r.Incumbent.ModelAlias, r.Incumbent.AliasMatches)
	fmt.Fprintf(&b, "  verdict: %s reason: %s\n", r.Verdict, r.Reason)
	return b.String()
}

// observeIncumbentPreflight performs the read-only ownership observation and
// classifies it. Every observation failure is a typed OBSERVATION_FAILED with the
// specific reason, never an inferred owner.
func observeIncumbentPreflight(ctx context.Context, deps incumbentObservationDeps, port int, label string) (incumbentPreflightReceipt, error) {
	receipt := incumbentPreflightReceipt{
		ListenerPort: port,
		ObservedAt:   deps.now().UTC().Format(time.RFC3339),
	}
	uid, err := deps.lookupUID()
	if err != nil {
		return receipt, fmt.Errorf("resolve user domain: %w", err)
	}
	target := fmt.Sprintf("gui/%d/%s", uid, label)
	receipt.LaunchdTarget = target
	receipt.ExpectedOwner.Label = label

	lsofRaw, err := deps.runLsof(ctx, port)
	if err != nil {
		receipt.Reason = fmt.Sprintf("observe listener: %v", err)
		return classifyIncumbentPreflight(receipt), nil
	}
	if lsofRaw != nil {
		owner, err := parseModelCanaryLsof(lsofRaw, port)
		if err != nil {
			receipt.Reason = fmt.Sprintf("parse listener: %v", err)
			return classifyIncumbentPreflight(receipt), nil
		}
		receipt.Incumbent.ListenerPresent = true
		receipt.Incumbent.ListenerPID = owner.PID
		identity, err := deps.readPS(ctx, owner.PID)
		if err != nil {
			receipt.Reason = fmt.Sprintf("read listener identity: %v", err)
			return classifyIncumbentPreflight(receipt), nil
		}
		receipt.Incumbent.CommandSHA256 = identity.ArgvSHA256
	}

	launchdRaw, err := deps.runLaunchd(ctx, "print", target)
	if err != nil {
		// launchctl print reports an absent job as a nonzero exit; that is an
		// observation of absence, not a tool failure. Anything else is.
		if !isIncumbentLaunchdAbsent(err) {
			receipt.Reason = fmt.Sprintf("read expected launchd job: %v", err)
			return classifyIncumbentPreflight(receipt), nil
		}
	} else {
		pid, plistPath, err := parseDarwinModelCanaryLaunchctl(launchdRaw, target)
		if err != nil {
			receipt.ExpectedOwner.JobPresent = true
			receipt.Reason = fmt.Sprintf("parse expected launchd job: %v", err)
			return classifyIncumbentPreflight(receipt), nil
		}
		receipt.ExpectedOwner.JobPresent = true
		receipt.ExpectedOwner.JobPID = pid
		receipt.ExpectedOwner.PlistPath = plistPath
	}

	if receipt.Incumbent.ListenerPresent && !receipt.ExpectedOwner.JobPresent {
		if labelDigest, ok := scanIncumbentDomainOwner(ctx, deps, uid, receipt.Incumbent.ListenerPID); ok {
			receipt.Incumbent.OwnerLabelSHA256 = labelDigest
			receipt.Incumbent.OwnerResolved = true
		}
	}

	if receipt.Incumbent.ListenerPresent {
		healthStatus, err := deps.probeHTTP(ctx, fmt.Sprintf("http://127.0.0.1:%d/health", port))
		if err != nil {
			receipt.Reason = fmt.Sprintf("probe /health: %v", err)
			return classifyIncumbentPreflight(receipt), nil
		}
		receipt.Incumbent.HealthStatus = healthStatus
		models, err := probeIncumbentModels(ctx, deps, port)
		if err != nil {
			receipt.Reason = fmt.Sprintf("probe /v1/models: %v", err)
			return classifyIncumbentPreflight(receipt), nil
		}
		receipt.Incumbent.ModelsStatus = models.status
		receipt.Incumbent.ModelAlias = models.firstAlias
	}
	return classifyIncumbentPreflight(receipt), nil
}

type incumbentModelsProbe struct {
	status     int
	firstAlias string
}

func probeIncumbentModels(ctx context.Context, deps incumbentObservationDeps, port int) (incumbentModelsProbe, error) {
	probe := incumbentModelsProbe{}
	status, body, err := deps.probeHTTPBody(ctx, fmt.Sprintf("http://127.0.0.1:%d/v1/models", port))
	if err != nil {
		return probe, err
	}
	probe.status = status
	if status == 200 {
		var payload struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			return probe, fmt.Errorf("decode /v1/models inventory: %w", err)
		}
		if len(payload.Data) > 0 {
			probe.firstAlias = payload.Data[0].ID
		}
	}
	return probe, nil
}

// isIncumbentLaunchdAbsent recognizes launchctl's absent-job shapes only: the
// "Could not find service ... in domain" diagnostic on a nonzero exit (observed
// as exit 1 and exit 113 on current macOS). Any other failure stays a tool error.
func isIncumbentLaunchdAbsent(err error) bool {
	if err == nil {
		return false
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() == 0 {
		return false
	}
	return strings.Contains(err.Error(), "Could not find service")
}

// scanIncumbentDomainOwner resolves which user-domain service owns a PID by
// scanning `launchctl print gui/<uid>`'s services table (rows of
// "pid state label"). The public-safe owner label digest hashes the raw label
// string, matching the preserved drill receipts.
func scanIncumbentDomainOwner(ctx context.Context, deps incumbentObservationDeps, uid, pid int) (string, bool) {
	raw, err := deps.runLaunchd(ctx, "print", fmt.Sprintf("gui/%d", uid))
	if err != nil {
		return "", false
	}
	label, ok := parseIncumbentDomainOwner(raw, pid)
	if !ok {
		return "", false
	}
	return digestBytes([]byte(label)), true
}

// parseIncumbentDomainOwner scans launchctl domain output for the service row
// owning pid. It requires exactly one owning row; ambiguity or absence returns
// false and the receipt reports owner_resolved=false.
func parseIncumbentDomainOwner(raw []byte, pid int) (string, bool) {
	if pid <= 0 {
		return "", false
	}
	matches := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(strings.TrimLeft(line, "\t"))
		if len(fields) < 2 {
			continue
		}
		rowPID, err := strconv.Atoi(fields[0])
		if err != nil || rowPID != pid {
			continue
		}
		label := fields[len(fields)-1]
		if label == "" || strings.Contains(label, "{") {
			continue
		}
		matches[label] = true
	}
	if len(matches) != 1 {
		return "", false
	}
	for label := range matches {
		return label, true
	}
	return "", false
}

func runModelIncumbentInstall(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model incumbent install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	plist := fs.String("plist", "", "rendered definition to bootstrap (from `fak model incumbent render`)")
	execute := fs.Bool("execute", false, "perform the launchctl bootstrap; the default is a dry-run plan")
	readyTimeout := fs.Duration("ready-timeout", 10*time.Minute, "how long to wait for the bootstrapped incumbent to pass /health and /v1/models")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	outcome, err := installIncumbent(context.Background(), liveIncumbentDeps(), incumbentInstallInput{
		Plist:        *plist,
		Execute:      *execute,
		ReadyTimeout: *readyTimeout,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak model incumbent install: %v\n", err)
		return 2
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(outcome)
	if !outcome.Admitted {
		return 2
	}
	return 0
}

type incumbentInstallInput struct {
	Plist        string
	Execute      bool
	ReadyTimeout time.Duration
}

type incumbentInstallOutcome struct {
	Schema         string   `json:"schema"`
	Issue          int      `json:"issue"`
	Executed       bool     `json:"executed"`
	Admitted       bool     `json:"admitted"`
	Plist          string   `json:"plist"`
	Label          string   `json:"label"`
	Domain         string   `json:"domain"`
	Plan           []string `json:"plan"`
	Refusal        string   `json:"refusal,omitempty"`
	ReadyEndpoints []string `json:"ready_endpoints,omitempty"`
}

// installIncumbent bootstraps a rendered definition into the user domain,
// refusing closed on every state the campaign contract refuses: an existing job,
// a foreign supervisor's incumbent on the port, or a definition whose label does
// not match the expected service. Migration away from a proven alternate owner
// remains the operator's explicit act outside this verb.
func installIncumbent(ctx context.Context, deps incumbentObservationDeps, input incumbentInstallInput) (incumbentInstallOutcome, error) {
	outcome := incumbentInstallOutcome{
		Schema:   incumbentInstallSchema,
		Issue:    incumbentOwnerIssueNumber,
		Executed: input.Execute,
	}
	plist := strings.TrimSpace(input.Plist)
	if plist == "" || !isDarwinAbsolutePath(plist) {
		return outcome, fmt.Errorf("--plist must be an absolute path to a rendered definition (got %q)", plist)
	}
	rawPlist, err := os.ReadFile(plist)
	if err != nil {
		return outcome, fmt.Errorf("read --plist: %w", err)
	}
	label, err := parseIncumbentPlistLabel(rawPlist)
	if err != nil {
		return outcome, fmt.Errorf("parse --plist: %w", err)
	}
	if label != incumbentDefaultLabel {
		return outcome, fmt.Errorf("refusing to bootstrap label %q; the campaign contract requires %s", label, incumbentDefaultLabel)
	}
	uid, err := deps.lookupUID()
	if err != nil {
		return outcome, fmt.Errorf("resolve user domain: %w", err)
	}
	domain := fmt.Sprintf("gui/%d", uid)
	target := fmt.Sprintf("%s/%s", domain, label)
	outcome.Plist = plist
	outcome.Label = label
	outcome.Domain = domain
	outcome.Plan = []string{
		"launchctl print " + target,
		"launchctl bootstrap " + domain + " " + plist,
	}
	// The ownership preflight gates every install, dry-run or execute: it binds
	// the expected job to the listener before any launchd mutation is planned.
	preflight, err := observeIncumbentPreflight(ctx, deps, incumbentDefaultPort, label)
	if err != nil {
		return outcome, err
	}
	switch preflight.Verdict {
	case incumbentVerdictOwned:
		outcome.Refusal = "expected job already owns the incumbent; nothing to install"
		return outcome, nil
	case incumbentVerdictJobAbsent:
		if preflight.Incumbent.ListenerPresent {
			owner := "unresolved"
			if preflight.Incumbent.OwnerResolved {
				owner = preflight.Incumbent.OwnerLabelSHA256
			}
			outcome.Refusal = fmt.Sprintf("alternate_launchd_supervisor_owns_incumbent: healthy listener PID %d is not owned by %s (owner label digest %s); migration is an explicit operator act outside this verb", preflight.Incumbent.ListenerPID, label, owner)
			return outcome, nil
		}
	default:
		outcome.Refusal = fmt.Sprintf("%s: %s", preflight.Verdict, preflight.Reason)
		return outcome, nil
	}
	if !input.Execute {
		outcome.Admitted = true
		outcome.Refusal = "dry-run plan only; pass --execute to bootstrap"
		return outcome, nil
	}
	if _, err := deps.runLaunchd(ctx, "bootstrap", domain, plist); err != nil {
		return outcome, fmt.Errorf("launchctl bootstrap: %w", err)
	}
	readyCtx, cancel := context.WithTimeout(ctx, input.ReadyTimeout)
	defer cancel()
	if err := waitIncumbentReady(readyCtx, deps, label); err != nil {
		return outcome, err
	}
	outcome.Admitted = true
	outcome.ReadyEndpoints = []string{
		fmt.Sprintf("http://127.0.0.1:%d/health", incumbentDefaultPort),
		fmt.Sprintf("http://127.0.0.1:%d/v1/models", incumbentDefaultPort),
	}
	return outcome, nil
}

// waitIncumbentReady polls until the bootstrapped job exists, binds the exact
// listener, and passes both endpoints. It reuses the full preflight classifier so
// readiness carries the same typed ownership semantics, not just an HTTP 200.
func waitIncumbentReady(ctx context.Context, deps incumbentObservationDeps, label string) error {
	for {
		preflight, err := observeIncumbentPreflight(ctx, deps, incumbentDefaultPort, label)
		if err == nil && preflight.Verdict == incumbentVerdictOwned {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("bootstrapped incumbent failed to become %s before the deadline: %w", incumbentVerdictOwned, ctx.Err())
		case <-time.After(500 * time.Millisecond):
		}
	}
}

// parseIncumbentPlistLabel extracts the Label from a rendered plist. It reads the
// plain-XML text the renderer emits; anything else is refused rather than parsed
// with a full plist engine.
func parseIncumbentPlistLabel(raw []byte) (string, error) {
	const key = "<key>Label</key>"
	text := string(raw)
	index := strings.Index(text, key)
	if index < 0 {
		return "", errors.New("plist has no <key>Label</key>")
	}
	rest := text[index+len(key):]
	open := strings.Index(rest, "<string>")
	close := strings.Index(rest, "</string>")
	if open < 0 || close < open {
		return "", errors.New("plist Label value is malformed")
	}
	label := strings.TrimSpace(rest[open+len("<string>") : close])
	if label == "" {
		return "", errors.New("plist Label value is empty")
	}
	return label, nil
}
