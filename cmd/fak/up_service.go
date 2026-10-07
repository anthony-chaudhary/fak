package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/boundedlog"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
	"github.com/anthony-chaudhary/fak/internal/systemservice"
)

// `fak up off|on|status` is the first-class switch for the default-on native
// service: the per-user LaunchAgent that keeps `fak up --headless` resident
// (#13535). The service stays ON by default; `off` frees the GPU lease, the
// model residency and the port for dev work (a bench run, a real-model test)
// and records that the service is intentionally off, and `on` restores it.
//
//   - off [--for <dur>]: write the dev-off marker, `launchctl disable` (so a
//     login or a KeepAlive respawn cannot bring it back), then `launchctl
//     bootout`, and wait until the process and the port are gone. With --for a
//     detached waker runs `on` when the window lapses, unless `on` ran first.
//   - on: `launchctl enable`, `launchctl bootstrap` from the plist on disk
//     (re-bootstrapping a loaded job whose cached definition is stale), clear
//     the marker, and wait until /healthz reports ready.
//   - status: label, pid, loaded/disabled, marker, health, the GPU lease
//     holder, and whether the loaded job has drifted from its plist.
//
// launchd caches a job's definition from the last bootstrap and ignores edits
// to its plist until the next bootout + bootstrap. That is how the live service
// kept its 2m idle-exit (and reloaded the 27B every ~2.5 min) after its plist
// gained --gpu-idle-exit 0. `status` names the drift as STALE_DEFINITION and
// `on` repairs it.

const (
	upServiceDefaultLabel = "com.fak.up"
	upServiceDefaultAddr  = "127.0.0.1:8080"
	upServiceStatusSchema = "fak.up.service.status.v1"
	upDevOffMarkerSchema  = "fak.up.devoff.v1"

	upServiceVerdictOn             = "ON"
	upServiceVerdictStarting       = "STARTING"
	upServiceVerdictDown           = "DOWN"
	upServiceVerdictOff            = "OFF"
	upServiceVerdictOffLapsed      = "OFF_LAPSED"
	upServiceVerdictStopped        = "STOPPED"
	upServiceVerdictStale          = "STALE_DEFINITION"
	upServiceVerdictNotInstalled   = "NOT_INSTALLED"
	upServiceVerdictNotSupported   = "NOT_SUPPORTED"
	upServiceVerdictNotReady       = "NOT_READY"
	upServiceVerdictStillRunning   = "STILL_RUNNING"
	upServiceVerdictLaunchctlError = "LAUNCHCTL_FAILED"
	upServiceVerdictPortConflict   = "PORT_CONFLICT"
)

// upServiceVerbs are the positional sub-verbs cmdUp peels off before it parses
// turnkey flags. Without the peel `fak up status` silently booted the model.
var upServiceVerbs = []string{"off", "on", "status"}

var upServiceLabelToken = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)

func isUpServiceVerb(argv []string) bool {
	return len(argv) > 0 && slices.Contains(upServiceVerbs, argv[0])
}

// upServiceDeps is every side effect the service verbs have, so tests drive
// the whole off/on/status contract against a scripted launchd.
type upServiceDeps struct {
	goos       string
	uid        func() int
	home       func() (string, error)
	stat       func(path string) error
	stateDir   func() (string, error)
	run        func(ctx context.Context, name string, args ...string) ([]byte, error)
	httpGet    func(ctx context.Context, url string) (int, []byte, error)
	listening  func(addr string) bool
	alive      func(pid int) bool
	lease      func() upServiceLease
	now        func() time.Time
	sleep      func(time.Duration)
	executable func() (string, error)
	spawnLapse func(exe string, args []string, logPath string) (int, error)
}

func liveUpServiceDeps() upServiceDeps {
	client := &http.Client{
		Timeout:       3 * time.Second,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse },
	}
	return upServiceDeps{
		goos: runtime.GOOS,
		uid:  os.Getuid,
		home: os.UserHomeDir,
		stat: func(path string) error { _, err := os.Stat(path); return err },
		stateDir: func() (string, error) {
			if v := strings.TrimSpace(os.Getenv("FAK_UP_SERVICE_STATE_DIR")); v != "" {
				return v, nil
			}
			dir, err := os.UserConfigDir()
			if err != nil {
				return "", err
			}
			return filepath.Join(dir, "fak", "up-service"), nil
		},
		run: func(ctx context.Context, name string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, name, args...).CombinedOutput()
		},
		httpGet: func(ctx context.Context, url string) (int, []byte, error) {
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
			if err != nil {
				return 0, nil, err
			}
			resp, err := client.Do(req)
			if err != nil {
				return 0, nil, err
			}
			defer resp.Body.Close()
			body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
			return resp.StatusCode, body, err
		},
		listening: func(addr string) bool {
			conn, err := net.DialTimeout("tcp", addr, 300*time.Millisecond)
			if err != nil {
				return false
			}
			_ = conn.Close()
			return true
		},
		alive: processalive.Check,
		lease: func() upServiceLease {
			probe := gpulease.ProbeHolderProgress(gpulease.HolderProgressOptions{Window: -1})
			lease := upServiceLease{Path: gpulease.DefaultPath(), Held: probe.Held}
			if probe.Held {
				lease.PID = probe.PID
			}
			return lease
		},
		now:        time.Now,
		sleep:      time.Sleep,
		executable: os.Executable,
		spawnLapse: spawnUpServiceLapseWaker,
	}
}

// upServiceLease is the machine-wide GPU lease as the service verbs report it.
type upServiceLease struct {
	Path string `json:"path,omitempty"`
	Held bool   `json:"held"`
	PID  int    `json:"pid,omitempty"`
}

// upDevOffMarker is the persisted "intentionally off for dev work" record.
// Token binds a --for lapse waker to the `off` that spawned it, so a later
// `on` or a newer `off` turns a stale waker into a no-op.
type upDevOffMarker struct {
	Schema    string    `json:"schema"`
	Label     string    `json:"label"`
	Since     time.Time `json:"since"`
	Until     time.Time `json:"until,omitzero"`
	Token     string    `json:"token"`
	WakerPID  int       `json:"waker_pid,omitempty"`
	CreatedBy string    `json:"created_by,omitempty"`
}

func (m *upDevOffMarker) lapsed(now time.Time) bool {
	return m != nil && !m.Until.IsZero() && !now.Before(m.Until)
}

// upServiceStatus is the typed `fak up status --json` record.
type upServiceStatus struct {
	Schema        string          `json:"schema"`
	Verdict       string          `json:"verdict"`
	Label         string          `json:"label"`
	Target        string          `json:"target,omitempty"`
	PlistPath     string          `json:"plist_path,omitempty"`
	Installed     bool            `json:"installed"`
	Loaded        bool            `json:"loaded"`
	Disabled      bool            `json:"disabled"`
	State         string          `json:"state,omitempty"`
	PID           int             `json:"pid,omitempty"`
	Runs          int             `json:"runs,omitempty"`
	LastExitCode  *int            `json:"last_exit_code,omitempty"`
	LaunchdError  string          `json:"launchd_error,omitempty"`
	LoadedArgs    []string        `json:"loaded_args,omitempty"`
	PlistArgs     []string        `json:"plist_args,omitempty"`
	Stale         bool            `json:"stale"`
	DevOff        *upDevOffMarker `json:"dev_off,omitempty"`
	Addr          string          `json:"addr"`
	Listening     bool            `json:"listening"`
	HealthCode    int             `json:"health_code,omitempty"`
	HealthReady   bool            `json:"health_ready"`
	HealthStatus  string          `json:"health_status,omitempty"`
	Lease         upServiceLease  `json:"gpu_lease"`
	Detail        string          `json:"detail,omitempty"`
	Next          string          `json:"next,omitempty"`
	LaunchctlUsed []string        `json:"launchctl,omitempty"`
}

// upServiceTarget is the resolved launchd identity of the service.
type upServiceTarget struct {
	label     string
	domain    string
	target    string
	plistPath string
}

func cmdUpService(argv []string) {
	if rc := runUpService(os.Stdout, os.Stderr, argv, liveUpServiceDeps()); rc != 0 {
		os.Exit(rc)
	}
}

func printUpServiceHelp(w io.Writer) {
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Service verbs (the default-on native LaunchAgent, macOS only; Linux/Windows return NOT_SUPPORTED):")
	fmt.Fprintln(w, "  fak up status [--json]      label, pid, loaded/disabled, dev-off marker, /healthz, GPU lease holder, stale-definition drift")
	fmt.Fprintln(w, "  fak up off [--for <dur>]    stop the service for dev work: dev-off marker + bootout; without --for also disable, so it stays off across logins until `fak up on`")
	fmt.Fprintln(w, "  fak up on                   enable + bootstrap from the plist (re-bootstraps a stale loaded definition), then wait for /healthz ready")
	fmt.Fprintln(w, "  common flags: --label <label> (default com.fak.up, env FAK_UP_SERVICE_LABEL)  --plist <path>  --json")
}

// runUpService dispatches `fak up off|on|status`. Exit codes: 0 the requested
// state holds (status: ON, OFF or STARTING); 1 the verb ran but the state does
// not hold (a launchctl failure, a timeout, a stale or stopped service); 2
// usage or NOT_SUPPORTED.
func runUpService(stdout, stderr io.Writer, argv []string, deps upServiceDeps) int {
	if !isUpServiceVerb(argv) {
		fmt.Fprintf(stderr, "fak up: unknown service verb; want one of %s\n", strings.Join(upServiceVerbs, "|"))
		return 2
	}
	verb := argv[0]
	switch verb {
	case "status":
		return runUpServiceStatus(stdout, stderr, argv[1:], deps)
	case "off":
		return runUpServiceOff(stdout, stderr, argv[1:], deps)
	case "on":
		return runUpServiceOn(stdout, stderr, argv[1:], deps)
	}
	return 2
}

type upServiceCommonFlags struct {
	label  *string
	plist  *string
	asJSON *bool
}

func registerUpServiceCommonFlags(fs *flag.FlagSet) upServiceCommonFlags {
	label := strings.TrimSpace(os.Getenv("FAK_UP_SERVICE_LABEL"))
	if label == "" {
		label = upServiceDefaultLabel
	}
	return upServiceCommonFlags{
		label:  fs.String("label", label, "launchd label of the native service"),
		plist:  fs.String("plist", "", "plist path (default ~/Library/LaunchAgents/<label>.plist)"),
		asJSON: fs.Bool("json", false, "emit the typed status record as JSON"),
	}
}

// parseUpServiceFlags parses a service verb's flags. It returns ok=false with
// the exit code when the verb must stop: 0 for --help, 2 for a usage error.
func parseUpServiceFlags(fs *flag.FlagSet, stderr io.Writer, argv []string) (int, bool) {
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: fak %s [flags]\n\n", fs.Name())
		printUpServiceHelp(stderr)
		fmt.Fprintln(stderr, "\nFlags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0, false
		}
		return 2, false
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak %s: unexpected argument %q\n", fs.Name(), fs.Arg(0))
		return 2, false
	}
	return 0, true
}

// resolveUpServiceTarget projects the service through internal/systemservice
// so the domain convention matches every other fak launchd surface. The label
// is used verbatim (the projection would prefix a non-com.fak label, silently
// targeting a different job).
func resolveUpServiceTarget(label, plistOverride string, deps upServiceDeps) (upServiceTarget, error) {
	if !upServiceLabelToken.MatchString(label) {
		return upServiceTarget{}, fmt.Errorf("invalid launchd label %q", label)
	}
	home, err := deps.home()
	if err != nil || home == "" {
		return upServiceTarget{}, fmt.Errorf("cannot resolve the home directory: %v", err)
	}
	proj, err := systemservice.ProjectLaunchd(&servicespec.Spec{
		Schema:   servicespec.SchemaV1,
		Identity: servicespec.Identity{Node: "local", Service: "up", Workload: label},
		Desired:  servicespec.DesiredRunning,
		Command:  []string{"fak", "up", "--headless"},
	}, systemservice.LaunchdInput{
		Type:     systemservice.LaunchAgent,
		UID:      deps.uid(),
		PlistDir: filepath.Join(home, "Library", "LaunchAgents"),
	})
	if err != nil {
		return upServiceTarget{}, err
	}
	t := upServiceTarget{label: label, domain: proj.Domain, target: proj.Domain + "/" + label,
		plistPath: filepath.Join(home, "Library", "LaunchAgents", label+".plist")}
	if p := strings.TrimSpace(plistOverride); p != "" {
		t.plistPath = p
	}
	return t, nil
}

func upServiceNotSupported(stdout, stderr io.Writer, verb string, asJSON bool, goos string) int {
	detail := fmt.Sprintf("fak up %s manages the macOS LaunchAgent; %s has no native-service switch yet", verb, goos)
	if asJSON {
		writeUpServiceJSON(stdout, upServiceStatus{Schema: upServiceStatusSchema, Verdict: upServiceVerdictNotSupported, Detail: detail})
	} else {
		fmt.Fprintf(stderr, "fak up %s: %s: %s\n", verb, upServiceVerdictNotSupported, detail)
	}
	return 2
}

func writeUpServiceJSON(w io.Writer, st upServiceStatus) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(st)
}

// launchctl runs one launchctl verb and records it for the operator.
func (st *upServiceStatus) launchctl(ctx context.Context, deps upServiceDeps, args ...string) ([]byte, error) {
	st.LaunchctlUsed = append(st.LaunchctlUsed, "launchctl "+strings.Join(args, " "))
	return deps.run(ctx, "/bin/launchctl", args...)
}

// observeUpService fills the launchd, plist, marker, health and lease facts and
// derives the verdict. It never changes launchd state.
func observeUpService(ctx context.Context, t upServiceTarget, deps upServiceDeps) upServiceStatus {
	st := upServiceStatus{
		Schema:    upServiceStatusSchema,
		Label:     t.label,
		Target:    t.target,
		PlistPath: t.plistPath,
		Addr:      upServiceDefaultAddr,
	}
	if plistArgs, err := readUpServicePlistArgs(ctx, t.plistPath, deps); err == nil {
		st.Installed = true
		st.PlistArgs = plistArgs
		st.Addr = upServiceAddrFromArgs(plistArgs)
	} else if !errors.Is(err, os.ErrNotExist) {
		st.Installed = true
		st.Detail = "plist unreadable: " + err.Error()
	}
	st.observeLaunchd(ctx, t, deps)
	if out, err := deps.run(ctx, "/bin/launchctl", "print-disabled", t.domain); err == nil {
		st.Disabled, _ = systemservice.ParseLaunchctlPrintDisabled(string(out), t.label)
	}
	if m, err := readUpDevOffMarker(t.label, deps); err == nil {
		st.DevOff = m
	}
	st.observeHealth(ctx, deps)
	st.Lease = deps.lease()
	st.Verdict, st.Next = deriveUpServiceVerdict(&st, deps.now())
	return st
}

func (st *upServiceStatus) observeLaunchd(ctx context.Context, t upServiceTarget, deps upServiceDeps) {
	st.Loaded, st.State, st.PID, st.Runs, st.LastExitCode, st.LoadedArgs = false, "", 0, 0, nil, nil
	st.Stale, st.LaunchdError = false, ""
	out, runErr := deps.run(ctx, "/bin/launchctl", "print", t.target)
	parsed, err := systemservice.ParseLaunchctlPrint(string(out))
	if errors.Is(err, systemservice.ErrLaunchctlServiceNotFound) {
		return
	}
	if err != nil {
		// Unknown is not "not loaded": a denied or truncated print must never
		// read as STOPPED or make off skip the bootout of a running job.
		st.LaunchdError = fmt.Sprintf("launchctl print %s: %v %v %s", t.target, err, runErr, strings.TrimSpace(string(out)))
		return
	}
	st.Loaded = true
	st.State = parsed.State
	st.PID = parsed.PID
	st.Runs = parsed.Runs
	st.LastExitCode = parsed.LastExitCode
	st.LoadedArgs = parsed.Arguments
	st.Stale = len(st.PlistArgs) > 0 && len(st.LoadedArgs) > 0 && !slices.Equal(st.PlistArgs, st.LoadedArgs)
}

func (st *upServiceStatus) observeHealth(ctx context.Context, deps upServiceDeps) {
	st.Listening = deps.listening(st.Addr)
	st.HealthCode, st.HealthReady, st.HealthStatus = 0, false, ""
	if !st.Listening {
		return
	}
	code, body, err := deps.httpGet(ctx, "http://"+st.Addr+"/healthz")
	if err != nil {
		return
	}
	st.HealthCode = code
	st.HealthReady, st.HealthStatus = parseUpHealthBody(code, body)
}

// parseUpHealthBody reads the turnkey /healthz body. The turnkey server always
// answers 200 while it can respond and carries readiness in "ok" (warming_up,
// stopping, ok), so an HTTP 200 alone is not ready. A 200 without a parsable
// body is taken as ready, which keeps a non-turnkey health endpoint usable.
func parseUpHealthBody(code int, body []byte) (bool, string) {
	if code != http.StatusOK {
		return false, ""
	}
	var h struct {
		OK     *bool  `json:"ok"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal(body, &h); err != nil || h.OK == nil {
		return true, h.Status
	}
	return *h.OK, h.Status
}

func deriveUpServiceVerdict(st *upServiceStatus, now time.Time) (string, string) {
	switch {
	case st.LaunchdError != "":
		return upServiceVerdictLaunchctlError, "launchd state is unknown (" + st.LaunchdError + ")"
	case !st.Installed && !st.Loaded:
		return upServiceVerdictNotInstalled, "no plist at " + st.PlistPath + "; install the LaunchAgent first"
	case !st.Loaded && st.DevOff != nil && st.DevOff.lapsed(now):
		return upServiceVerdictOffLapsed, "the dev-off window lapsed at " + st.DevOff.Until.Format(time.RFC3339) + " but nothing restored the service; run `fak up on`"
	case !st.Loaded && st.DevOff != nil:
		if st.DevOff.Until.IsZero() {
			return upServiceVerdictOff, "intentionally off for dev work; run `fak up on` to restore"
		}
		return upServiceVerdictOff, "intentionally off until " + st.DevOff.Until.Format(time.RFC3339) + "; run `fak up on` to restore sooner"
	case !st.Loaded && st.HealthReady:
		return upServiceVerdictPortConflict, "the service is not loaded but another process answers /healthz on " + st.Addr + "; `fak up on` would fail to bind"
	case !st.Loaded:
		return upServiceVerdictStopped, "not loaded and no dev-off marker; the service is default-on, run `fak up on`"
	case st.Stale:
		return upServiceVerdictStale, "launchd is running a cached definition that differs from the plist (it ignores plist edits until bootout + bootstrap); run `fak up on` to re-bootstrap"
	case st.HealthReady && st.PID > 0:
		return upServiceVerdictOn, ""
	case st.HealthReady:
		return upServiceVerdictPortConflict, "the job has no running process, yet something else answers /healthz on " + st.Addr + "; the service cannot bind its port"
	case st.PID > 0 || strings.EqualFold(st.State, "spawn scheduled"):
		return upServiceVerdictStarting, "loaded and starting (model load); /healthz is not ready yet"
	default:
		return upServiceVerdictDown, "loaded but not running (launchd may be throttling respawns); check the service stderr log"
	}
}

func upServiceStatusExitCode(verdict string) int {
	switch verdict {
	case upServiceVerdictOn, upServiceVerdictOff, upServiceVerdictStarting:
		return 0
	case upServiceVerdictNotSupported:
		return 2
	default:
		return 1
	}
}

func runUpServiceStatus(stdout, stderr io.Writer, argv []string, deps upServiceDeps) int {
	fs := flag.NewFlagSet("up status", flag.ContinueOnError)
	common := registerUpServiceCommonFlags(fs)
	if rc, ok := parseUpServiceFlags(fs, stderr, argv); !ok {
		return rc
	}
	if deps.goos != "darwin" {
		return upServiceNotSupported(stdout, stderr, "status", *common.asJSON, deps.goos)
	}
	t, err := resolveUpServiceTarget(*common.label, *common.plist, deps)
	if err != nil {
		fmt.Fprintf(stderr, "fak up status: %v\n", err)
		return 1
	}
	ctx := context.Background()
	st := observeUpService(ctx, t, deps)
	if *common.asJSON {
		writeUpServiceJSON(stdout, st)
	} else {
		printUpServiceStatus(stdout, "status", st)
	}
	return upServiceStatusExitCode(st.Verdict)
}

func printUpServiceStatus(w io.Writer, verb string, st upServiceStatus) {
	fmt.Fprintf(w, "fak up %s: %s\n", verb, st.Verdict)
	fmt.Fprintf(w, "  service     %s (%s)\n", st.Label, st.Target)
	fmt.Fprintf(w, "  plist       %s", st.PlistPath)
	if !st.Installed {
		fmt.Fprint(w, " (missing)")
	}
	fmt.Fprintln(w)
	enabled := "enabled"
	if st.Disabled {
		enabled = "disabled"
	}
	if st.Loaded {
		fmt.Fprintf(w, "  launchd     loaded, %s, state=%s", enabled, st.State)
		if st.PID > 0 {
			fmt.Fprintf(w, " pid=%d", st.PID)
		}
		if st.Runs > 0 {
			fmt.Fprintf(w, " runs=%d", st.Runs)
		}
		if st.LastExitCode != nil {
			fmt.Fprintf(w, " last-exit=%d", *st.LastExitCode)
		}
		fmt.Fprintln(w)
	} else if st.LaunchdError != "" {
		fmt.Fprintf(w, "  launchd     unknown, %s\n", enabled)
	} else {
		fmt.Fprintf(w, "  launchd     not loaded, %s\n", enabled)
	}
	if st.Stale {
		fmt.Fprintln(w, "  definition  STALE: the loaded job's arguments differ from the plist")
		fmt.Fprintf(w, "              loaded: %s\n", strings.Join(st.LoadedArgs, " "))
		fmt.Fprintf(w, "              plist:  %s\n", strings.Join(st.PlistArgs, " "))
	}
	switch {
	case st.DevOff == nil:
		fmt.Fprintln(w, "  dev-off     none")
	case st.DevOff.Until.IsZero():
		fmt.Fprintf(w, "  dev-off     since %s (until `fak up on`)\n", st.DevOff.Since.Format(time.RFC3339))
	default:
		fmt.Fprintf(w, "  dev-off     since %s until %s", st.DevOff.Since.Format(time.RFC3339), st.DevOff.Until.Format(time.RFC3339))
		if st.DevOff.WakerPID > 0 {
			fmt.Fprintf(w, " (waker pid %d)", st.DevOff.WakerPID)
		}
		fmt.Fprintln(w)
	}
	switch {
	case !st.Listening:
		fmt.Fprintf(w, "  health      %s not listening\n", st.Addr)
	case st.HealthReady:
		fmt.Fprintf(w, "  health      http://%s/healthz %d ready", st.Addr, st.HealthCode)
		if st.HealthStatus != "" {
			fmt.Fprintf(w, " (status=%s)", st.HealthStatus)
		}
		fmt.Fprintln(w)
	default:
		fmt.Fprintf(w, "  health      http://%s/healthz %d not ready", st.Addr, st.HealthCode)
		if st.HealthStatus != "" {
			fmt.Fprintf(w, " (status=%s)", st.HealthStatus)
		}
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "  gpu lease   %s\n", describeUpServiceLease(st))
	for _, c := range st.LaunchctlUsed {
		fmt.Fprintf(w, "  ran         %s\n", c)
	}
	if st.Detail != "" {
		fmt.Fprintf(w, "  detail      %s\n", st.Detail)
	}
	if st.Next != "" {
		fmt.Fprintf(w, "  next        %s\n", st.Next)
	}
}

func describeUpServiceLease(st upServiceStatus) string {
	switch {
	case !st.Lease.Held:
		return "free"
	case st.Lease.PID > 0 && st.Lease.PID == st.PID:
		return fmt.Sprintf("held by pid %d (this service)", st.Lease.PID)
	default:
		return "held by " + gpulease.FormatHolderPID(st.Lease.PID)
	}
}

func runUpServiceOff(stdout, stderr io.Writer, argv []string, deps upServiceDeps) int {
	fs := flag.NewFlagSet("up off", flag.ContinueOnError)
	common := registerUpServiceCommonFlags(fs)
	window := fs.Duration("for", 0, "stay off for this long, then restore the service (0 = until `fak up on`)")
	wait := fs.Duration("wait", 60*time.Second, "how long to wait for the service process to exit")
	if rc, ok := parseUpServiceFlags(fs, stderr, argv); !ok {
		return rc
	}
	if *window < 0 {
		fmt.Fprintln(stderr, "fak up off: --for must be >= 0")
		return 2
	}
	if deps.goos != "darwin" {
		return upServiceNotSupported(stdout, stderr, "off", *common.asJSON, deps.goos)
	}
	t, err := resolveUpServiceTarget(*common.label, *common.plist, deps)
	if err != nil {
		fmt.Fprintf(stderr, "fak up off: %v\n", err)
		return 1
	}
	ctx := context.Background()
	st := upServiceStatus{Schema: upServiceStatusSchema, Label: t.label, Target: t.target, PlistPath: t.plistPath, Addr: upServiceDefaultAddr}
	if plistArgs, err := readUpServicePlistArgs(ctx, t.plistPath, deps); err == nil {
		st.Installed = true
		st.PlistArgs = plistArgs
		st.Addr = upServiceAddrFromArgs(plistArgs)
	} else if !errors.Is(err, os.ErrNotExist) {
		st.Installed = true
		st.Detail = "plist unreadable: " + err.Error()
	}
	st.observeLaunchd(ctx, t, deps)
	if !st.Installed && !st.Loaded && st.LaunchdError == "" {
		// A typo'd --label must not persist a disable override for a job that
		// does not exist while the real service keeps running.
		st.Verdict, st.Next = deriveUpServiceVerdict(&st, deps.now())
		return finishUpServiceVerb(stdout, stderr, "off", *common.asJSON, st, 1)
	}

	now := deps.now()
	marker := &upDevOffMarker{Schema: upDevOffMarkerSchema, Label: t.label, Since: now, Token: newUpDevOffToken(), CreatedBy: "fak up off"}
	if *window > 0 {
		marker.Until = now.Add(*window)
	}
	// The marker lands first so the intent is recorded even if launchctl fails.
	if err := writeUpDevOffMarker(marker, deps); err != nil {
		fmt.Fprintf(stderr, "fak up off: write dev-off marker: %v\n", err)
		return 1
	}
	// An indefinite off also disables the job so a login cannot bring it back
	// before `fak up on`. A timed off does not: a reboot ends the window early
	// (the service is default-on) instead of leaving it disabled with no waker.
	if *window == 0 {
		if out, err := st.launchctl(ctx, deps, "disable", t.target); err != nil {
			fmt.Fprintf(stderr, "fak up off: launchctl disable %s: %v %s (a login could restore the service)\n", t.target, err, strings.TrimSpace(string(out)))
		} else {
			st.Disabled = true
		}
	}
	stopPID := st.PID
	if st.Loaded || st.LaunchdError != "" {
		if out, err := st.launchctl(ctx, deps, "bootout", t.target); err != nil && !upLaunchctlNotLoaded(out) {
			st.observeLaunchd(ctx, t, deps)
			if st.Loaded || st.LaunchdError != "" {
				st.Verdict = upServiceVerdictLaunchctlError
				st.Detail = fmt.Sprintf("launchctl bootout %s: %v %s", t.target, err, strings.TrimSpace(string(out)))
				return finishUpServiceVerb(stdout, stderr, "off", *common.asJSON, st, 1)
			}
		}
	}
	gone := waitUpServiceGone(ctx, t, &st, stopPID, *wait, deps)
	if *window > 0 {
		if pid, err := startUpServiceLapseWaker(t, marker, deps); err != nil {
			fmt.Fprintf(stderr, "fak up off: could not start the --for waker (%v); run `fak up on` after %s\n", err, marker.Until.Format(time.RFC3339))
		} else {
			marker.WakerPID = pid
			// Record the waker only while this off's marker is still current: a
			// short window may already have lapsed and been restored.
			if cur, err := readUpDevOffMarker(t.label, deps); err == nil && cur.Token == marker.Token {
				_ = writeUpDevOffMarker(marker, deps)
			}
		}
	}
	st.DevOff = marker
	st.Lease = deps.lease()
	st.observeHealth(ctx, deps)
	if !gone {
		st.Verdict = upServiceVerdictStillRunning
		st.Detail = fmt.Sprintf("after %s launchd still has %s loaded or its process (pid %d) is alive", *wait, t.label, stopPID)
		return finishUpServiceVerb(stdout, stderr, "off", *common.asJSON, st, 1)
	}
	st.Installed = true // a job we just booted out counts as installed
	st.Verdict, st.Next = deriveUpServiceVerdict(&st, deps.now())
	if st.Listening {
		st.Detail = "the service is off, but another process still listens on " + st.Addr
	}
	return finishUpServiceVerb(stdout, stderr, "off", *common.asJSON, st, 0)
}

// waitUpServiceGone polls until launchd no longer has the job loaded and the
// stopped pid is dead. The port is reported, not waited on: an unrelated dev
// server on the same address must not make a completed off look stuck.
func waitUpServiceGone(ctx context.Context, t upServiceTarget, st *upServiceStatus, pid int, wait time.Duration, deps upServiceDeps) bool {
	deadline := deps.now().Add(wait)
	for {
		st.observeLaunchd(ctx, t, deps)
		if !st.Loaded && st.LaunchdError == "" && (pid <= 0 || !deps.alive(pid)) {
			return true
		}
		if !deps.now().Before(deadline) {
			return false
		}
		deps.sleep(500 * time.Millisecond)
	}
}

func runUpServiceOn(stdout, stderr io.Writer, argv []string, deps upServiceDeps) int {
	fs := flag.NewFlagSet("up on", flag.ContinueOnError)
	common := registerUpServiceCommonFlags(fs)
	wait := fs.Duration("wait", 10*time.Minute, "how long to wait for /healthz to report ready (the model load)")
	noWait := fs.Bool("no-wait", false, "return after bootstrap without waiting for /healthz")
	lapseToken := fs.String("lapse-token", "", "internal: run as the `off --for` waker bound to this marker token")
	lapseAt := fs.Int64("lapse-at", 0, "internal: unix time the `off --for` window lapses")
	if rc, ok := parseUpServiceFlags(fs, stderr, argv); !ok {
		return rc
	}
	if deps.goos != "darwin" {
		return upServiceNotSupported(stdout, stderr, "on", *common.asJSON, deps.goos)
	}
	t, err := resolveUpServiceTarget(*common.label, *common.plist, deps)
	if err != nil {
		fmt.Fprintf(stderr, "fak up on: %v\n", err)
		return 1
	}
	if *lapseToken != "" && !waitUpServiceLapse(t.label, *lapseToken, time.Unix(*lapseAt, 0), deps) {
		fmt.Fprintf(stdout, "fak up on: dev-off marker for %s changed before the window lapsed; nothing to restore\n", t.label)
		return 0
	}
	ctx := context.Background()
	st := upServiceStatus{Schema: upServiceStatusSchema, Label: t.label, Target: t.target, PlistPath: t.plistPath, Addr: upServiceDefaultAddr}
	plistArgs, err := readUpServicePlistArgs(ctx, t.plistPath, deps)
	if err != nil {
		st.Verdict = upServiceVerdictNotInstalled
		st.Detail = fmt.Sprintf("cannot read plist %s: %v", t.plistPath, err)
		return finishUpServiceVerb(stdout, stderr, "on", *common.asJSON, st, 1)
	}
	st.Installed = true
	st.PlistArgs = plistArgs
	st.Addr = upServiceAddrFromArgs(plistArgs)

	if *lapseToken != "" {
		// The waker re-checks right before acting, so an `off` issued while
		// it slept is never overridden.
		if m, err := readUpDevOffMarker(t.label, deps); err != nil || m.Token != *lapseToken {
			fmt.Fprintf(stdout, "fak up on: dev-off marker for %s changed; nothing to restore\n", t.label)
			return 0
		}
	}
	if out, err := st.launchctl(ctx, deps, "enable", t.target); err != nil {
		fmt.Fprintf(stderr, "fak up on: launchctl enable %s: %v %s\n", t.target, err, strings.TrimSpace(string(out)))
	}
	st.observeLaunchd(ctx, t, deps)
	if st.LaunchdError != "" {
		st.Verdict = upServiceVerdictLaunchctlError
		st.Detail = st.LaunchdError
		return finishUpServiceVerb(stdout, stderr, "on", *common.asJSON, st, 1)
	}
	if st.Loaded && st.Stale {
		// A cached definition that differs from the plist only changes on a
		// fresh bootstrap, so reload it rather than trusting the running job.
		stalePID := st.PID
		if out, err := st.launchctl(ctx, deps, "bootout", t.target); err != nil && !upLaunchctlNotLoaded(out) {
			// launchctl can report an error for a bootout that still removed
			// the job; only a job that is still loaded is a failure.
			st.observeLaunchd(ctx, t, deps)
			if st.Loaded || st.LaunchdError != "" {
				st.Verdict = upServiceVerdictLaunchctlError
				st.Detail = fmt.Sprintf("launchctl bootout %s: %v %s", t.target, err, strings.TrimSpace(string(out)))
				return finishUpServiceVerb(stdout, stderr, "on", *common.asJSON, st, 1)
			}
		}
		if !waitUpServiceGone(ctx, t, &st, stalePID, 60*time.Second, deps) {
			st.Verdict = upServiceVerdictStillRunning
			st.Detail = fmt.Sprintf("the stale %s did not unload within 60s; not re-bootstrapping over it", t.label)
			return finishUpServiceVerb(stdout, stderr, "on", *common.asJSON, st, 1)
		}
	}
	if st.Loaded && !st.Stale && st.PID <= 0 && !strings.EqualFold(st.State, "spawn scheduled") {
		// Loaded and current but down (launchd throttling after crashes):
		// kickstart restarts it now instead of waiting out the throttle.
		if out, err := st.launchctl(ctx, deps, "kickstart", "-k", t.target); err != nil {
			fmt.Fprintf(stderr, "fak up on: launchctl kickstart -k %s: %v %s\n", t.target, err, strings.TrimSpace(string(out)))
		}
	}
	if !st.Loaded {
		if out, err := st.launchctl(ctx, deps, "bootstrap", t.domain, t.plistPath); err != nil {
			st.observeLaunchd(ctx, t, deps)
			if !st.Loaded {
				st.Verdict = upServiceVerdictLaunchctlError
				st.Detail = fmt.Sprintf("launchctl bootstrap %s %s: %v %s", t.domain, t.plistPath, err, strings.TrimSpace(string(out)))
				return finishUpServiceVerb(stdout, stderr, "on", *common.asJSON, st, 1)
			}
		}
	}
	// The service is enabled and loaded again: the dev-off intent is over, and
	// any --for waker finds its token gone and exits.
	if err := removeUpDevOffMarker(t.label, deps); err != nil {
		fmt.Fprintf(stderr, "fak up on: clear dev-off marker: %v\n", err)
	}
	if *noWait {
		final := observeUpService(ctx, t, deps)
		final.LaunchctlUsed = st.LaunchctlUsed
		return finishUpServiceVerb(stdout, stderr, "on", *common.asJSON, final, upServiceStatusExitCode(final.Verdict))
	}
	final, ready := waitUpServiceReady(ctx, t, *wait, deps)
	final.LaunchctlUsed = st.LaunchctlUsed
	if !ready {
		if final.Verdict == upServiceVerdictStarting || final.Verdict == upServiceVerdictOn {
			final.Verdict = upServiceVerdictNotReady
		}
		final.Detail = fmt.Sprintf("loaded, but /healthz on %s was not ready within %s", final.Addr, *wait)
		if final.Next == "" {
			final.Next = "the model may still be loading; re-check with `fak up status`"
		}
		return finishUpServiceVerb(stdout, stderr, "on", *common.asJSON, final, 1)
	}
	return finishUpServiceVerb(stdout, stderr, "on", *common.asJSON, final, 0)
}

func waitUpServiceReady(ctx context.Context, t upServiceTarget, wait time.Duration, deps upServiceDeps) (upServiceStatus, bool) {
	deadline := deps.now().Add(wait)
	for {
		st := observeUpService(ctx, t, deps)
		if st.Verdict == upServiceVerdictOn {
			return st, true
		}
		if !deps.now().Before(deadline) {
			return st, false
		}
		deps.sleep(2 * time.Second)
	}
}

// waitUpServiceLapse is the `off --for` waker's sleep. It polls the WALL clock
// (a sleeping Mac pauses the monotonic clock) and returns true only when the
// window lapsed with the marker that spawned it still in place.
func waitUpServiceLapse(label, token string, at time.Time, deps upServiceDeps) bool {
	for {
		m, err := readUpDevOffMarker(label, deps)
		if err != nil || m.Token != token {
			return false
		}
		remaining := at.Unix() - deps.now().Unix()
		if remaining <= 0 {
			return true
		}
		step := 30 * time.Second
		if d := time.Duration(remaining) * time.Second; d < step {
			step = d
		}
		deps.sleep(step)
	}
}

func startUpServiceLapseWaker(t upServiceTarget, m *upDevOffMarker, deps upServiceDeps) (int, error) {
	exe, err := deps.executable()
	if err != nil {
		return 0, err
	}
	dir, err := deps.stateDir()
	if err != nil {
		return 0, err
	}
	args := []string{"up", "on", "--label", t.label, "--plist", t.plistPath,
		"--lapse-token", m.Token, "--lapse-at", strconv.FormatInt(m.Until.Unix(), 10)}
	return deps.spawnLapse(exe, args, filepath.Join(dir, t.label+".lapse.log"))
}

// upServiceLapseLogMaxBytes caps the `fak up off --for` waker log (16 MiB). Each
// waker spawn appends to the same per-label log; past the cap it rotates to one .1
// generation (internal/boundedlog) so repeated off/on cycles cannot grow it forever.
const upServiceLapseLogMaxBytes = boundedlog.DefaultMaxBytes

// rotateUpServiceLapseLog is the best-effort size bound the spawner applies before
// it opens the log for append: a failed rotation never blocks the waker.
func rotateUpServiceLapseLog(logPath string, maxBytes int64) {
	_, _ = boundedlog.RotateIfOver(logPath, maxBytes)
}

func finishUpServiceVerb(stdout, stderr io.Writer, verb string, asJSON bool, st upServiceStatus, rc int) int {
	if asJSON {
		writeUpServiceJSON(stdout, st)
		return rc
	}
	w := stdout
	if rc != 0 {
		w = stderr
	}
	printUpServiceStatus(w, verb, st)
	return rc
}

// upLaunchctlNotLoaded reports whether a failed bootout means "already gone".
func upLaunchctlNotLoaded(out []byte) bool {
	s := strings.ToLower(string(out))
	return strings.Contains(s, "no such process") || strings.Contains(s, "could not find") || strings.Contains(s, "not loaded")
}

// readUpServicePlistArgs reads ProgramArguments from the plist. Existence is
// an os.Stat fact (never a sniff of localized plutil text), and only the
// ProgramArguments key is converted, so a <data> or <date> value elsewhere in
// the plist cannot break the read. A plist with only Program yields [Program].
func readUpServicePlistArgs(ctx context.Context, path string, deps upServiceDeps) ([]string, error) {
	if err := deps.stat(path); err != nil {
		return nil, err
	}
	out, err := deps.run(ctx, "/usr/bin/plutil", "-extract", "ProgramArguments", "json", "-o", "-", "--", path)
	if err == nil {
		var args []string
		if err := json.Unmarshal(out, &args); err != nil {
			return nil, fmt.Errorf("plutil %s ProgramArguments: %w", path, err)
		}
		return args, nil
	}
	prog, perr := deps.run(ctx, "/usr/bin/plutil", "-extract", "Program", "raw", "-o", "-", "--", path)
	if perr == nil && strings.TrimSpace(string(prog)) != "" {
		return []string{strings.TrimSpace(string(prog))}, nil
	}
	return nil, fmt.Errorf("plutil %s: %v %s", path, err, strings.TrimSpace(string(out)))
}

// upServiceAddrFromArgs finds the --addr the service binds, or the turnkey
// default. A bare ":port" means loopback for the probe.
func upServiceAddrFromArgs(args []string) string {
	addr := ""
	for i, a := range args {
		switch {
		case a == "--addr" || a == "-addr":
			if i+1 < len(args) {
				addr = args[i+1]
			}
		case strings.HasPrefix(a, "--addr="):
			addr = strings.TrimPrefix(a, "--addr=")
		case strings.HasPrefix(a, "-addr="):
			addr = strings.TrimPrefix(a, "-addr=")
		}
	}
	if addr == "" {
		return upServiceDefaultAddr
	}
	if host, port, err := net.SplitHostPort(addr); err == nil && (host == "" || host == "0.0.0.0" || host == "::") {
		return net.JoinHostPort("127.0.0.1", port)
	}
	return addr
}

func upDevOffMarkerPath(label string, deps upServiceDeps) (string, error) {
	dir, err := deps.stateDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, label+".dev-off.json"), nil
}

func readUpDevOffMarker(label string, deps upServiceDeps) (*upDevOffMarker, error) {
	path, err := upDevOffMarkerPath(label, deps)
	if err != nil {
		return nil, err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m upDevOffMarker
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("dev-off marker %s: %w", path, err)
	}
	if m.Schema != upDevOffMarkerSchema || m.Label != label {
		return nil, fmt.Errorf("dev-off marker %s: schema %q label %q", path, m.Schema, m.Label)
	}
	return &m, nil
}

func writeUpDevOffMarker(m *upDevOffMarker, deps upServiceDeps) error {
	path, err := upDevOffMarkerPath(m.Label, deps)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return atomicWriteFile(path, append(raw, '\n'), 0o644)
}

func removeUpDevOffMarker(label string, deps upServiceDeps) error {
	path, err := upDevOffMarkerPath(label, deps)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func newUpDevOffToken() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return hex.EncodeToString(b[:])
}
