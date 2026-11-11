package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/opencoderesume"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

type opencodeLaunchOptions struct {
	guard           bool
	dryRun          bool
	probePrompt     string
	splitMode       string
	splitWhere      string
	splitInterval   time.Duration
	policyPath      string
	apiKeyEnv       string
	baseURL         string
	remoteServe     string
	model           string
	auditPath       string
	noAudit         bool
	quiet           bool
	localAuto       bool
	halo            bool
	metal           bool
	ggufPath        string
	gpuBackend      string
	tokenizerPath   string
	auto            bool
	skipPermissions bool
	workspaceDir    string
	// continueFlag / sessionID are the operator-named resume passthrough (#14330):
	// they map onto OpenCode's NATIVE --continue / --session <id> flags, which both
	// the TUI root command and `run` accept. sessionID is shape-validated by
	// opencoderesume.IsSessionID before launch — an ill-shaped id would make
	// `opencode run --session` CREATE a fresh session instead of resuming one.
	continueFlag bool
	sessionID    string
	passthrough  []string
}

var opencodeLaunchRun = execOpencodeLaunchChild

// opencodeLauncherVerify is the opencoderesume.Verify seam the `fak opencode resume`
// preflight gate calls; tests replace it to pin verdicts without spawning opencode.
var opencodeLauncherVerify = opencoderesume.Verify

// opencodeLauncherAppendJournal is the sessionjournal.Append seam the launcher writes its
// lifecycle rows through; tests replace it to capture rows or force failures.
var opencodeLauncherAppendJournal = sessionjournal.Append

const opencodeLaunchWorkingDirEnv = "FAK_OPENCODE_LAUNCH_WORKING_DIR"

// opencodeLaunchCaptureLimit bounds how much child stdout the launcher retains for the
// post-exit session-id extraction (#14330): enough for the JSONL head where OpenCode
// announces its session, without pinning megabytes per launch.
const opencodeLaunchCaptureLimit = 1 << 20

// opencodeLaunchCapture is a bounded tee of the launched child's stdout. The terminal
// stream is untouched (the tee only MIRRORS bytes); the retained prefix exists so the
// open/close journal pair can be joined to the real OpenCode session id after exit.
type opencodeLaunchCapture struct {
	mu     sync.Mutex
	buf    bytes.Buffer
	capped bool
}

func (c *opencodeLaunchCapture) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.capped {
		room := opencodeLaunchCaptureLimit - c.buf.Len()
		if len(p) <= room {
			c.buf.Write(p)
		} else {
			c.buf.Write(p[:room])
			c.capped = true
		}
	}
	return len(p), nil
}

func (c *opencodeLaunchCapture) String() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.buf.String()
}

var (
	opencodeLaunchCaptureMu   sync.Mutex
	opencodeLaunchCaptureSlot *opencodeLaunchCapture
)

// publishOpencodeLaunchCapture records the capture that the in-flight launch tees into.
// One CLI process launches exactly one child at a time, so a single slot suffices.
func publishOpencodeLaunchCapture(c *opencodeLaunchCapture) {
	opencodeLaunchCaptureMu.Lock()
	opencodeLaunchCaptureSlot = c
	opencodeLaunchCaptureMu.Unlock()
}

// takeOpencodeLaunchCapture returns the last launch's captured child stdout and clears
// the slot. It yields "" when the launch went through a stubbed opencodeLaunchRun.
func takeOpencodeLaunchCapture() string {
	opencodeLaunchCaptureMu.Lock()
	defer opencodeLaunchCaptureMu.Unlock()
	s := ""
	if opencodeLaunchCaptureSlot != nil {
		s = opencodeLaunchCaptureSlot.String()
	}
	opencodeLaunchCaptureSlot = nil
	return s
}

// opencodeResumePassthrough is the launcher-owned slice of the child argv that resumes an
// existing OpenCode session (#14330): the native --continue / --session <id> flags, placed
// right after the command verb (`run`) or at the front of an interactive launch, ahead of
// any operator passthrough. The id was already shape-validated by the caller.
func opencodeResumePassthrough(o opencodeLaunchOptions) []string {
	var args []string
	if o.continueFlag {
		args = append(args, "--continue")
	}
	if o.sessionID != "" {
		args = append(args, "--session", o.sessionID)
	}
	return args
}

// opencodeLaunchJournalEvent builds one crash-journal lifecycle row for an opencode launch
// (#14330). The open row carries CWD + Argv (the fold key `fak session recover`'s journal
// crash merge joins on); the close row carries only Reason (the exit status) — matching
// the row shapes `fak sessionjournal` itself writes.
func opencodeLaunchJournalEvent(kind sessionjournal.Kind, id, cwd string, argv []string, reason string) sessionjournal.Event {
	now := time.Now().UTC()
	bt, _ := sessionjournal.BootTime(now)
	ev := sessionjournal.Event{
		Kind:  kind,
		ID:    id,
		TS:    now.Format(time.RFC3339),
		Boot:  sessionjournal.BootID(bt),
		PID:   os.Getpid(),
		Host:  sessionJournalHost(),
		Agent: "opencode",
	}
	if kind == sessionjournal.KindOpen {
		ev.CWD = cwd
		ev.Argv = argv
	} else {
		ev.Reason = reason
	}
	return ev
}

// opencodeLauncherRecordJournal appends one lifecycle row best-effort: a write failure logs
// to stderr and returns false — journaling must never fail a launch (#14330).
func opencodeLauncherRecordJournal(stderr io.Writer, ev sessionjournal.Event) bool {
	if err := opencodeLauncherAppendJournal("", ev); err != nil {
		fmt.Fprintf(stderr, "fak opencode: session journal: %v\n", err)
		return false
	}
	return true
}

func cmdOpencode(argv []string) {
	os.Exit(runOpencode(os.Stdout, os.Stderr, argv))
}

func runOpencode(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "config" {
		return runOpencodeConfig(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "free-models" {
		return runOpencodeFreeModels(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "resume" {
		return runOpencodeResume(stdout, stderr, argv[1:])
	}

	fs := flag.NewFlagSet("opencode", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "opencode")
	guard := fs.Bool("guard", false, "launch OpenCode through fak guard for this session")
	dryRun := fs.Bool("dry-run", false, "print the OpenCode command and exit without launching")
	probePrompt := fs.String("probe", "", "run a single headless probe turn with this prompt and exit")
	skipPermissions := fs.Bool("skip-permissions", true, "pass --auto to the OpenCode child when running unattended")
	_ = fs.Bool("pure", false, "accepted for compatibility; OpenCode v2 run has no --pure, so nothing is forwarded")
	auto := fs.Bool("auto", false, "pass --auto to opencode child for non-interactive execution")
	splitMode := fs.String("split", "auto", "open the 20% fak-info pane when possible: auto|on|off")
	splitWhere := fs.String("split-where", "bottom", "with --split: place the 20% fak-info pane as a bottom strip or right column")
	splitInterval := fs.Duration("split-interval", 2*time.Second, "with --split: fak-info refresh interval")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: guard's embedded floor)")
	workDir := fs.String("dir", ".", "workspace directory to run guard and OpenCode in")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the upstream OpenAI API key (default: OPENAI_API_KEY)")
	baseURL := fs.String("base-url", "", "upstream provider base URL; advanced override passed to fak guard")
	remoteServe := fs.String("remote-serve", "", "send inference to a remote fak serve (HOST or HOST:PORT), while this local guard adjudicates")
	model := fs.String("model", "", "upstream model id override passed to fak guard")
	auditPath := fs.String("audit", "", "write guard's decision journal to this file (or 'off')")
	noAudit := fs.Bool("no-audit", false, "disable guard's decision journal")
	quiet := fs.Bool("quiet", false, "suppress guard's startup banner and exit summary")
	localAuto := fs.Bool("local", false, "auto-detect a local OpenAI-compatible model server for guard's upstream")
	halo := fs.Bool("halo", false, "target local AMD Strix Halo appliance server (http://127.0.0.1:8080/v1)")
	strix := fs.Bool("strix", false, "alias for --halo")
	metal := fs.Bool("metal", false, "with --gguf: require Apple Silicon Metal GPU acceleration")
	ggufPath := fs.String("gguf", "", "run a local in-kernel GGUF model as guard's upstream")
	gpuBackend := fs.String("backend", "", "with --gguf: compute backend")
	tokenizerPath := fs.String("tokenizer", "", "with --gguf: tokenizer override")
	continueFlag := fs.Bool("continue", false, "pass --continue to the OpenCode child (resume its most recent session)")
	sessionID := fs.String("session", "", "OpenCode session id to resume; passed to the child as --session <id>")
	resumeID := fs.String("resume", "", "alias for --session")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak opencode [launcher flags] [-- <opencode args...>]")
		fmt.Fprintln(stderr, "       fak opencode config [--write] [--addr ADDR] [--model MODEL]")
		fmt.Fprintln(stderr, "       fak opencode resume --session <id> [--prompt PROMPT] [--guard] [--dir DIR]")
		fmt.Fprintln(stderr, "  e.g. fak opencode")
		fmt.Fprintln(stderr, "       fak opencode --dry-run")
		fmt.Fprintln(stderr, "       fak opencode --probe \"Use bash to print hello\"")
		fmt.Fprintln(stderr, "       fak opencode --policy my-floor.json -- run \"check the repo\"")
		fmt.Fprintln(stderr, "       fak opencode --session ses_e3861b794ffeKK5UOg25IFE1mR")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}
	if !parseFlags(fs, argv) {
		return 2
	}
	if err := validateOpencodeLaunchSplit(*splitMode, *splitWhere); err != nil {
		fmt.Fprintf(stderr, "fak opencode: %v\n", err)
		return 2
	}
	// Resume passthrough validation (#14330): --resume is an explicit alias of --session,
	// and an ill-shaped id is refused BEFORE any exec — `opencode run --session` CREATES
	// the session when the id does not exist, so garbage in would fabricate a fresh empty
	// session instead of failing. A well-shaped id is passed through natively: the operator
	// is explicitly naming their own session, so the launcher does not re-verify existence
	// here (the `fak opencode resume` subcommand is the verified one-touch path).
	session := strings.TrimSpace(*sessionID)
	resumed := strings.TrimSpace(*resumeID)
	switch {
	case session != "" && resumed != "" && session != resumed:
		fmt.Fprintf(stderr, "fak opencode: --session %q and --resume %q disagree; pass exactly one\n", session, resumed)
		return 2
	case resumed != "":
		session = resumed
	}
	if session != "" && !opencoderesume.IsSessionID(session) {
		fmt.Fprintf(stderr, "fak opencode: --session %q is not a valid OpenCode session id (want ses_ + 16-64 alphanumerics); refusing to launch — an ill-shaped id would CREATE a new opencode session instead of resuming\n", session)
		return 2
	}
	*workDir = pathutil.ExpandTilde(*workDir)
	resolvedWorkDir, err := resolveOpencodeWorkspaceDir(*workDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak opencode: --dir: %v\n", err)
		return 2
	}

	if *halo || *strix {
		if *baseURL == "" {
			host := os.Getenv("FAK_HALO_HOST")
			if host == "" {
				host = os.Getenv("FAK_STRIX_HOST")
			}
			if host == "" {
				host = "127.0.0.1:8080"
			}
			if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
				host = "http://" + host
			}
			if !strings.HasSuffix(host, "/v1") {
				host = strings.TrimSuffix(host, "/") + "/v1"
			}
			*baseURL = host
		}
		if *model == "" {
			*model = projectassets.ResolveDynamicHaloModel(resolvedWorkDir)
		}
		if !*quiet {
			fmt.Fprintf(stderr, "fak opencode: targeting local Halo server at %s (model: %s)\n", *baseURL, *model)
		}
	}
	*ggufPath = pathutil.ExpandTilde(*ggufPath)
	*tokenizerPath = pathutil.ExpandTilde(*tokenizerPath)

	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		if *metal || (*ggufPath != "" && *gpuBackend == "") {
			// Apple-Silicon Metal is the CPU-session forward (`fak guard --metal`), NOT a
			// compute --backend: leave gpuBackend empty so guard resolves the session seam.
			*metal = true
		}
	}
	if *baseURL == "" && *remoteServe == "" && *ggufPath == "" && *apiKeyEnv == "" && !*localAuto {
		if os.Getenv("OPENAI_API_KEY") == "" {
			if base, modelID, label, found := guardDetectLocalBackend(); found {
				*localAuto = true
				if !*guard {
					*baseURL = base
				}
				if *model == "" {
					*model = modelID
				}
				if !*quiet {
					fmt.Fprintf(stderr, "fak opencode: auto-connected to local %s at %s (model: %s) (one-touch)\n", label, base, modelID)
				}
			} else {
				// On Apple Silicon macOS, if no local server is running and no API key is set, assume gguf default with Metal!
				if runtime.GOOS == "darwin" {
					*ggufPath = "default"
					if *model == "" {
						*model = "qwen38:27b"
					}
					if runtime.GOARCH == "arm64" {
						// Session seam, not a compute --backend (one-touch must not pass
						// `--backend metal`, which is unregistered on this host).
						*metal = true
					}
					if !*quiet {
						fmt.Fprintln(stderr, "fak opencode: no local model server running — assuming in-kernel model (qwen38:27b) with Metal GPU acceleration (one-touch)")
					}
				} else {
					if *model == "" {
						*model = projectassets.ResolveDynamicHaloModel(resolvedWorkDir)
					}
				}
			}
		}
	}

	fakBin := tuiExecutable()
	launch := opencodeLaunchOptions{
		guard:           *guard,
		dryRun:          *dryRun,
		probePrompt:     *probePrompt,
		splitMode:       *splitMode,
		splitWhere:      *splitWhere,
		splitInterval:   *splitInterval,
		policyPath:      *policyPath,
		apiKeyEnv:       *apiKeyEnv,
		baseURL:         *baseURL,
		remoteServe:     *remoteServe,
		model:           *model,
		auditPath:       *auditPath,
		noAudit:         *noAudit,
		quiet:           *quiet,
		localAuto:       *localAuto,
		halo:            *halo || *strix,
		metal:           *metal,
		ggufPath:        *ggufPath,
		gpuBackend:      *gpuBackend,
		tokenizerPath:   *tokenizerPath,
		auto:            *auto,
		skipPermissions: (*guard || flagSet(fs, "skip-permissions")) && *skipPermissions,
		continueFlag:    *continueFlag,
		sessionID:       session,
		passthrough:     fs.Args(),
	}
	launch.workspaceDir = resolvedWorkDir
	launchEnv := opencodeUpsertEnv(os.Environ(), opencodeLaunchWorkingDirEnv, resolvedWorkDir)
	if launch.guard {
		var err error
		launch, launchEnv, err = prepareOpencodeRouterCredential(launch, launchEnv)
		if err != nil {
			fmt.Fprintf(stderr, "fak opencode: %v\n", err)
			return 2
		}
	}
	argvOut := buildOpencodeLaunchArgv(fakBin, launch)
	if !launch.guard {
		argvOut = buildOpencodeDirectArgv(launch)
		directBase := launch.baseURL
		if directBase == "" && launch.remoteServe != "" {
			resolved, err := resolveGuardRemoteServe(launch.remoteServe)
			if err != nil {
				fmt.Fprintf(stderr, "fak opencode: --remote-serve: %v\n", err)
				return 2
			}
			directBase = strings.TrimRight(resolved, "/") + "/v1"
		}
		config, err := projectassets.GenerateOpenCodeConfig(directBase, launch.model)
		if err == nil {
			// Launch-scoped X-Fak-Session-Id via the provider's options.headers.
			config, err = openCodeConfigWithSessionHeader(config, "fak", newLaunchSessionID("opencode"))
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak opencode: session config: %v\n", err)
			return 1
		}
		launchEnv = append(launchEnv, "OPENCODE_CONFIG_CONTENT="+string(config))
	}

	if launch.dryRun {
		fmt.Fprintln(stderr, "fak opencode: dry-run - not launching")
		if launch.guard {
			fmt.Fprintln(stderr, "  view        = agent 80% / fak info 20% (--split "+launch.splitMode+")")
		} else {
			fmt.Fprintln(stderr, "  guard       = off (pass --guard for kernel adjudication)")
		}
		fmt.Fprintln(stderr, "  provider    = openai (Chat Completions /v1/chat/completions)")
		fmt.Fprintln(stderr, "  command     = "+strings.Join(argvOut, " "))
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}

	if launch.guard {
		// Read project assets for launch policy only; a guarded launch must not
		// materialize or update repository adapter assets. Explicit writers remain
		// `fak project-assets sync` and the OpenCode config verbs.
		if _, err := projectassets.Build(launch.workspaceDir, false); err != nil && !launch.quiet {
			fmt.Fprintf(stderr, "fak opencode: warning: %v\n", err)
		}
	}
	if err := projectassets.VerifyOpenCodeSnapshot(launch.workspaceDir); err != nil && !launch.quiet {
		fmt.Fprintf(stderr, "fak opencode: warning: %v\n", err)
	}

	// Crash-journal registration (#14330): the open row lands BEFORE the child execs so a
	// machine-wide crash leaves a foldable record for `fak session recover`'s journal crash
	// merge. The id is the operator-named session when known; otherwise the launch-scoped
	// fak id — the real OpenCode session id only becomes observable AFTER exit (the child
	// announces it on stdout), so it cannot key the pre-exec open row and is surfaced below
	// instead. Best-effort by contract: a journal write failure never fails the launch.
	journalID := launch.sessionID
	if journalID == "" {
		journalID = newLaunchSessionID("opencode")
	}
	if journalID != "" {
		opencodeLauncherRecordJournal(stderr, opencodeLaunchJournalEvent(sessionjournal.KindOpen, journalID, launch.workspaceDir, argvOut, ""))
	}

	started := time.Now()
	if launch.guard {
		fmt.Fprintln(stderr, "fak opencode: launching OpenCode through fak guard ...")
	} else {
		fmt.Fprintln(stderr, "fak opencode: launching OpenCode directly (without guard) ...")
	}
	code := opencodeLaunchRun(stdout, stderr, argvOut, launchEnv)
	if journalID != "" {
		opencodeLauncherRecordJournal(stderr, opencodeLaunchJournalEvent(sessionjournal.KindClose, journalID, "", nil, fmt.Sprintf("exit %d", code)))
	}
	if launch.sessionID == "" {
		// Post-exit identity join: extract the session id the child announced (when the
		// launcher did not already know it) so operators and future recovery tooling can
		// join the fak launch id to the real OpenCode session.
		if id := extractOpenCodeSessionID(takeOpencodeLaunchCapture()); id != "" && opencoderesume.IsSessionID(id) && !launch.quiet {
			fmt.Fprintf(stderr, "fak opencode: observed OpenCode session %s\n", id)
		}
	}
	if code == 0 {
		fmt.Fprintf(stderr, "fak opencode: OpenCode completed successfully in %s\n", time.Since(started).Round(time.Millisecond))
	}
	return code
}

// runOpencodeResume is `fak opencode resume` — the operator-facing one-touch exact-resume
// (#14330). Unlike the plain launcher flags (which trust an operator-named id), this path
// verifies the ses_ identity against the project-scoped OpenCode store FIRST and launches
// ONLY on VerdictResumable: `opencode run --session <id>` CREATES the session when the id
// does not exist, so every other verdict refuses with exit 2 rather than fabricating a
// fresh empty session and calling it recovery. On success it execs the headless exact-resume
// argv (`opencode run --session <id> [prompt]`, internal/opencoderesume.ResumeArgv), fronted
// by `fak guard --` exactly like the launcher's guarded child boundary when --guard is set.
func runOpencodeResume(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("opencode resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sessionID := fs.String("session", "", "OpenCode session id to resume (required)")
	prompt := fs.String("prompt", "", "headless prompt sent to the resumed session")
	guard := fs.Bool("guard", false, "front the resume launch with fak guard for kernel adjudication")
	dir := fs.String("dir", ".", "project directory the session must belong to (default: current directory)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak opencode resume --session <id> [--prompt PROMPT] [--guard] [--dir DIR]")
		fmt.Fprintln(stderr, "  e.g. fak opencode resume --session ses_e3861b794ffeKK5UOg25IFE1mR")
		fmt.Fprintln(stderr, "       fak opencode resume --session ses_... --prompt \"continue\" --guard")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}
	if !parseFlags(fs, argv) {
		return 2
	}
	id := strings.TrimSpace(*sessionID)
	if id == "" {
		fmt.Fprintln(stderr, "fak opencode resume: --session is required")
		return 2
	}
	if !opencoderesume.IsSessionID(id) {
		fmt.Fprintf(stderr, "fak opencode resume: --session %q is not a valid OpenCode session id (want ses_ + 16-64 alphanumerics); refusing to launch — an ill-shaped id would CREATE a new opencode session instead of resuming\n", id)
		return 2
	}
	resolvedDir, err := resolveOpencodeWorkspaceDir(*dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak opencode resume: --dir: %v\n", err)
		return 2
	}
	verdict, verr := opencodeLauncherVerify(context.Background(), opencoderesume.Options{SessionID: id, Dir: resolvedDir})
	if verdict != opencoderesume.VerdictResumable {
		fmt.Fprintf(stderr, "fak opencode resume: not resumable in %s: %s\n", resolvedDir, verdict)
		if verr != nil {
			fmt.Fprintf(stderr, "fak opencode resume: %v\n", verr)
		}
		fmt.Fprintln(stderr, "refusing to launch: opencode run --session CREATES the session when the id does not exist, so an unverified id would fabricate a fresh empty session and call it recovery")
		return 2
	}
	childArgv, err := opencoderesume.ResumeArgv("opencode", id, *prompt, false)
	if err != nil {
		fmt.Fprintf(stderr, "fak opencode resume: %v\n", err)
		return 2
	}
	argvOut := childArgv
	if *guard {
		argvOut = append([]string{tuiExecutable(), "guard", "--"}, childArgv...)
	}
	launchEnv := opencodeUpsertEnv(os.Environ(), opencodeLaunchWorkingDirEnv, resolvedDir)
	started := time.Now()
	if *guard {
		fmt.Fprintln(stderr, "fak opencode resume: launching OpenCode through fak guard ...")
	} else {
		fmt.Fprintln(stderr, "fak opencode resume: launching OpenCode directly ...")
	}
	code := opencodeLaunchRun(stdout, stderr, argvOut, launchEnv)
	if code == 0 {
		fmt.Fprintf(stderr, "fak opencode resume: OpenCode completed successfully in %s\n", time.Since(started).Round(time.Millisecond))
	}
	return code
}

func prepareOpencodeRouterCredential(o opencodeLaunchOptions, env []string) (opencodeLaunchOptions, []string, error) {
	if explicit := strings.TrimSpace(o.apiKeyEnv); explicit != "" {
		if !validOpencodeEnvName(explicit) {
			return o, env, fmt.Errorf("--api-key-env must name a valid environment variable")
		}
		if key := strings.TrimSpace(opencodeEnvValue(env, explicit)); key == "" || isTemplateCredential(key) {
			return o, env, fmt.Errorf("--api-key-env %s is empty or invalid", explicit)
		}
		o.apiKeyEnv = explicit
		return o, env, nil
	}
	if !isRegisteredFakRouterURL(o.baseURL, o.workspaceDir) {
		return o, env, nil
	}
	key := resolveOpencodeRouterAPIKey(o.baseURL, o.workspaceDir, env)
	if key == "" {
		return o, env, fmt.Errorf("registered fak router credential unavailable or rejected; set %s, pair this router with `fak node use`, or pass --api-key-env VAR", guardRouterKeyEnv)
	}
	o.apiKeyEnv = guardRouterKeyEnv
	return o, opencodeUpsertEnv(env, guardRouterKeyEnv, key), nil
}

func validOpencodeEnvName(name string) bool {
	for i, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' || (i > 0 && r >= '0' && r <= '9') {
			continue
		}
		return false
	}
	return name != ""
}

func opencodeEnvValue(env []string, name string) string {
	prefix := name + "="
	value := ""
	for _, entry := range env {
		if strings.HasPrefix(entry, prefix) {
			value = strings.TrimPrefix(entry, prefix)
		}
	}
	return value
}

func opencodeUpsertEnv(env []string, name, value string) []string {
	prefix := name + "="
	out := make([]string, 0, len(env)+1)
	for _, entry := range env {
		if !strings.HasPrefix(entry, prefix) {
			out = append(out, entry)
		}
	}
	return append(out, prefix+value)
}

func buildOpencodeDirectArgv(o opencodeLaunchOptions) []string {
	return append([]string{"opencode"}, opencodeChildArgs(o)...)
}

// opencodeChildArgs is the OpenCode argv after the binary. OpenCode v2 binds a
// flag to the command it follows and its root command has no --model, so the
// model pin goes after `run`; an interactive launch takes the model from the
// session config's "model" key instead.
func opencodeChildArgs(o opencodeLaunchOptions) []string {
	rest := o.passthrough
	switch {
	case o.probePrompt != "":
	case len(rest) > 0 && rest[0] == "run":
		rest = rest[1:]
	default:
		// Interactive launch: the native resume flags ride at the front, ahead of any
		// operator passthrough (OpenCode v2 binds a flag to the command it follows).
		return append(opencodeResumePassthrough(o), rest...)
	}
	args := []string{"run"}
	args = append(args, opencodeResumePassthrough(o)...)
	guardedModel := ""
	if childModel := guardOpenCodeModelName(strings.TrimSpace(o.model)); childModel != "" {
		providerID := guardOpenCodeLegacyProviderID
		if o.guard && isRegisteredFakRouterURL(o.baseURL, o.workspaceDir) {
			providerID = guardOpenCodeRouterProviderID
		}
		guardedModel = providerID + "/" + childModel
		args = append(args, "--model", guardedModel)
	}
	if o.probePrompt != "" {
		args = append(args, "--format", "json")
		if o.auto || o.skipPermissions {
			args = append(args, "--auto")
		}
		args = append(args, o.probePrompt)
	}
	args = append(args, rest...)
	if guardedModel != "" {
		args = pinOpenCodeModelArgs(args, 1, guardedModel)
	}
	return args
}

func resolveOpencodeWorkspaceDir(raw string) (string, error) {
	dir := pathutil.ExpandTilde(strings.TrimSpace(raw))
	if dir == "" {
		return "", fmt.Errorf("workspace directory is empty")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s is not a directory", abs)
	}
	return filepath.Clean(abs), nil
}

func isRegisteredFakRouterURL(baseURL string, workspaceDirs ...string) bool {
	base, baseOK := normalizedOpenCodeRouterOrigin(baseURL)
	if !baseOK {
		return false
	}
	workspaceDir := "."
	if len(workspaceDirs) > 0 && strings.TrimSpace(workspaceDirs[0]) != "" {
		workspaceDir = workspaceDirs[0]
	}
	candidates := []string{os.Getenv("FAK_AGENT_ROUTER_ORIGIN"), workspaceOpenCodeRouterOrigin(workspaceDir)}
	if cfg, err := nodeReadCfg(); err == nil {
		candidates = append(candidates, cfg.URL)
	}
	for _, candidate := range candidates {
		if registered, ok := normalizedOpenCodeRouterOrigin(candidate); ok && base == registered {
			return true
		}
	}
	return false
}

type opencodeRouterOptions struct {
	APIKey  string `json:"apiKey"`
	BaseURL string `json:"baseURL"`
}

func workspaceOpenCodeRouterOptions(dir string) opencodeRouterOptions {
	data, err := os.ReadFile(filepath.Join(dir, "opencode.json"))
	if err != nil {
		return opencodeRouterOptions{}
	}
	var cfg struct {
		Provider map[string]struct {
			Options opencodeRouterOptions `json:"options"`
		} `json:"provider"`
	}
	if json.Unmarshal(data, &cfg) != nil {
		return opencodeRouterOptions{}
	}
	return cfg.Provider[guardOpenCodeRouterProviderID].Options
}

func workspaceOpenCodeRouterOrigin(dir string) string {
	return workspaceOpenCodeRouterOptions(dir).BaseURL
}

func resolveOpencodeRouterAPIKey(baseURL, workspaceDir string, env []string) string {
	origin := strings.TrimSuffix(strings.TrimRight(baseURL, "/"), "/v1")
	if key := opencodeEnvValue(env, guardRouterKeyEnv); key != "" {
		return verifiedRouterCredential(origin, key)
	}
	if key, paired := pairedNodeRouterAPIKey(origin); paired {
		return key
	}
	options := workspaceOpenCodeRouterOptions(workspaceDir)
	base, baseOK := normalizedOpenCodeRouterOrigin(baseURL)
	registered, registeredOK := normalizedOpenCodeRouterOrigin(options.BaseURL)
	key := strings.TrimSpace(options.APIKey)
	if !baseOK || !registeredOK || base != registered || isTemplateCredential(key) {
		return ""
	}
	return key
}

func normalizedOpenCodeRouterOrigin(raw string) (string, bool) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.User != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.RawQuery != "" || u.Fragment != "" {
		return "", false
	}
	switch strings.TrimRight(u.EscapedPath(), "/") {
	case "", "/v1":
	default:
		return "", false
	}
	return strings.ToLower(u.Scheme) + "://" + strings.ToLower(u.Host), true
}

func validateOpencodeLaunchSplit(mode, where string) error {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "auto", "on", "true", "1", "yes", "off", "false", "0", "no":
	default:
		return fmt.Errorf("--split must be auto|on|off, got %q", mode)
	}
	switch strings.TrimSpace(strings.ToLower(where)) {
	case "", "bottom", "right":
		return nil
	default:
		return fmt.Errorf("--split-where must be %q or %q, got %q", "bottom", "right", where)
	}
}

func buildOpencodeLaunchArgv(fakBin string, o opencodeLaunchOptions) []string {
	argv := []string{
		fakBin,
		"guard",
		"--provider", "openai",
		"--split", firstNonEmpty(strings.TrimSpace(o.splitMode), "auto"),
		"--split-where", firstNonEmpty(strings.TrimSpace(o.splitWhere), "bottom"),
		"--split-interval", o.splitInterval.String(),
	}
	appendKV := func(flag, value string) {
		if strings.TrimSpace(value) != "" {
			argv = append(argv, flag, value)
		}
	}
	appendKV("--policy", o.policyPath)
	appendKV("--api-key-env", o.apiKeyEnv)
	appendKV("--base-url", o.baseURL)
	appendKV("--remote-serve", o.remoteServe)
	appendKV("--model", guardOpenCodeModelName(o.model))
	appendKV("--audit", o.auditPath)
	if o.noAudit {
		argv = append(argv, "--audit", "off")
	}
	if o.quiet {
		argv = append(argv, "--quiet")
	}
	if o.localAuto {
		argv = append(argv, "--local")
	}
	if o.probePrompt != "" {
		argv = append(argv, "--probe")
	}
	appendKV("--gguf", o.ggufPath)
	if o.metal {
		argv = append(argv, "--metal")
	}
	appendKV("--backend", o.gpuBackend)
	appendKV("--tokenizer", o.tokenizerPath)

	argv = append(argv, "--", "opencode")
	return append(argv, opencodeChildArgs(o)...)
}

func execOpencodeLaunchChild(stdout, stderr io.Writer, argv, env []string) int {
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	return execOpencodeLaunchChildContext(ctx, stdout, stderr, argv, env)
}

func execOpencodeLaunchChildContext(ctx context.Context, stdout, stderr io.Writer, argv, env []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak opencode: empty command")
		return 2
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = opencodeEnvValue(env, opencodeLaunchWorkingDirEnv)
	cmd.Env = opencodeRemoveEnv(env, opencodeLaunchWorkingDirEnv)
	// Tee the child's stdout into a bounded capture so the post-exit session-id
	// extraction (#14330) sees what the operator saw; the terminal stream is untouched.
	capture := &opencodeLaunchCapture{}
	publishOpencodeLaunchCapture(capture)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, io.MultiWriter(stdout, capture), stderr
	cmd.WaitDelay = 5 * time.Second
	cmd.Cancel = func() error {
		if cmd.Process != nil && cmd.Process.Pid > 0 {
			procguard.KillPID(cmd.Process.Pid)
		}
		return nil
	}
	if err := cmd.Run(); err != nil {
		code := childprocess.ExitCode(err, 1)
		if code == 1 {
			fmt.Fprintf(stderr, "fak opencode: %v\n", err)
		}
		return code
	}
	return 0
}

func opencodeRemoveEnv(env []string, name string) []string {
	prefix := name + "="
	out := make([]string, 0, len(env))
	for _, entry := range env {
		if !strings.HasPrefix(entry, prefix) {
			out = append(out, entry)
		}
	}
	return out
}

func opencodeConfigModeLabel(mode projectassets.OpenCodeConfigMode) string {
	if mode == projectassets.OpenCodeModeMac {
		return "mac metal native"
	}
	if mode == projectassets.OpenCodeModeHalo {
		return "halo"
	}
	return "gateway"
}

func runOpencodeConfig(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("opencode config", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", "127.0.0.1:8080", "fak serve gateway listen address")
	model := fs.String("model", projectassets.DefaultOpenCodeModelID, "served model ID")
	halo := fs.Bool("halo", false, "configure opencode.json for local AMD Strix Halo server")
	strix := fs.Bool("strix", false, "alias for --halo")
	mac := fs.Bool("mac", false, "configure opencode.json for native in-kernel Apple Silicon Metal inference")
	nativeMetal := fs.Bool("native-metal", false, "alias for --mac")
	write := fs.Bool("write", false, "write or update opencode.json in the current workspace")
	dir := fs.String("dir", ".", "workspace directory containing opencode.json")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak opencode config [--write] [--addr ADDR] [--model MODEL] [--halo|--strix|--mac|--native-metal]")
		fmt.Fprintln(stderr, "  e.g. fak opencode config --mac --write")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	macMode := *mac || *nativeMetal
	if macMode {
		if *model == projectassets.DefaultOpenCodeModelID || *model == "fak-local" {
			*model = projectassets.DefaultOpenCodeMacModelID
		}
	} else if (*halo || *strix) && *model == projectassets.DefaultOpenCodeModelID {
		*model = projectassets.ResolveDynamicHaloModel(*dir)
	}
	baseURL := *addr
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		baseURL = "http://" + baseURL
	}
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL = strings.TrimSuffix(baseURL, "/") + "/v1"
	}
	mode := projectassets.OpenCodeModeGateway
	switch {
	case macMode:
		mode = projectassets.OpenCodeModeMac
	case *halo || *strix:
		mode = projectassets.OpenCodeModeHalo
	}

	if *write {
		modified, err := projectassets.EnsureOpenCodeProviderConfigMode(*dir, baseURL, *model, mode)
		if err != nil {
			fmt.Fprintf(stderr, "fak opencode config: %v\n", err)
			return 1
		}
		label := opencodeConfigModeLabel(mode)
		if modified {
			fmt.Fprintf(stdout, "fak opencode config: updated %s with provider \"fak\" (%s, baseURL: %s, model: %s)\n", filepath.Join(*dir, "opencode.json"), label, baseURL, *model)
		} else {
			fmt.Fprintf(stdout, "fak opencode config: %s already has up-to-date provider \"fak\" (%s)\n", filepath.Join(*dir, "opencode.json"), label)
		}
		return 0
	}
	out, err := projectassets.GenerateOpenCodeConfigMode(baseURL, *model, mode)
	if err != nil {
		fmt.Fprintf(stderr, "fak opencode config: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(out))
	return 0
}
