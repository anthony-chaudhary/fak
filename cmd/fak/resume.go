package main

// resume.go — `fak resume`, the operator front door to the DETERMINISTIC RESUME-CACHE
// decision (internal/resume). It answers the one question an operator asks when bringing a
// dormant agent session back to life: "I am resuming a 250k-token session — what happens to
// the prompt cache, and what should I do about it?"
//
//	fak resume plan --resident-tokens 250000 --idle-seconds 7200
//	fak resume plan --image ./session-img --json
//
// The decision is PURE (internal/resume.Plan): same facts in, same priced verdict out — no
// clock, no model, no network. This shell does only the I/O the pure leaf must not: it reads
// the facts from flags (and, with --image, from a portable session image's trajectory +
// metadata), calls Plan, and renders the report as an aligned table or raw JSON. It is the
// exact split session_cmd.go uses — the decision lives in the leaf, the wire lives here.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resumebackoff"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/sessionimage"
	"github.com/anthony-chaudhary/fak/internal/vcachecal"
)

// cmdResume is the `fak resume` entry point; it maps the testable core's exit code to the
// process exit code, mirroring cmdSession.
func cmdResume(argv []string) { os.Exit(runResume(os.Stdout, os.Stderr, argv)) }

// runResume is the testable core: it returns the process exit code (0 ok, 1 a runtime error,
// 2 a usage error) and takes its streams explicitly so a test drives it without a process.
func runResume(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		resumeUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "plan":
		return runResumePlan(stdout, stderr, argv[1:])
	case "validate":
		return runResumeValidate(stdout, stderr, argv[1:])
	case "scan":
		return runResumeScan(stdout, stderr, argv[1:])
	case "sweep":
		return runResumeSweep(stdout, stderr, argv[1:])
	case "stopped":
		return runResumeStopped(stdout, stderr, argv[1:])
	case "dedup":
		return runResumeDedup(stdout, stderr, argv[1:])
	case "status":
		return runResumeStatus(stdout, stderr, argv[1:])
	case "self":
		return runResumeSelf(stdout, stderr, argv[1:])
	case "admit":
		return runResumeAdmit(stdout, stderr, argv[1:])
	case "watchdog":
		return runResumeWatchdog(stdout, stderr, argv[1:])
	case "cap":
		return runResumeCap(stdout, stderr, argv[1:])
	case "backoff":
		return runResumeBackoff(stdout, stderr, argv[1:])
	case "rearm":
		return runResumeRearm(stdout, stderr, argv[1:])
	case "hold":
		return runResumeHold(stdout, stderr, argv[1:])
	case "release":
		return runResumeRelease(stdout, stderr, argv[1:])
	case "resolve":
		return runResumeResolve(stdout, stderr, argv[1:])
	case "identity":
		return runResumeIdentity(stdout, stderr, argv[1:])
	case "drive":
		return runResumeDrive(stdout, stderr, argv[1:])
	case "why":
		return runResumeWhy(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		resumeUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak resume: unknown subcommand %q (want plan, validate, scan, sweep, stopped, status, self, admit, watchdog, hold, release, resolve, identity, drive, or why)\n", argv[0])
		resumeUsage(stderr)
		return 2
	}
}

// runResumePlan parses the resume facts, optionally grounds them on a real session image,
// computes the deterministic plan, and renders it.
func runResumePlan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "resume")
	residentTokens := fs.Int("resident-tokens", 0, "size of the context that would be re-prefilled on a full resume (the whole transcript)")
	idleSeconds := fs.Int64("idle-seconds", -1, "how long the session was dormant before this resume (-1 = unknown; drives cold-vs-warm against the TTL)")
	ttlStr := fs.String("ttl", "5m", "provider cache TTL tier the session used: 5m (default) or 1h")
	horizon := fs.Int("horizon", 0, "turns expected to remain after resume (0 = default)")
	shedBudget := fs.Int("shed-budget", 0, "CUT target in tokens — what a shed keeps (0 = default ~48k)")
	seedTokens := fs.Int("seed-tokens", 0, "RESET carryover seed size in tokens (0 = default)")
	inputPrice := fs.Float64("input-price", 5, "model base input price per million tokens (default: Opus 4.8 = 5)")
	outputPrice := fs.Float64("output-price", 25, "model base output price per million tokens (default: Opus 4.8 = 25)")
	outputPerTurn := fs.Int("output-per-turn", 0, "completion tokens per modeled turn (0 = default)")
	imageDir := fs.String("image", "", "ground the plan on a portable session image directory: derive resident tokens from its trajectory and idle from its timestamp")
	transcript := fs.String("transcript", "", "ground the plan on a REAL Claude Code session transcript (.jsonl): derive resident tokens from the last assistant turn's prompt size and idle from its timestamp")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	if !parseFlags(fs, argv) {
		return 2
	}

	ttl, ok := parseResumeTTL(*ttlStr)
	if !ok {
		fmt.Fprintf(stderr, "fak resume plan: bad --ttl %q (want 5m or 1h)\n", *ttlStr)
		return 2
	}

	in := resume.Input{
		ResidentTokens:      *residentTokens,
		IdleSeconds:         *idleSeconds,
		TTL:                 ttl,
		Pricing:             resume.Pricing{InputPerMTokUSD: *inputPrice, OutputPerMTokUSD: *outputPrice},
		HorizonTurns:        *horizon,
		ShedBudgetTokens:    *shedBudget,
		SeedTokens:          *seedTokens,
		OutputTokensPerTurn: *outputPerTurn,
	}

	var groundNote string
	if *imageDir != "" {
		note, code := groundOnImage(stderr, *imageDir, &in, fs)
		if code != 0 {
			return code
		}
		groundNote = note
	}
	if *transcript != "" {
		note, code := groundOnTranscript(stderr, *transcript, &in, fs)
		if code != 0 {
			return code
		}
		groundNote = note
	}

	if in.ResidentTokens <= 0 {
		fmt.Fprintln(stderr, "fak resume plan: need --resident-tokens > 0 (or an --image / --transcript that carries token usage)")
		return 2
	}

	rep := resume.Plan(in)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak resume plan")
	}
	renderResumeReport(stdout, rep, groundNote)
	return 0
}

// runResumeAdmit is the PER-SOURCE concurrency gate: it folds the host's current
// live-resume census and recent launch ledger into a snapshot, applies the tunable
// source policy, and returns an admit/refuse verdict. A launcher self-gates on it before
// it spawns a `claude --resume` — exit 0 admit, exit 3 refused — so the per-source 529
// burst wall (#1341/#1344) is bounded by ONE audited decision a launcher cannot route
// around, instead of the per-tick / per-account caps that never counted the box's total
// live resumes. The decision is pure (resume.AdmitSource); this shell does only the I/O
// the leaf forbids: the OS process census and the ledger read.
func runResumeAdmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume admit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "resume")
	ledger := fs.String("ledger", defaultResumeLedger(), "launch ledger JSONL path (the durable record every launcher appends to)")
	policyPath := fs.String("policy", defaultResumeSourcePolicy(), "per-source admission policy JSON path")
	maxLive := fs.Int("max-live", 4, "host-wide ceiling on live `claude --resume` processes across all accounts (0 disables)")
	maxPerWindow := fs.Int("max-per-window", 10, "max recorded launches in the trailing window (0 disables)")
	windowSec := fs.Int64("window-sec", 300, "the rolling launch-rate window, in seconds")
	minSpacingSec := fs.Int64("min-spacing-sec", 8, "host-wide minimum seconds between launches (0 disables)")
	asJSON := fs.Bool("json", false, "emit the decision as JSON")
	quiet := fs.Bool("quiet", false, "suppress the human line (for use as a launcher gate that reads only the exit code)")
	explain := fs.Bool("explain", false, "print the full governor posture: policy path+values+source, ledger path, live census, recent refusals/fail-opens, and the verdict (#2173)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak resume admit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	// The policy file is the standing config; explicit flags OVERRIDE it (flag-over-file,
	// the same precedence launch_admission's --global-cap etc. take). The fail-open loader
	// makes a missing file the permissive default, then we layer the CLI ceilings on top.
	// A PRESENT-but-malformed policy REFUSES (exit 3, POLICY_MALFORMED) instead of
	// usage-erroring (exit 2): launchers fail open on unexpected gate exits, so an exit-2
	// here would silently turn a policy typo into a fully permissive host (#2173). The
	// refusal defers launches loudly — the ledger carries the token — until the typo is fixed.
	policies, err := resume.LoadSourcePolicy(*policyPath)
	if err != nil {
		d := resume.SourceDecision{
			Admit:  false,
			Reason: resume.ReasonPolicyMalformed,
			Summary: fmt.Sprintf("policy %s is present but unreadable (%v) — fix or remove it; a MISSING policy is permissive, a malformed one refuses",
				*policyPath, err),
		}
		if *asJSON {
			if err := writeIndentedJSON(stdout, map[string]any{
				"schema":      "fak.resume-admit.v1",
				"ledger_path": *ledger,
				"policy_path": *policyPath,
				"decision":    d,
			}); err != nil {
				fmt.Fprintf(stderr, "fak resume admit: encode json: %v\n", err)
				return 1
			}
		} else if !*quiet {
			fmt.Fprintf(stdout, "%-6s %-22s %s\n", "REFUSE", d.Reason, d.Summary)
		}
		fmt.Fprintf(stderr, "fak resume admit: %v\n", err)
		return 3
	}
	policy := policies.Default
	applyResumeSourceFlagOverrides(fs, &policy, *maxLive, *maxPerWindow, *windowSec, *minSpacingSec)

	now := time.Now()
	snap := foldSourceSnapshot(*ledger, now)
	d := resume.AdmitSource(snap, policy, now)

	if *asJSON {
		doc := map[string]any{
			"schema":      "fak.resume-admit.v1",
			"ledger_path": *ledger,
			"policy_path": *policyPath,
			"snapshot":    snap,
			"decision":    d,
		}
		if *explain {
			doc["explain"] = explainSourceGovernor(*policyPath, *ledger, policy, now)
		}
		if err := writeIndentedJSON(stdout, doc); err != nil {
			fmt.Fprintf(stderr, "fak resume admit: encode json: %v\n", err)
			return 1
		}
	} else if *explain {
		renderSourceGovernorExplain(stdout, *policyPath, *ledger, policy, snap, d, now)
	} else if !*quiet {
		verdict := "ADMIT"
		if !d.Admit {
			verdict = "REFUSE"
		}
		fmt.Fprintf(stdout, "%-6s %-22s %s\n", verdict, d.Reason, d.Summary)
	}

	// Exit 3 when refused, so a launcher can gate with `fak resume admit && spawn`,
	// matching `fak loop admit` and launch_admission.py's exit-3 DEFER contract.
	if !d.Admit {
		return 3
	}
	return 0
}

// governorLedgerStats folds the trailing-24h governor activity out of the launch
// ledger: how many real launches fired, how many the gate deferred, and how many
// fail-open warning rows launchers recorded because the governor was unavailable.
// These are the status rows for source-governor refusals and fail-open launches
// (#2173), read straight off the durable ledger every launcher already appends to.
type governorLedgerStats struct {
	Launched24h    int   `json:"launched_24h"`
	Deferred24h    int   `json:"deferred_24h"`
	FailOpen24h    int   `json:"gate_fail_open_24h"`
	LastLaunchUnix int64 `json:"last_launch_unix,omitempty"`
}

// scanGovernorLedgerStats reads the ledger JSONL once and classifies each trailing-24h
// row by phase: gate_fail_open (a launcher ran without the governor), deferred (the
// gate refused), any other non-launch phase (broker_denied/considered/skipped/settled…)
// ignored, and only what remains — "launched" or a legacy phase-less row — a real
// launch. The launch rule is isNonLaunchPhase, the same one scanLaunchLedger applies,
// so the two readers cannot drift: a denylist miss here once counted broker_denied as
// a launch and poisoned the spacing floor. A missing/unreadable ledger yields zeros:
// no record is no activity, never an error.
func scanGovernorLedgerStats(path string, now time.Time) governorLedgerStats {
	var st governorLedgerStats
	f, err := os.Open(path)
	if err != nil {
		return st
	}
	defer f.Close()

	cutoff := now.UTC().Add(-24 * time.Hour).Unix()
	type lrec struct {
		Ts    string `json:"ts"`
		Phase string `json:"phase"`
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r lrec
		if json.Unmarshal(line, &r) != nil {
			continue
		}
		ts := parseTranscriptUnix(r.Ts)
		if ts == 0 {
			continue
		}
		inWindow := ts >= cutoff
		phase := strings.ToLower(strings.TrimSpace(r.Phase))
		switch {
		case phase == "gate_fail_open":
			if inWindow {
				st.FailOpen24h++
			}
		case phase == "deferred":
			if inWindow {
				st.Deferred24h++
			}
		case isNonLaunchPhase(phase):
			// non-launch bookkeeping rows (broker_denied, settled, considered, …):
			// not activity worth surfacing, and never a launch — nothing started
		default:
			// A real launch: "launched", or a legacy phase-less row (pid/cause only).
			if inWindow {
				st.Launched24h++
			}
			// Deliberately un-windowed: the posture report wants the TRUE most recent
			// launch even when it is older than 24h ("last launch <ts>", not "never"),
			// matching the un-windowed `last` scanLaunchLedger feeds the spacing floor.
			if ts > st.LastLaunchUnix {
				st.LastLaunchUnix = ts
			}
		}
	}
	return st
}

// explainSourceGovernor assembles the machine-readable governor posture: where the
// policy came from and what it effectively says, where the ledger lives, the recent
// refusal / fail-open activity, and which binary answered. One `--explain` call
// replaces reading `.fak/` and the scheduler actions by hand (#2173).
func explainSourceGovernor(policyPath, ledgerPath string, policy resume.SourcePolicy, now time.Time) map[string]any {
	exe, _ := os.Executable()
	_, statErr := os.Stat(policyPath)
	_, ledgerErr := os.Stat(ledgerPath)
	return map[string]any{
		"policy_path":        policyPath,
		"policy_file_exists": statErr == nil,
		"policy_env_set":     strings.TrimSpace(os.Getenv("FAK_RESUME_SOURCE_POLICY")) != "",
		"policy_effective":   policy,
		"ledger_path":        ledgerPath,
		"ledger_exists":      ledgerErr == nil,
		"recent":             scanGovernorLedgerStats(ledgerPath, now),
		"executable":         exe,
	}
}

// renderSourceGovernorExplain prints the human governor-posture report — the one-command
// answer to "what rail is this host actually running under, and has it been bypassed?"
func renderSourceGovernorExplain(w io.Writer, policyPath, ledgerPath string, policy resume.SourcePolicy, snap resume.SourceSnapshot, d resume.SourceDecision, now time.Time) {
	exe, _ := os.Executable()
	policySrc := "default path"
	if strings.TrimSpace(os.Getenv("FAK_RESUME_SOURCE_POLICY")) != "" {
		policySrc = "env FAK_RESUME_SOURCE_POLICY"
	}
	policyState := "MISSING — permissive file defaults; CLI ceilings apply"
	if _, err := os.Stat(policyPath); err == nil {
		policyState = "exists"
	}
	st := scanGovernorLedgerStats(ledgerPath, now)
	lastLaunch := "never"
	if st.LastLaunchUnix > 0 {
		lastLaunch = time.Unix(st.LastLaunchUnix, 0).UTC().Format(time.RFC3339)
	}
	verdict := "ADMIT"
	if !d.Admit {
		verdict = "REFUSE"
	}
	fmt.Fprintf(w, "source governor posture\n")
	fmt.Fprintf(w, "  policy:   %s (%s; via %s)\n", policyPath, policyState, policySrc)
	fmt.Fprintf(w, "            max_live_resumes=%d max_launches_per_window=%d window_seconds=%d min_launch_spacing_seconds=%d\n",
		policy.MaxLiveResumes, policy.MaxLaunchesPerWindow, policy.WindowSeconds, policy.MinLaunchSpacingSeconds)
	fmt.Fprintf(w, "  ledger:   %s\n", ledgerPath)
	fmt.Fprintf(w, "            trailing 24h: launched=%d deferred=%d gate_fail_open=%d; last launch %s\n",
		st.Launched24h, st.Deferred24h, st.FailOpen24h, lastLaunch)
	fmt.Fprintf(w, "  census:   %d live `claude --resume` process(es); %d launch(es) in the policy window\n",
		snap.LiveResumeCount, d.WindowLaunches)
	fmt.Fprintf(w, "  binary:   %s\n", exe)
	fmt.Fprintf(w, "  verdict:  %s %s — %s\n", verdict, d.Reason, d.Summary)
	if st.FailOpen24h > 0 {
		fmt.Fprintf(w, "  WARNING:  %d gate_fail_open row(s) in 24h — launchers ran WITHOUT the governor; check fak.exe resolution on the scheduled-task path\n", st.FailOpen24h)
	}
}

// applyResumeSourceFlagOverrides layers explicitly-set CLI ceilings over the loaded
// policy. A flag the operator did not set leaves the file's value (or the permissive
// zero) intact; a flag they did set wins. This keeps the policy file the durable default
// while letting a one-off invocation (or a launcher that hard-codes its ceilings) override.
func applyResumeSourceFlagOverrides(fs *flag.FlagSet, policy *resume.SourcePolicy, maxLive, maxPerWindow int, windowSec, minSpacingSec int64) {
	if flagSet(fs, "max-live") {
		policy.MaxLiveResumes = maxLive
	} else if policy.MaxLiveResumes == 0 {
		policy.MaxLiveResumes = maxLive // a fresh policy file inherits the CLI default ceiling
	}
	if flagSet(fs, "max-per-window") {
		policy.MaxLaunchesPerWindow = maxPerWindow
	} else if policy.MaxLaunchesPerWindow == 0 {
		policy.MaxLaunchesPerWindow = maxPerWindow
	}
	if flagSet(fs, "window-sec") {
		policy.WindowSeconds = windowSec
	} else if policy.WindowSeconds == 0 {
		policy.WindowSeconds = windowSec
	}
	if flagSet(fs, "min-spacing-sec") {
		policy.MinLaunchSpacingSeconds = minSpacingSec
	} else if policy.MinLaunchSpacingSeconds == 0 {
		policy.MinLaunchSpacingSeconds = minSpacingSec
	}
}

// resumeProcRe matches a `claude --resume <session-id>` invocation in a process command
// line — the same signal the python audit tools (resume_sweep.live_resume_sids) key on.
// It tolerates `claude`, `claude.exe`, a full path, and any flags between the exe and
// `--resume`; the trailing token is the session id (a uuid or any non-space run).
var resumeProcRe = regexp.MustCompile(`(?i)claude(?:\.exe)?\b.*--resume\s+(\S+)`)

// countLiveResumes returns how many processes on this host are a live `claude --resume`,
// across every account — the host-wide standing-concurrency truth the per-source 529 wall
// keys on, which no per-tick / per-account cap ever measured. It uses the same audited
// cross-platform census procguard already ships (Windows CIM CommandLine; POSIX /proc or
// ps), so there is one process-enumeration implementation, not a fork.
func countLiveResumes() int {
	procs, collectErr := procguard.CollectRelations()
	if collectErr != "" {
		// #5385: a census that did not RUN is not a quiet host. Dropping this error made the
		// host-wide live ceiling silently inert wherever the collector could not read the
		// platform's process table — observed on darwin reporting 0 while four `claude
		// --resume` drivers were live, i.e. exactly AT the default ceiling, which therefore
		// never fired. The count below is still the only number this function has, so it is
		// still returned; what changes is that the operator now hears that it is not a
		// measurement, the way `fak fleet` already surfaces collectErr.
		fmt.Fprintf(os.Stderr, "fak resume: live-resume census failed (%s) — the host-wide live ceiling is NOT enforceable this run\n", collectErr)
	}
	n := 0
	for _, p := range procs {
		if resumeProcRe.MatchString(p.Cmdline) {
			n++
		}
	}
	return n
}

// foldSourceSnapshot builds the SourceSnapshot the pure decision consumes: the live
// process census plus the recorded launch timestamps from the durable ledger. The two
// signals are independent — the census is the OS truth, the ledger is the launch record —
// so neither has to trust the other.
func foldSourceSnapshot(ledgerPath string, now time.Time) resume.SourceSnapshot {
	times, last := scanLaunchLedger(ledgerPath)
	return resume.SourceSnapshot{
		LiveResumeCount: countLiveResumes(),
		LaunchUnixTimes: times,
		LastLaunchUnix:  last,
	}
}

// scanLaunchLedger reads the launch ledger JSONL and returns the unix-second timestamps
// of recorded LAUNCHES (and the most recent one). A row whose `phase` marks a non-launch
// (deferred/considered/skipped) is excluded so the gate's own DEFER rows never count as
// launch pressure — the same `_is_launch` rule launch_admission.py uses. Rows vary in
// shape (some carry no `phase`/`pid`); only `ts` and the optional `phase` are read, so a
// minimal or forward-extended row is handled. A missing/unreadable ledger yields no
// launches (fail-open: an absent record never blocks a launch).
func scanLaunchLedger(path string) (times []int64, last int64) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0
	}
	defer f.Close()

	type lrec struct {
		Ts    string `json:"ts"`
		Phase string `json:"phase"`
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r lrec
		if json.Unmarshal(line, &r) != nil {
			continue
		}
		if isNonLaunchPhase(r.Phase) {
			continue
		}
		ts := parseTranscriptUnix(r.Ts)
		if ts == 0 {
			continue
		}
		times = append(times, ts)
		if ts > last {
			last = ts
		}
	}
	return times, last
}

// isNonLaunchPhase reports whether a ledger row's phase marks something that is NOT a
// fired launch. Named launch phases are an allowlist: "launched" and "resumed" record
// spawns; every other named phase is bookkeeping and therefore cannot create launch
// pressure. This matches resume.Attempt's typed classifier and prevents a new decision
// token such as "revived" from silently resetting LastLaunchUnix on every watchdog tick
// (#8722). This is the ONE reader-side launch rule in cmd/fak: every ledger scanner
// classifies through it so the rules cannot drift apart. An empty phase remains a launch
// for the pre-phase legacy rows that carry pid/cause and recorded a real spawn.
func isNonLaunchPhase(phase string) bool {
	switch strings.ToLower(strings.TrimSpace(phase)) {
	case "", "launched", "resumed":
		return false
	default:
		return true
	}
}

// defaultResumeLedger is the durable launch ledger every launcher appends to, under the
// fleet registry dir (FLEET_REG_DIR, default host Fleet registry when present, else
// tools/_registry) — the same path the registered watchdog uses.
func defaultResumeLedger() string {
	reg := strings.TrimSpace(os.Getenv("FLEET_REG_DIR"))
	if reg == "" {
		reg = defaultFleetRegistryDir()
	}
	return filepath.Join(reg, "resume_ledger.jsonl")
}

// defaultResumeSourcePolicy is the per-source policy path: FAK_RESUME_SOURCE_POLICY if
// set, else .fak/resume-source-policy.json (the same env-then-default-path idiom
// defaultLoopPolicy uses for the loop governor).
func defaultResumeSourcePolicy() string {
	if v := strings.TrimSpace(os.Getenv("FAK_RESUME_SOURCE_POLICY")); v != "" {
		return v
	}
	return filepath.Join(".fak", "resume-source-policy.json")
}

// runResumeValidate is the VALIDATION half of the verb: it back-tests the resume-cache
// projection against billed reality. It scans a corpus of real Claude Code transcripts, lifts
// each one's per-turn usage records (the cache_read / cache_creation counts the provider
// actually billed — no transcript content), and feeds them to resume.Backtest, which scores
// how often the projection's cold/warm posture call agreed with what the provider did and how
// exactly the cold-cost premise held. It is the deterministic, observable answer to "is the
// projection's cache-value call EFFECTIVE on our real sessions?" — the honest precursor to
// auto-firing the plan on a live resume.
func runResumeValidate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "resume")
	corpus := fs.String("corpus", "", "directory of real Claude Code transcripts (.jsonl, scanned recursively) to back-test the projection against")
	ttlStr := fs.String("ttl", "5m", "provider cache TTL tier to score the projection at: 5m (default) or 1h")
	maxFiles := fs.Int("max-files", 0, "cap the number of transcript files scanned (0 = no cap)")
	asJSON := fs.Bool("json", false, "emit the raw BacktestReport JSON instead of the human table")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *corpus == "" {
		fmt.Fprintln(stderr, "fak resume validate: need --corpus DIR (a directory of .jsonl transcripts)")
		return 2
	}
	ttl, ok := parseResumeTTL(*ttlStr)
	if !ok {
		fmt.Fprintf(stderr, "fak resume validate: bad --ttl %q (want 5m or 1h)\n", *ttlStr)
		return 2
	}

	// Expand a leading ~ so `-corpus ~/.claude/projects` works under cmd.exe /
	// PowerShell (which pass ~ through literally) - the same way the GGUF flag does.
	corpusDir := pathutil.ExpandTilde(*corpus)
	files, err := findTranscripts(corpusDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume validate: scan corpus %q: %v\n", *corpus, err)
		return 1
	}
	if *maxFiles > 0 && len(files) > *maxFiles {
		files = files[:*maxFiles]
	}
	if len(files) == 0 {
		fmt.Fprintf(stderr, "fak resume validate: no .jsonl transcripts under %q\n", *corpus)
		return 1
	}

	sessions := make([][]resume.ObservedTurn, 0, len(files))
	scanned := 0
	for _, path := range files {
		turns := loadTranscriptTurns(path)
		if len(turns) >= 2 { // a session needs at least one adjacent pair to score
			sessions = append(sessions, turns)
		}
		scanned++
	}

	rep := resume.Backtest(sessions, ttl, resume.DefaultRecoveryBand())
	cal := vcachecal.CalibrateResumeTTL(gapBucketsToResumeBuckets(rep.Buckets), rep.TTLSeconds*1000)
	if *asJSON {
		out := resumeValidateReport{BacktestReport: rep, TTLCalibration: cal}
		return encodeJSONOrFail(stdout, stderr, out, "fak resume validate")
	}
	renderBacktestReport(stdout, rep, scanned, len(sessions))
	renderTTLCalibration(stdout, cal)
	return 0
}

// resumeValidateReport is the `fak resume validate --json` envelope: the back-test residual
// (internal/resume.Backtest) PLUS the #1614 TTL-calibration verdict fit from the SAME gap
// buckets — whether the provider TTL the back-test assumed is well-calibrated against real
// resume timing, and a suggested revision when it is not. BacktestReport is embedded (not
// nested under its own key) so every existing consumer of the flat report shape keeps working
// unchanged; TTLCalibration is purely additive.
type resumeValidateReport struct {
	resume.BacktestReport
	TTLCalibration vcachecal.TTLCalibrationVerdict `json:"ttl_calibration"`
}

// gapBucketsToResumeBuckets adapts internal/resume's gap-bucketed back-test tallies into the
// generic vcachecal.ResumeGapBucket shape the calibration fold consumes. This is the seam
// vcacheobserve already uses to bridge vcachecal.Calibration against real provider telemetry
// (contextjoin.go) — resume (tier 1) cannot import vcachecal (tier 2), so the join happens
// here, in the tier-4 shell that already imports both.
func gapBucketsToResumeBuckets(buckets []resume.GapBucket) []vcachecal.ResumeGapBucket {
	out := make([]vcachecal.ResumeGapBucket, 0, len(buckets))
	for _, b := range buckets {
		out = append(out, vcachecal.ResumeGapBucket{
			LoSeconds: b.LoSeconds,
			HiSeconds: b.HiSeconds,
			WarmN:     b.WarmN,
			ColdN:     b.ColdN,
		})
	}
	return out
}

// renderTTLCalibration prints the #1614 verdict: whether the provider TTL the back-test
// assumed is well-calibrated against real resume timing, and — when it is not — the closed
// reason plus a suggested revision fit from the SAME evidence, never auto-applied.
func renderTTLCalibration(w io.Writer, v vcachecal.TTLCalibrationVerdict) {
	fmt.Fprintf(w, "\nTTL calibration (assumed %dms against %d real resume(s)):\n", v.AssumedTTLMillis, v.N)
	verdict := "WELL-CALIBRATED"
	if !v.WellCalibrated {
		verdict = "MISCALIBRATED"
	}
	fmt.Fprintf(w, "  %s (%s)\n", verdict, v.Reason)
	if v.WithinTTLN > 0 {
		fmt.Fprintf(w, "  within-TTL warm rate: %.1f%% (n=%d)\n", v.WithinTTLWarmRate*100, v.WithinTTLN)
	}
	if v.PastTTLN > 0 {
		fmt.Fprintf(w, "  past-TTL warm rate:   %.1f%% (n=%d)\n", v.PastTTLWarmRate*100, v.PastTTLN)
	}
	if v.SuggestedTTLMillis > 0 {
		fmt.Fprintf(w, "  suggested TTL: %dms (%ds) — fit from the widest reliably-warm observed bucket, not auto-applied\n",
			v.SuggestedTTLMillis, v.SuggestedTTLMillis/1000)
	}
}

// findTranscripts walks a corpus directory and returns every .jsonl file under it (sorted, so
// the scan and the report are deterministic). A directory it cannot read is an error; a single
// unreadable file is simply skipped by the loader, never fatal.
func findTranscripts(root string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip an unreadable subtree rather than abort the whole scan
		}
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".jsonl") {
			out = append(out, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(out)
	return out, nil
}

// loadTranscriptTurns reads one Claude Code transcript and returns its ordered assistant turns
// as the content-free ObservedTurns the back-test scores: the record timestamp and the three
// input-token axes. It is best-effort — a malformed or timestamp-less line is skipped, never
// fatal — and reuses the exact usage shape scanTranscriptResident reads.
func loadTranscriptTurns(path string) []resume.ObservedTurn {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	type usage struct {
		InputTokens         int `json:"input_tokens"`
		CacheReadTokens     int `json:"cache_read_input_tokens"`
		CacheCreationTokens int `json:"cache_creation_input_tokens"`
	}
	type jrec struct {
		Type      string `json:"type"`
		Timestamp string `json:"timestamp"`
		Message   *struct {
			Role  string `json:"role"`
			Usage *usage `json:"usage"`
		} `json:"message"`
	}
	var out []resume.ObservedTurn
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var jr jrec
		if json.Unmarshal(line, &jr) != nil {
			continue
		}
		if jr.Message == nil || jr.Message.Usage == nil || jr.Message.Role != "assistant" {
			continue
		}
		ts := parseTranscriptUnix(jr.Timestamp)
		if ts == 0 {
			continue // a turn with no usable time cannot anchor a gap
		}
		out = append(out, resume.ObservedTurn{
			UnixSeconds:         ts,
			InputTokens:         jr.Message.Usage.InputTokens,
			CacheCreationTokens: jr.Message.Usage.CacheCreationTokens,
			CacheReadTokens:     jr.Message.Usage.CacheReadTokens,
		})
	}
	return out
}

// renderBacktestReport prints the deterministic validation residual: the headline posture
// accuracy and the two miss directions, the per-gap cache-survival curve (where the single TTL
// cutoff agrees with the provider's real reuse window), and the cold-cost validation. Every
// number is the provider's own usage scored against the projection, never a fak figure.
func renderBacktestReport(w io.Writer, r resume.BacktestReport, scanned, sessions int) {
	fmt.Fprintf(w, "resume validate — back-test of the cache-posture projection against billed reality\n")
	fmt.Fprintf(w, "corpus: %d transcripts scanned, %d scorable sessions  ttl=%s (%ds)\n\n",
		scanned, sessions, r.TTL, r.TTLSeconds)

	fmt.Fprintf(w, "boundaries: %d pairs  %d scored  %d ambiguous (excluded)\n", r.Pairs, r.Scored, r.Ambiguous)
	fmt.Fprintf(w, "posture-prediction accuracy: %.1f%% (%d/%d)\n", r.Accuracy*100, r.Agree, r.Scored)
	fmt.Fprintf(w, "  misses: proj=COLD obs=WARM (TTL shorter than reality): %d\n", r.ProjColdObsWarm)
	fmt.Fprintf(w, "          proj=WARM obs=COLD (prefix dropped early)      : %d\n\n", r.ProjWarmObsCold)

	fmt.Fprintf(w, "%-16s %9s %10s %7s %7s %7s\n", "gap-bucket(s)", "n", "mean_recov", "warm%", "cold%", "ambig%")
	for _, b := range r.Buckets {
		if b.N == 0 {
			continue
		}
		fmt.Fprintf(w, "%-16s %9d %10.2f %6.0f%% %6.0f%% %6.0f%%\n",
			bucketLabel(b.LoSeconds, b.HiSeconds), b.N, b.MeanRecovery,
			100*pct(b.WarmN, b.N), 100*pct(b.ColdN, b.N), 100*pct(b.AmbiguousN, b.N))
	}

	fmt.Fprintf(w, "\ncold-cost validation (within-file gaps): %d confirmed-cold boundaries\n", r.ConfirmedCold)
	if r.ConfirmedCold > 0 {
		fmt.Fprintf(w, "  cache_creation / prompt on cold turns: %.2f  (1.00 = the projection's 'whole resident re-written')\n", r.ColdWriteRatioMean)
	}

	fmt.Fprintf(w, "\ncross-file resume re-prefills (first turn of a session file — the genuine multi-hour resume):\n")
	fmt.Fprintf(w, "  %d large resume re-prefills: %d cold (re-prefilled) · %d warm (cross-session cache hit)\n",
		r.FirstTurnResumes, r.FirstTurnCold, r.FirstTurnWarmHit)
	if r.FirstTurnCold > 0 {
		fmt.Fprintf(w, "  cold re-prefill: mean %.0f tok, cache_creation/prompt = %.2f (write-premium SHARE of the cold cost;\n",
			r.FirstTurnColdReprefillTokMean, r.FirstTurnColdWriteShareMean)
		fmt.Fprintf(w, "    below 1.0 means the resume re-cached only part — the projection over-states cold cost by the rest)\n")
	}
	fmt.Fprintln(w, "  (every number is the provider's own usage scored against the projection, not a fak figure)")
}

// bucketLabel renders a gap bucket's range; the open-ended top bucket prints as "N+".
func bucketLabel(lo, hi int64) string {
	if hi >= 1<<61 {
		return fmt.Sprintf("%d+", lo)
	}
	return fmt.Sprintf("%d-%d", lo, hi)
}

// pct is a zero-safe fraction.
func pct(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

// from it (unless the operator set them explicitly). It returns a one-line note about what
// it derived, and an exit code (0 ok, 1 load/parse error). The resident size is the sum of
// the trajectory's per-turn token estimates; idle is now - the image's UpdatedUnix.
func groundOnImage(stderr io.Writer, dir string, in *resume.Input, fs *flag.FlagSet) (string, int) {
	img, err := sessionimage.LoadDir(dir)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume plan: load image %q: %v\n", dir, err)
		return "", 1
	}
	turns, err := img.Trajectory()
	if err != nil {
		fmt.Fprintf(stderr, "fak resume plan: read image trajectory: %v\n", err)
		return "", 1
	}
	sum := 0
	for _, t := range turns {
		if t.TokenEstimate > 0 {
			sum += t.TokenEstimate
		}
	}
	// Only fill what the operator did not pin on the command line.
	if !flagSet(fs, "resident-tokens") && sum > 0 {
		in.ResidentTokens = sum
	}
	if !flagSet(fs, "idle-seconds") && img.Meta.UpdatedUnix > 0 {
		idle := time.Now().Unix() - img.Meta.UpdatedUnix
		if idle < 0 {
			idle = 0
		}
		in.IdleSeconds = idle
	}
	model := img.Meta.Model
	if model == "" {
		model = "unknown"
	}
	return fmt.Sprintf("image %s (model %s, %d turns, resident≈%d tok)", dir, model, len(turns), sum), 0
}

// groundOnTranscript reads a REAL Claude Code session transcript (.jsonl) and fills the
// resident-token and idle facts from it (unless the operator pinned them). The resident
// context that a resume re-prefills is the prompt size of the MOST RECENT assistant turn:
// the provider's reported input_tokens + cache_read_input_tokens + cache_creation_input_tokens
// for that turn (the full prompt the model last had to read). Idle is now minus the last
// record's timestamp. This is the deterministic, observable counterpart to `claude --resume`:
// it answers "this exact session I am about to resume — what happens to the cache?".
func groundOnTranscript(stderr io.Writer, path string, in *resume.Input, fs *flag.FlagSet) (string, int) {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume plan: open transcript %q: %v\n", path, err)
		return "", 1
	}
	defer f.Close()

	resident, model, lastUnix, turns, ok := scanTranscriptResident(f)
	if !ok {
		fmt.Fprintf(stderr, "fak resume plan: transcript %q has no assistant turn with usage — pass --resident-tokens\n", path)
		return "", 1
	}
	if !flagSet(fs, "resident-tokens") && resident > 0 {
		in.ResidentTokens = resident
	}
	if !flagSet(fs, "idle-seconds") && lastUnix > 0 {
		idle := time.Now().Unix() - lastUnix
		if idle < 0 {
			idle = 0
		}
		in.IdleSeconds = idle
	}
	if model == "" {
		model = "unknown"
	}
	return fmt.Sprintf("transcript %s (model %s, %d turns, resident=%d tok from last assistant prompt)", path, model, turns, resident), 0
}

// scanTranscriptResident scans a Claude Code transcript JSONL and returns the resident
// context size (the last assistant turn's total prompt tokens), the model that turn used,
// the last record's unix timestamp, the number of assistant turns seen, and whether any
// assistant usage was found. It is best-effort over real data: a malformed line is skipped,
// never fatal. Only the fields it needs are typed (forward-compatible by construction).
func scanTranscriptResident(r io.Reader) (resident int, model string, lastUnix int64, turns int, ok bool) {
	type usage struct {
		InputTokens         int `json:"input_tokens"`
		CacheReadTokens     int `json:"cache_read_input_tokens"`
		CacheCreationTokens int `json:"cache_creation_input_tokens"`
	}
	type jrec struct {
		Type      string `json:"type"`
		Timestamp string `json:"timestamp"`
		Message   *struct {
			Role  string `json:"role"`
			Model string `json:"model"`
			Usage *usage `json:"usage"`
		} `json:"message"`
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20) // a single tool-result line can be large
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var jr jrec
		if json.Unmarshal(line, &jr) != nil {
			continue
		}
		if t := parseTranscriptUnix(jr.Timestamp); t > lastUnix {
			lastUnix = t
		}
		if jr.Message == nil || jr.Message.Usage == nil || jr.Message.Role != "assistant" {
			continue
		}
		turns++
		// The most recent assistant turn's prompt size IS the resident context a resume
		// re-prefills: the uncached remainder plus whatever the provider had cached.
		resident = jr.Message.Usage.InputTokens + jr.Message.Usage.CacheReadTokens + jr.Message.Usage.CacheCreationTokens
		if jr.Message.Model != "" {
			model = jr.Message.Model
		}
		ok = true
	}
	return resident, model, lastUnix, turns, ok
}

// parseTranscriptUnix parses a Claude Code transcript timestamp (RFC3339, e.g.
// "2026-06-26T18:31:17.123Z") into unix seconds, returning 0 on any parse failure so a
// missing/odd timestamp simply does not advance the idle clock.
func parseTranscriptUnix(s string) int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.Unix()
	}
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.Unix()
	}
	return 0
}

// flagSet reports whether a flag was explicitly provided on the command line (vs left at its
// default), so --image / --transcript only fill the facts the operator did not pin.
func flagSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

// parseResumeTTL maps the --ttl token to a resume.CacheTTL.
func parseResumeTTL(s string) (resume.CacheTTL, bool) {
	switch s {
	case "5m", "5min", "ephemeral", "":
		return resume.TTL5m, true
	case "1h", "1hr", "hour":
		return resume.TTL1h, true
	}
	return "", false
}

// renderResumeReport prints the deterministic plan as an aligned, scannable table: the
// projected cache posture, the three priced strategies, and the recommendation with its
// closed reason. Every dollar is a projection over the resident-token count at the supplied
// base price, never a witnessed bill (the leaf's fence, surfaced here too).
func renderResumeReport(w io.Writer, r resume.Report, imgNote string) {
	if imgNote != "" {
		fmt.Fprintf(w, "grounded on %s\n", imgNote)
	}
	idle := "unknown"
	if r.IdleSeconds >= 0 {
		idle = fmt.Sprintf("%ds", r.IdleSeconds)
	}
	fmt.Fprintf(w, "resume plan — resident=%d tok  idle=%s  ttl=%s (%ds)  posture=%s (%s)\n",
		r.ResidentTokens, idle, r.TTL, r.TTLSeconds, upper(string(r.Posture)), r.PostureReason)
	fmt.Fprintf(w, "model input=$%.2f/MTok output=$%.2f/MTok  horizon=%d turns  output/turn=%d\n\n",
		r.Pricing.InputPerMTokUSD, r.Pricing.OutputPerMTokUSD, r.HorizonTurns, r.OutputTokensPerTurn)

	fmt.Fprintf(w, "%-12s %10s %6s %16s %14s %16s\n",
		"strategy", "prefill", "keep", "cold-reprefill", "first-turn", fmt.Sprintf("horizon(%d)", r.HorizonTurns))
	for _, c := range r.Strategies {
		mark := "  "
		if c.Strategy == r.Recommended {
			mark = "->"
		}
		fmt.Fprintf(w, "%s%-10s %10d %5.0f%% %16s %14s %16s\n",
			mark, c.Strategy, c.PrefillTokens, c.ContextKeptFraction*100,
			usd(c.ColdReprefillUSD), usd(c.FirstTurnUSD), usd(c.HorizonUSD))
	}

	fmt.Fprintf(w, "\nrecommended: %s  (%s)\n", upper(string(r.Recommended)), r.Reason)
	if r.RecommendedSavingsUSD > 0 {
		fmt.Fprintf(w, "  projected horizon saving vs resume_full: %s over %d turns\n", usd(r.RecommendedSavingsUSD), r.HorizonTurns)
	}
	if r.BreakEvenTurns > 0 {
		fmt.Fprintf(w, "  warm-burst gate: a cut repays its re-prefill after %d turns\n", r.BreakEvenTurns)
	}
	fmt.Fprintln(w, "  (dollars are a projection over the resident-token count, not a witnessed bill)")
}

// usd renders a dollar figure: small values keep enough precision to be meaningful, larger
// ones round to cents. A cold 250k re-prefill is ~$1.56; a reset seed is ~$0.025 — both
// need to read cleanly.
func usd(v float64) string {
	switch {
	case v == 0:
		return "$0"
	case v < 1:
		return fmt.Sprintf("$%.4f", v)
	default:
		return fmt.Sprintf("$%.2f", v)
	}
}

// upper uppercases an ASCII token for the header line (posture/strategy emphasis) without
// pulling in strings just for this.
func upper(s string) string {
	return asciiUpper(s)
}

// runResumeHold is the operator WRITER half of the drive-state alignment: it records a durable
// pause / drain / stop of a specific session in the UUID-keyed drive-state store the watchdog
// reads, so an operator's intent for a session is HONORED by the automatic resume layer instead
// of being fought by it. This is the one operator surface that lands in the watchdog's own
// keyspace — the Claude transcript UUID `claude --resume` takes — because the `fak session`
// control plane is keyed by the disjoint gateway trace and can never reach the watchdog.
//
//	fak resume hold <session-id>                 # pause auto-resume (reversible)
//	fak resume hold <session-id> --state stopped # terminal: never auto-resume again
//	fak resume hold --list                       # show the current effective holds
//
// A hold is durable (no TTL — it survives the descriptor registry's 30-min GC) and reversible
// only by the operator (`fak resume release`), never by the watchdog. Exit 0 ok, 1 runtime, 2 usage.
func runResumeHold(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume hold", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "resume")
	state := fs.String("state", "paused", "drive-state to record: paused (reversible hold), draining, or stopped (terminal intent — never auto-resume)")
	reason := fs.String("reason", "", "optional operator note recorded with the hold")
	regDirFlag := fs.String("reg-dir", "", "registry dir holding resume_drivestate.jsonl (default: the same regDir the watchdog resolves — $FLEET_REG_DIR, else the host Fleet registry, else <repo>/tools/_registry)")
	list := fs.Bool("list", false, "list the current effective operator holds instead of setting one")
	asJSON := fs.Bool("json", false, "with --list, emit the holds as JSON")
	if listRequestedRaw(argv) {
		if !parseFlags(fs, argv) {
			return 2
		}
		return renderResumeHolds(stdout, stderr, resolveSweepRegDir(*regDirFlag), *asJSON)
	}
	sid, parsed := parseInterspersedResumeHold(fs, argv, stderr)
	if !parsed {
		return 2
	}
	regDir := resolveSweepRegDir(*regDirFlag)
	if *list {
		return renderResumeHolds(stdout, stderr, regDir, *asJSON)
	}
	st, ok := normalizeHoldState(*state)
	if !ok {
		fmt.Fprintf(stderr, "fak resume hold: bad --state %q (want paused, draining, or stopped)\n", *state)
		return 2
	}
	if err := os.MkdirAll(regDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "fak resume hold: create registry dir %q: %v\n", regDir, err)
		return 1
	}
	rwAppendLedger(rwDriveStateLedger(regDir), map[string]any{
		"ts": rwNowISO(), "session": sid, "state": string(st),
		"reason": strings.TrimSpace(*reason), "via": "fak resume hold",
	})
	if st == resume.DriveStopped {
		fmt.Fprintf(stdout, "held %s as stopped (terminal intent) — the resume watchdog will never auto-resume it; the hold is durable (it outlives the descriptor registry's 30-minute GC) and reversed only by `fak resume release %s`\n",
			shortID(sid), shortID(sid))
	} else {
		fmt.Fprintf(stdout, "held %s as %s — the resume watchdog will not auto-resume it; release with `fak resume release %s`\n",
			shortID(sid), st, shortID(sid))
	}
	return 0
}

// runResumeRelease reverses an operator hold by appending a `running` row to the drive-state
// store, then re-folds through the SAME pure leaf the watchdog reads (resume.FoldDriveStates)
// and reports the state that actually results — so it stays correct whether the fold treats a
// release as lifting every hold or a terminal stop as sticky. It is the operator's explicit
// reversal — the ONE thing that un-holds a session; the watchdog never does. Exit 0 released,
// 1 runtime, 2 usage, 3 the hold is terminal and still stands after the release.
func runResumeRelease(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume release", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "resume")
	regDirFlag := fs.String("reg-dir", "", "registry dir holding resume_drivestate.jsonl (default: the same regDir the watchdog resolves)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak resume release: need a <session-id> to release")
		return 2
	}
	sid := strings.TrimSpace(fs.Arg(0))
	regDir := resolveSweepRegDir(*regDirFlag)
	if err := os.MkdirAll(regDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "fak resume release: create registry dir %q: %v\n", regDir, err)
		return 1
	}
	rwAppendLedger(rwDriveStateLedger(regDir), map[string]any{
		"ts": rwNowISO(), "session": sid, "state": string(resume.DriveRunning), "via": "fak resume release",
	})
	// Report the state the watchdog will now read. If a hold survives the release, the store's
	// fold treats it as terminal — say so instead of falsely claiming a clear.
	if st := rwLoadDriveStates(regDir)[sid]; st.HeldByOperator() {
		fmt.Fprintf(stderr, "fak resume release: recorded a release for %s, but it is still held (%s) — the store treats this hold as terminal; remove its row from %s to fully clear it.\n",
			shortID(sid), st, rwDriveStateLedger(regDir))
		return 3
	}
	fmt.Fprintf(stdout, "released %s — the resume watchdog may auto-resume it again if it dies\n", shortID(sid))
	return 0
}

// normalizeHoldState maps an operator --state token to the closed drive-state a hold records.
// It accepts the imperative aliases (pause/drain/stop) as well as the state tokens, and refuses
// running/throttled (use `fak resume release` to clear a hold, not `hold --state running`).
func normalizeHoldState(s string) (resume.WatchdogDriveState, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "paused", "pause", "hold", "":
		return resume.DrivePaused, true
	case "draining", "drain":
		return resume.DriveDraining, true
	case "stopped", "stop":
		return resume.DriveStopped, true
	}
	return "", false
}

// renderResumeHolds prints (or emits as JSON) the current effective operator holds — the
// sessions the watchdog will not auto-resume. It folds the store to the latest state per
// session and keeps only the held ones, so a released session drops off the list.
func renderResumeHolds(stdout, stderr io.Writer, regDir string, asJSON bool) int {
	states := rwLoadDriveStates(regDir)
	type holdRow struct {
		Session string `json:"session"`
		State   string `json:"state"`
	}
	rows := make([]holdRow, 0, len(states))
	for sid, st := range states {
		if st.HeldByOperator() {
			rows = append(rows, holdRow{Session: sid, State: string(st)})
		}
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Session < rows[j].Session })
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":  "fak.resume-holds.v1",
			"reg_dir": regDir,
			"holds":   rows,
		}, "fak resume hold --list")
	}
	if len(rows) == 0 {
		fmt.Fprintf(stdout, "no operator holds in %s — the watchdog may auto-resume any dead session\n", rwDriveStateLedger(regDir))
		return 0
	}
	fmt.Fprintf(stdout, "operator holds (%s):\n", rwDriveStateLedger(regDir))
	for _, r := range rows {
		fmt.Fprintf(stdout, "  %-12s %s\n", shortID(r.Session), r.State)
	}
	fmt.Fprintln(stdout, "  release one with `fak resume release <session-id>`")
	return 0
}

func resumeUsage(w io.Writer) {
	fmt.Fprint(w, `fak resume — the deterministic RESUME-CACHE decision

  fak resume plan [--resident-tokens N] [--idle-seconds S] [--ttl 5m|1h]
                  [--horizon N] [--shed-budget N] [--seed-tokens N]
                  [--input-price F] [--output-price F] [--output-per-turn N]
                  [--image DIR] [--transcript FILE.jsonl] [--json]

  fak resume validate --corpus DIR [--ttl 5m|1h] [--max-files N] [--json]

  fak resume scan --store DIR [--ttl 5m|1h] [--horizon N] [--shed-budget N]
                  [--input-price F] [--output-price F] [--all] [--json]

  fak resume sweep [--window MIN] [--min-records N] [--bucket B] [--probe]
                   [--include-resumed] [--home DIR] [--reg-dir DIR] [--json]

  fak resume stopped [--window-h H] [--home DIR] [--json]

  fak resume dedup [--window-h H] [--home DIR] [--ledger FILE] [--apply] [--json]

  fak resume status --store DIR [--ledger FILE] [--max-attempts N] [--all] [--json]

  fak resume self [--session SID] [--store DIR] [--ledger FILE] [--max-attempts N] [--json]

  fak resume admit [--max-live N] [--max-per-window N] [--window-sec S]
                   [--min-spacing-sec S] [--ledger FILE] [--policy FILE]
                   [--json] [--quiet] [--explain]

  fak resume watchdog [--live] [--window-h H] [--max-per-tick N] [--max-attempts N]
                      [--spacing-sec S] [--probe MODE] [--reg-dir DIR] [--log-dir DIR]
                      [--no-refresh]

  fak resume hold <session-id> [--state paused|draining|stopped] [--reason S] [--reg-dir DIR]
  fak resume hold --list [--reg-dir DIR] [--json]
  fak resume release <session-id> [--reg-dir DIR]

  fak resume resolve <session-id> [--home DIR] [--cwd DIR] [--dry-run]
                     [--no-probe] [--wait] [--no-wait] [--json]

  fak resume identity <uuid|trace> [--reg-dir DIR] [--json]

  fak resume drive <uuid> [--reg-dir DIR] [--json]

  fak resume why <session-id> [--home DIR] [--ledger FILE] [--max-attempts N] [--json]

plan answers "I am resuming a long session — what happens to the prompt cache, and what
should I do?" It projects the cache posture (cold if idle exceeds the TTL, warm if not),
prices RESUME_FULL / CUT / RESET, and recommends a cut-by-default re-entry. Pure and
deterministic: same facts in, same priced verdict out.

admit is the PER-SOURCE concurrency gate a launcher self-gates on before it spawns a
"claude --resume": it counts the LIVE resume processes on this host across all accounts
(the dimension the server-side 529 burst wall keys on, which no per-tick / per-account
cap measured) plus the recent launch rate from the durable ledger, and returns ADMIT
(exit 0) or REFUSE (exit 3) with a structured reason. Gate a launch with:
  fak resume admit --quiet && claude --resume <sid> ...
The policy file (FAK_RESUME_SOURCE_POLICY, default .fak/resume-source-policy.json;
tracked template examples/resume-source-policy.example.json) fails OPEN when missing
but REFUSES (POLICY_MALFORMED, exit 3) when present-and-unparseable, so a typo can
never silently drop the rail. --explain prints the full posture — policy path+values,
ledger, live census, trailing-24h deferred/gate_fail_open rows — in one command.

validate back-tests that projection against billed reality: it scans a corpus of real
Claude Code transcripts, scores how often the cold/warm posture call agreed with the
provider's own cache_read / cache_creation records, and measures how exactly the cold-cost
premise held. The deterministic, observable answer to "is the cache-value call effective?".

resolve decides which account "claude --resume <sid>" should pin to: it locates the
owner (host-last, newest-mtime) across ~/.claude*, and — when the owner is throttled —
re-homes the transcript onto the least-loaded healthy Claude worker and pins there
(PIN / REHOME / PIN_BLOCKED). When the owner's reset is imminent (<= 15 min) the verdict
is WAIT_RESET instead — waiting for the owner beats copying the transcript onto another
loaded seat — and --wait turns that into behavior: sleep out the reset (narrated as a
countdown on stderr), re-resolve, and print the pin dir, so one command self-heals the
account wall. --no-wait forces the immediate re-home. stdout is the CLAUDE_CONFIG_DIR to
set; --dry-run decides without copying. The Go port of tools/resume_resolver.py.

identity resolves the join between the two session keyspaces the resume layer straddles: the
Claude Code transcript UUID (the id "claude --resume" takes) and the gateway / guard TRACE id
the operator control plane keys on. Give it either id and it prints the paired one — plus the
recorded handle/account/via provenance — from the durable, GC-immune resume_identity.jsonl store
(so a join survives the descriptor registry's 30-minute GC). Exit 4 on no join. It resolves the
identity only; it resumes nothing.

drive is the read-only operator lens onto the drive-CARRY channel: given a transcript UUID it
folds the same durable resume_drivestate.jsonl the watchdog reads and prints the drive-state a
relaunch would RESTORE for that uuid — the operator hold (if any) plus the carried budget /
objective — or a "would come up fresh" line when the store holds nothing for it. It is strictly
read-only: it never launches, admits, or writes a ledger row. --json emits the folded record.

why is the single-session narrative over the same folds: locate the session across every
~/.claude* account (no --store to know), read how it died from the transcript's own
records — including the mid-turn death that writes NO error record and used to vanish
from every readout — fold the resume journey from the ledger, read the owner account's
block/reset state, dry-run the resolve decision, and print the story with the one
command to run. The answer to "my session just died as a random error — what happened?".

scan walks a whole transcript store and finds the sessions that crashed on a rate limit
and never resumed — the ones that need a managed restart — then prints each one's cache
plan (cut/reset vs a cold full re-prefill). The detect-and-plan step before a restart: it
sizes each session from its last REAL model turn, so the synthetic rate-limit refusal that
ends a crashed session never mis-sizes it to zero.

sweep is the manifest-free DISCOVERY half of the cross-account resume layer (the Go port
of tools/resume_sweep.py): it walks every ~/.claude*/projects transcript touched in the
window, resolves each session's SUPERSET copy (uuid-set + last-ts, never file mtime), and
buckets it by the action it needs — LIMIT_RESET_PASSED / API_ERR (resumable now),
LIMIT_RESET_FUTURE (wait), AUTH (needs /login), LIVE (leave alone). The failure mode is
adjudicated off the error record only, never assistant prose. It resumes nothing.

stopped triages every recently-STOPPED top-level session by HOW it stopped (the Go port of
tools/stopped_sessions.py): a current synthetic limit banner (STOPPED_LIMIT), an auth wall
(STOPPED_AUTH), a tool_use that never got its result (STOPPED_MIDTOOL when there is evidence
the driver process is gone, else MIDTOOL_UNKNOWN — that same tail is what a driver still
inside a SLOW tool call leaves, so without evidence the row defers instead of being resumed
onto a live transcript, #5386), an
interruption, a parked background wait, a wrap-up, or a quiet stop — then decides
resume / defer / skip, deferring any session whose ACCOUNT is throttled. It resumes nothing.
Each row carries TWO independent axes (#3800): disp is ONLY the stop-cause; the dedup
verdict rides its own dup_of_live + live_sibling fields (a stopped duplicate of a live
sibling is skipped with its real cause preserved), and when one terminal turn carries both
an auth wall and a current limit banner, auth wins and the outranked limit is retained on
also_signals instead of being silently dropped.

status is the PROVE-THE-RESUME-TOOK runbook over the same store plus the durable resume
ledger. For every crashed-or-resumed session it folds one label (pending / launched /
took / re-stranded / gave-up / settled) read from the transcript's own turns, not the
launcher's "launched" ledger row (which alone cannot tell a resume that took from one
that silently no-op'd). Actionable sessions sort first, so an agent bringing a dead
batch back reads the ordered list, acts on the top, and re-runs.

self is the WORKER-FACING mirror of status: instead of a store-wide operator sweep, it
answers the first-person question one guarded session asks about ITSELF — "was I resumed,
did it take, will another attempt fire, or does a human own me now?" — folding the same
labels over this session's ledger history plus its own transcript. It is fail-closed: a
session with no resume history reads the honest floor (pending, nothing to recover), never
a fabricated "took". Same folds as status and the fak_resume_history MCP tool, so all three
agree. The session defaults to $CLAUDE_SESSION_ID, else the newest transcript in --store.

example (from inside a guarded session, ask what my own resume posture is):
  fak resume self --store ~/.claude/projects/<project>

watchdog is ONE TICK of the cross-account resume layer (the Go port of
tools/fleet_resume_watchdog.py, designed for a ~5-minute schedule and safe by hand):
refresh the session registry + AUTO_RESUME plan, then resume each planned session under
its owning account — gated by the self-resume guard, the worker policy, the outcome-aware
once-gate (a resume that died recoverably stays eligible up to the attempt cap; a clean
finish or an auth wall burns it), and the host-wide per-source admission. DRY-RUN by
default: pass --live (or FAK_LIVE=1) to actually spawn, capped per tick and paced between
spawns. Launches are recorded in the durable resume ledger BEFORE anything else.

hold / release align the automatic resume layer with the operator: "hold" records a durable
pause/drain/stop of a specific session so the watchdog will NOT auto-resume it, and "release"
reverses it. The hold outranks the watchdog's transcript-forensic resume (it beats even a
proven re-death), has no TTL (it survives the descriptor registry's 30-minute GC), and is
reversed ONLY by the operator, never by the watchdog. It is keyed by the Claude session id
"claude --resume" takes — the one key the operator surface and the watchdog share (the
"fak session pause/stop" control plane is keyed by the disjoint gateway trace, so it cannot
reach the watchdog on its own). --state stopped is terminal intent; the default paused is a
reversible hold. "hold --list" shows the current holds.

example (stop the watchdog from resurrecting a session you deliberately ended):
  fak resume hold <session-id> --state stopped --reason "superseded by #1234"

example (resume a 250k session idle 2h on a 5-minute cache):
  fak resume plan --resident-tokens 250000 --idle-seconds 7200

example (plan the resume of a REAL Claude Code session you are about to --resume):
  fak resume plan --transcript ~/.claude/projects/<ns>/<uuid>.jsonl

example (back-test the projection against your real session history):
  fak resume validate --corpus ~/.claude/projects

example (find the rate-limited crashes in a project and plan each managed restart):
  fak resume scan --store ~/.claude/projects/<project>

example (read where every crashed/resumed session stands and what still needs action):
  fak resume status --store ~/.claude/projects/<project>

example (one dead session — the full story and the exact command to bring it back):
  fak resume why <session-id>

example (resume one session, self-healing an imminent owner reset by waiting it out):
  CLAUDE_CONFIG_DIR="$(fak resume resolve -wait <sid>)" claude --resume <sid>
`)
}

func runResumeBackoff(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("resume backoff", flag.ContinueOnError)
	fs.SetOutput(stderr)
	session := fs.String("session", "", "session id")
	signature := fs.String("signature", "", "knownbad signature")
	historyPath := fs.String("history", "", "JSON event history")
	base := fs.Duration("base", time.Minute, "base delay")
	ceiling := fs.Duration("ceiling", time.Hour, "maximum delay")
	window := fs.Duration("window", time.Hour, "coalescing window")
	threshold := fs.Int("park-threshold", 3, "distinct sessions before parking")
	crashLoopBudget := fs.Int("crash-loop-budget", 3, "maximum attempts for an unchanged crash signature before quarantine")
	if !parseFlags(fs, args) || *session == "" || *signature == "" {
		fmt.Fprintln(stderr, "fak resume backoff: --session and --signature are required")
		return 2
	}
	var history []resumebackoff.Event
	if *historyPath != "" {
		b, err := os.ReadFile(*historyPath)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if err = json.Unmarshal(b, &history); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	d := resumebackoff.Decide(resumebackoff.Input{Session: *session, Signature: *signature, Now: time.Now().UTC(), History: history, Base: *base, Ceiling: *ceiling, Window: *window, ParkThreshold: *threshold, CrashLoopBudget: *crashLoopBudget})
	if err := json.NewEncoder(stdout).Encode(d); err != nil {
		return 1
	}
	if d.Eligible {
		return 0
	}
	return 3
}
