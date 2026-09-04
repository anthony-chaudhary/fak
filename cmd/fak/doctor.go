package main

// fak doctor — a read-only operator diagnostic that wires the answer-shape WITNESS
// into actionable recommendations. It runs the degeneration/verbosity guard
// (internal/answershape) over a candidate answer/result AND cross-checks the real
// kernel admit verdict the context-MMU would reach on the same bytes
// (ctxmmu.ScreenBytes), then prints what to do about each finding. It is the fak
// analogue of `dos doctor`: the witness is the measurement, the doctor is the
// recommendation. Read-only, off the hot path, no session or gateway required.
//
// Exit 0 = healthy, 1 = at least one finding, 2 = usage error — so `fak doctor`
// also composes as a CI gate over a captured answer.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/answershape"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// Recommendation is one doctor finding: a check, its severity, what it found, and
// the operator action it recommends.
type Recommendation struct {
	Check     string `json:"check"`
	Severity  string `json:"severity"` // "ok" | "warn"
	Finding   string `json:"finding"`
	Recommend string `json:"recommend,omitempty"`
}

const (
	sevOK   = "ok"
	sevWarn = "warn"
)

// doctorReport is the structured result of a doctor run over one text.
type doctorReport struct {
	Shape           answershape.Report `json:"answer_shape"`
	KernelAdmit     string             `json:"kernel_admit"` // the abi.ReasonName the context-MMU would reach (or "none")
	KernelWouldHold bool               `json:"kernel_would_hold"`
	Recommendations []Recommendation   `json:"recommendations"`
	Findings        int                `json:"findings"`
}

func cmdDoctor(argv []string) {
	os.Exit(runDoctor(os.Stdin, os.Stdout, os.Stderr, argv))
}

// runDoctor is the testable core of `fak doctor`: it returns the exit code and
// takes explicit streams.
func runDoctor(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "binary" {
		argv = append([]string{"--binary"}, argv[1:]...)
	}
	if len(argv) > 0 && argv[0] == "codex-mcp-warning" {
		return runDoctorCodexMCPWarning(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "terminal-risk" {
		return runDoctorTerminalRisk(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "mcp" {
		return runDoctorMCP(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "movers" {
		return runDoctorMovers(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && (argv[0] == "launch-posture" || argv[0] == "posture") {
		return runDoctorLaunchPosture(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && (argv[0] == "defaults-selfcheck" || argv[0] == "defaults") {
		return runDoctorDefaultsSelfcheck(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "serve" {
		return runServeDoctor(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "trust" {
		return runDoctorTrust(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && (argv[0] == "telemetry" || argv[0] == "health") {
		return runDoctorTelemetry(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "doctor")
	text := fs.String("text", "", `text to diagnose, or "-" for stdin (default: stdin if neither --text nor --file is given)`)
	file := fs.String("file", "", "read the text from this file instead of --text/stdin")
	maxRepeat := fs.Float64("max-repeat", answershape.DefaultMaxRepeat, "largest in-shape repeat fraction (0..1); <=0 disables the repeat check")
	maxChars := fs.Int("max-chars", 0, "largest in-shape rune count; 0 disables the length check")
	ngram := fs.Int("ngram", answershape.DefaultNGram, "word n-gram width for the repeat metric")
	binary := fs.Bool("binary", false, "diagnose the running fak executable and sibling fak/fak.exe binaries for stale shadowing")
	asJSON := fs.Bool("json", false, "emit the doctor report as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak doctor: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	if *binary {
		exe, err := os.Executable()
		if err != nil {
			fmt.Fprintf(stderr, "fak doctor: resolve executable: %v\n", err)
			return 1
		}
		candidates := appversion.DefaultBinaryDoctorCandidates(exe)
		processes, processScanError := appversion.CollectBinaryProcesses(candidates)
		rep := appversion.DiagnoseBinaryWithProcesses(exe, candidates, processes, processScanError)

		// Layer the VCS-stamp freshness verdict onto the file/process-level report. The
		// sibling checks above catch a newer fak.exe sitting on disk, but NOT a binary built
		// from an old commit — or, worst of all, one carrying no VCS stamp at all, which can
		// never be checked for staleness and so must be flagged loudly rather than silently
		// passed (the exact hole that let a stale, unstamped guard run undetected).
		stampRec := runningStampFreshness(discoverRepoRoot())
		rep.Recommendations = append(rep.Recommendations, stampRec)
		if stampRec.Severity == appversion.SeverityWarn {
			rep.Findings++
		}

		// The sibling checks above only see fak/fak.exe next to the running exe. They are
		// blind to the most common "my fak is old / a different folder shows a different
		// version" failure: a stale, often UNSTAMPED fak early on PATH (e.g. ~/bin)
		// shadowing a fresh build later on PATH (~/go/bin or a checkout). Scan the whole
		// PATH in resolution order, read each binary's OWN embedded stamp, and judge the
		// one that actually wins when you type `fak`.
		rep.PathBinaries = scanPathBinaries(exe)
		pathRec := appversion.PathShadowRecommendation(rep.PathBinaries)
		rep.Recommendations = append(rep.Recommendations, pathRec)
		if pathRec.Severity == appversion.SeverityWarn {
			rep.Findings++
		}

		if *asJSON {
			b, _ := json.MarshalIndent(rep, "", "  ")
			fmt.Fprintln(stdout, string(b))
		} else {
			writeBinaryDoctorHuman(stdout, rep)
		}
		if rep.Findings > 0 {
			return 1
		}
		return 0
	}

	input, err := readShapeInput(*text, *file, stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak doctor: %v\n", err)
		return 2
	}

	rep := diagnose(input, answershape.Limits{MaxRepeat: *maxRepeat, MaxChars: *maxChars, NGram: *ngram})

	return renderJSONOrHuman(stdout, *asJSON, rep, writeDoctorHuman, rep.Findings > 0)
}

func writeBinaryDoctorHuman(w io.Writer, rep appversion.BinaryReport) {
	fmt.Fprintln(w, "== fak doctor: binary freshness ==")
	fmt.Fprintf(w, "executable: %s\n", rep.Executable)
	for _, img := range rep.Images {
		tag := "candidate"
		if img.Current {
			tag = "current"
		}
		if !img.Exists {
			fmt.Fprintf(w, "  [%s] %s (missing)\n", tag, img.Path)
			continue
		}
		suffix := ""
		if img.Newer {
			suffix = " newer-than-current"
		}
		fmt.Fprintf(w, "  [%s] %s size=%d sha=%s%s\n", tag, img.Path, img.Size, shortHash(img.SHA256), suffix)
	}
	if rep.ProcessScanError != "" {
		fmt.Fprintf(w, "processes: %s\n", rep.ProcessScanError)
	} else if len(rep.Processes) == 0 {
		fmt.Fprintln(w, "processes: no live candidate fak processes found")
	} else {
		fmt.Fprintln(w, "processes:")
		for _, p := range rep.Processes {
			tag := "candidate"
			if p.Current {
				tag = "current"
			}
			suffix := ""
			if !p.SameCurrent && p.SHA256 != "" {
				suffix = " different-from-current"
			}
			fmt.Fprintf(w, "  [%s] pid=%d %s sha=%s%s\n", tag, p.PID, p.Path, shortHash(p.SHA256), suffix)
		}
	}
	if len(rep.PathBinaries) > 0 {
		fmt.Fprintln(w, "fak on PATH (resolution order — rank 0 is what `fak` runs):")
		for _, b := range rep.PathBinaries {
			tag := "shadowed"
			if b.Winner {
				tag = "WINNER  "
			}
			fmt.Fprintf(w, "  [%s] #%d %s  %s\n", tag, b.Rank, b.Path, appversionPathAge(b))
		}
		fmt.Fprintln(w, "  note: `fak version` line 1 is read from the working directory's VERSION file, not the binary —")
		fmt.Fprintln(w, "        so the SAME binary prints different versions from different folders. The commit/date above is")
		fmt.Fprintln(w, "        the binary's own build stamp; that is the reliable identity.")
	}
	for _, r := range rep.Recommendations {
		tag := "OK  "
		if r.Severity == appversion.SeverityWarn {
			tag = "WARN"
		}
		fmt.Fprintf(w, "[%s] %-18s %s\n", tag, r.Check, r.Finding)
		if r.Recommend != "" {
			fmt.Fprintf(w, "       recommend: %s\n", r.Recommend)
		}
	}
	if rep.Findings == 0 {
		fmt.Fprintln(w, "doctor: healthy (0 findings)")
	} else {
		fmt.Fprintf(w, "doctor: %d finding(s)\n", rep.Findings)
	}
}

// appversionPathAge renders a PATH binary's identity for the human doctor line:
// its commit+date when stamped (with a +uncommitted marker), or an explicit
// "UNSTAMPED" tag plus the file mtime when it carries no VCS provenance — so an
// unstamped binary never reads as if it had a known commit. A read failure is
// surfaced rather than hidden.
func appversionPathAge(b appversion.PathBinary) string {
	if b.StampError != "" {
		return fmt.Sprintf("size=%d (stamp unreadable: %s)", b.Size, b.StampError)
	}
	if b.Stamped {
		short := b.Commit
		if len(short) > 12 {
			short = short[:12]
		}
		s := "commit=" + short
		if b.CommitTime != "" {
			s += " built=" + b.CommitTime
		}
		if b.Dirty {
			s += " +uncommitted"
		}
		return s
	}
	s := "UNSTAMPED (cannot attest commit)"
	if b.ModTime != "" {
		s += " file-mtime=" + b.ModTime
	}
	return s
}

// scanPathBinaries enumerates every fak/fak.exe resolvable on PATH, in resolution
// order, reading each one's OWN embedded build stamp. The stamp is what tells a
// stale binary from a current one: `fak version` line 1 is resolved from the
// working directory's VERSION file (appversion.Current), so it looks current even
// on an ancient binary — only the VCS stamp travels with the binary. The current
// exe is read in-process (no exec); every other candidate self-reports via
// `<path> version --json`, timed so a wedged binary cannot hang the doctor.
func scanPathBinaries(exe string) []appversion.PathBinary {
	names := []string{"fak"}
	if runtime.GOOS == "windows" {
		// PowerShell resolves `fak` to fak.exe, so the .exe is what actually runs; list it first.
		names = []string{"fak.exe", "fak"}
	}
	dirs := filepath.SplitList(os.Getenv("PATH"))
	probe := func(path string) (appversion.PathBinary, bool) {
		st, err := os.Stat(path)
		if err != nil || st.IsDir() {
			return appversion.PathBinary{}, false
		}
		e := appversion.PathBinary{
			Size:    st.Size(),
			ModTime: st.ModTime().UTC().Format(time.RFC3339),
		}
		id, ok := readBinaryIdentityStamp(path, exe)
		if !ok {
			e.StampError = "could not read `version --json`"
			return e, true
		}
		e.Commit = id.Commit
		e.CommitTime = id.CommitTime
		e.Dirty = id.Dirty
		e.Stamped = id.Stamped
		e.AppVersion = id.AppVersion
		return e, true
	}
	return appversion.ScanPathForFak(dirs, names, probe)
}

// pathIdentity is the subset of `fak version --json` scanPathBinaries needs.
type pathIdentity struct {
	AppVersion string `json:"app_version"`
	Commit     string `json:"commit"`
	CommitTime string `json:"commit_time"`
	Dirty      bool   `json:"dirty"`
	Stamped    bool   `json:"stamped"`
}

// readBinaryIdentityStamp returns the build identity of the fak binary at path.
// For the running exe it reads the in-process VCS stamp (reliable, no exec); for
// any other path it execs `<path> version --json` with a short timeout and parses
// the identity. A binary too old to emit `version --json`, or one that errors,
// yields ok=false so the caller records a read error instead of a false stamp.
func readBinaryIdentityStamp(path, exe string) (pathIdentity, bool) {
	absPath, _ := filepath.Abs(filepath.Clean(path))
	absExe, _ := filepath.Abs(filepath.Clean(exe))
	if strings.EqualFold(absPath, absExe) {
		s := binstamp.Self()
		id := pathIdentity{AppVersion: appversion.Current(), Commit: s.Revision, Dirty: s.Dirty, Stamped: s.HasVCS && s.Revision != ""}
		if bi, ok := debug.ReadBuildInfo(); ok {
			for _, kv := range bi.Settings {
				if kv.Key == "vcs.time" {
					id.CommitTime = kv.Value
				}
			}
		}
		return id, true
	}
	if out, ok := runBinaryVersion(path, "version", "--json"); ok {
		var id pathIdentity
		if json.Unmarshal([]byte(out), &id) == nil && (id.Stamped || strings.TrimSpace(id.AppVersion) != "") {
			return id, true
		}
	}
	// Fallback for binaries too old to support `version --json`: they print the human
	// `version` output regardless of the flag, so parse the "build:" line. This keeps a
	// stamped-but-ancient fak reporting its real commit + date instead of "unreadable".
	if out, ok := runBinaryVersion(path, "version"); ok {
		if id, ok := parseHumanVersion(out); ok {
			return id, true
		}
	}
	return pathIdentity{}, false
}

// runBinaryVersion execs `<path> <args…>` with a short timeout and returns stdout.
// A nonzero exit or timeout yields ok=false so the caller can fall back or record
// a read error rather than trust partial output.
func runBinaryVersion(path string, args ...string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, path, args...).Output()
	if err != nil {
		return "", false
	}
	return string(out), true
}

// parseHumanVersion extracts identity from the 3-line human `fak version` output
// ("<app-version>" / "build: <rev>[ +uncommitted]  (committed <time>)" / "go: …").
// It is the fallback path for binaries predating `version --json`. A "build:" line
// with no comparable revision ("module vX" / "(no VCS stamp …)") yields an
// unstamped-but-read identity so the app version is still reported.
func parseHumanVersion(out string) (pathIdentity, bool) {
	lines := strings.Split(out, "\n")
	if len(lines) == 0 {
		return pathIdentity{}, false
	}
	id := pathIdentity{AppVersion: strings.TrimSpace(lines[0])}
	sawBuild := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "build:") {
			continue
		}
		sawBuild = true
		rest := strings.TrimSpace(strings.TrimPrefix(line, "build:"))
		id.Dirty = strings.Contains(rest, "+uncommitted")
		if i := strings.Index(rest, "(committed "); i >= 0 {
			if j := strings.Index(rest[i:], ")"); j >= 0 {
				id.CommitTime = strings.TrimSpace(rest[i+len("(committed ") : i+j])
			}
		}
		fields := strings.Fields(rest)
		if len(fields) > 0 {
			rev := fields[0]
			if rev != "" && rev != "module" && rev != "(no" {
				id.Commit = rev
				id.Stamped = true
			}
		}
		break
	}
	if !sawBuild && strings.TrimSpace(id.AppVersion) == "" {
		return pathIdentity{}, false
	}
	return id, true
}

func shortHash(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	if h == "" {
		return "-"
	}
	return h
}

// runningStampFreshness reads THIS binary's embedded VCS stamp and judges it against the
// repo's converged line (origin/main, falling back to local HEAD when origin is not
// fetched), returning the recommendation for `fak doctor --binary`. It is read-only and does
// no network fetch — a diagnostic must not mutate repo state. When run outside a repo the
// verdict is a benign "not checked" (CauseNoHead), never a false alarm.
func runningStampFreshness(repoRoot string) appversion.BinaryRecommendation {
	stamp := binstamp.Self()
	headRev, headRef := "", ""
	if repoRoot != "" {
		if rev := repoRevOf(repoRoot, "origin/main"); rev != "" {
			headRev, headRef = rev, "origin/main"
		} else if rev := repoRevOf(repoRoot, "HEAD"); rev != "" {
			headRev, headRef = rev, "HEAD"
		}
	}
	verdict, cause := binstamp.Explain(stamp, headRev)
	return stampFreshnessRecommendation(verdict, cause, stamp.Revision, headRev, headRef)
}

// stampFreshnessRecommendation is the pure mapping from a binstamp verdict to an operator
// recommendation. The two WARN cases are the ones that mean "the fak you are running is not
// the one you think": an old commit (CauseDiverged) and — the load-bearing one — a binary
// with no VCS stamp at all (CauseUnstamped), which cannot be checked for staleness and so is
// itself the defect. The remaining causes are informational (an OK line, not a finding): a
// dev's dirty build and running outside a repo are both legitimately uncheckable.
func stampFreshnessRecommendation(verdict binstamp.Freshness, cause binstamp.Cause, runningRev, headRev, headRef string) appversion.BinaryRecommendation {
	const check = "binary-vcs-stamp"
	short := func(s string) string {
		s = strings.TrimSpace(s)
		if len(s) > 12 {
			return s[:12]
		}
		if s == "" {
			return "(none)"
		}
		return s
	}
	switch cause {
	case binstamp.CauseUnstamped:
		return appversion.BinaryRecommendation{
			Check:    check,
			Severity: appversion.SeverityWarn,
			Finding:  "this fak carries no embedded VCS stamp — it cannot attest which commit it was built from, so staleness is UNVERIFIABLE",
			Recommend: "rebuild with VCS stamping so the running binary is checkable: a plain `go build ./cmd/fak` " +
				"inside the repo stamps the commit, while `-buildvcs=false`, `go install …@version`, or a build " +
				"where git is unavailable strip it. `fak self-update --force` installs a stamped origin/main build.",
		}
	case binstamp.CauseDiverged: // Stale — a clean commit that differs from HEAD.
		return appversion.BinaryRecommendation{
			Check:     check,
			Severity:  appversion.SeverityWarn,
			Finding:   fmt.Sprintf("running build %s is a different commit than %s (%s) — a newer fak exists", short(runningRev), short(headRev), headRef),
			Recommend: "run `fak self-update` to build+gate+install the current " + headRef + " over this binary.",
		}
	case binstamp.CauseDirty:
		return appversion.BinaryRecommendation{
			Check:    check,
			Severity: appversion.SeverityOK,
			Finding:  fmt.Sprintf("built from a dirty tree (base commit %s) — a dev build; staleness not checked", short(runningRev)),
		}
	case binstamp.CauseNoHead:
		return appversion.BinaryRecommendation{
			Check:    check,
			Severity: appversion.SeverityOK,
			Finding:  fmt.Sprintf("build %s present, but no HEAD to compare against (not a repo / unreadable) — staleness not checked", short(runningRev)),
		}
	default: // CauseMatched — Fresh.
		return appversion.BinaryRecommendation{
			Check:    check,
			Severity: appversion.SeverityOK,
			Finding:  fmt.Sprintf("build %s matches %s (%s) — current", short(runningRev), short(headRev), headRef),
		}
	}
}

// diagnose runs both checks over input and assembles the recommendations. It is
// pure (no I/O), so tests can assert the recommendation set directly.
func diagnose(input []byte, lim answershape.Limits) doctorReport {
	shape := answershape.Measure(input, lim)

	// Cross-check the real kernel admit rung: would the context-MMU hold these bytes
	// out of model context at write time? ScreenBytes is the shipped predicate the
	// gate uses, so the doctor reports the gate's actual decision, not a parallel one.
	reason, wouldHold := ctxmmu.ScreenBytes(input)

	rep := doctorReport{
		Shape:           shape,
		KernelAdmit:     abi.ReasonName(reason),
		KernelWouldHold: wouldHold,
	}

	if shape.Degenerate {
		rep.Recommendations = append(rep.Recommendations, Recommendation{
			Check:    "answer-shape",
			Severity: sevWarn,
			Finding:  joinReasons(shape),
			Recommend: "the answer is degenerate (looping or runaway). Cap output tokens, lower " +
				"temperature, or exclude this result from context. The in-kernel enforcement is " +
				"the context-MMU repeat/oversize admit rung (internal/ctxmmu).",
		})
	} else {
		rep.Recommendations = append(rep.Recommendations, Recommendation{
			Check:    "answer-shape",
			Severity: sevOK,
			Finding:  fmt.Sprintf("in shape (repeat %.2f <= %.2f, %d chars)", shape.RepeatFraction, lim.MaxRepeat, shape.Chars),
		})
	}

	if wouldHold {
		rep.Recommendations = append(rep.Recommendations, Recommendation{
			Check:    "kernel-admit",
			Severity: sevWarn,
			Finding:  fmt.Sprintf("the context-MMU would QUARANTINE this result (%s)", abi.ReasonName(reason)),
			Recommend: "at write time the kernel would hold these bytes out of model-visible context — " +
				"treat the result as poison/secret/pollution, not a normal answer.",
		})
	} else {
		rep.Recommendations = append(rep.Recommendations, Recommendation{
			Check:    "kernel-admit",
			Severity: sevOK,
			Finding:  "the context-MMU would admit this result (no secret/injection/pollution screen)",
		})
	}

	for _, r := range rep.Recommendations {
		if r.Severity == sevWarn {
			rep.Findings++
		}
	}
	return rep
}

func writeDoctorHuman(w io.Writer, rep doctorReport) {
	fmt.Fprintln(w, "== fak doctor: answer health ==")
	for _, r := range rep.Recommendations {
		tag := "OK  "
		if r.Severity == sevWarn {
			tag = "WARN"
		}
		fmt.Fprintf(w, "[%s] %-12s %s\n", tag, r.Check, r.Finding)
		if r.Recommend != "" {
			fmt.Fprintf(w, "       recommend: %s\n", r.Recommend)
		}
	}
	if rep.Findings == 0 {
		fmt.Fprintln(w, "doctor: healthy (0 findings)")
	} else {
		fmt.Fprintf(w, "doctor: %d finding(s)\n", rep.Findings)
	}
}

type float64SliceFlag []float64

func (f *float64SliceFlag) String() string {
	if f == nil || len(*f) == 0 {
		return ""
	}
	parts := make([]string, len(*f))
	for i, v := range *f {
		parts[i] = fmt.Sprintf("%g", v)
	}
	return strings.Join(parts, ",")
}

func (f *float64SliceFlag) Set(v string) error {
	for _, part := range strings.Split(v, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		val, err := strconv.ParseFloat(part, 64)
		if err != nil {
			return fmt.Errorf("invalid float value: %w", err)
		}
		*f = append(*f, val)
	}
	return nil
}

func runDoctorTelemetry(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak doctor telemetry", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dbFlag := fs.String("db", "", "path to SQLite DB (default: auto-discover opencode.db)")
	promptTokens := fs.Int("prompt-tokens", 0, "observed prompt tokens in turn")
	baselineTokens := fs.Int("baseline-tokens", 0, "baseline prompt tokens")
	var latencies float64SliceFlag
	fs.Var(&latencies, "latency", "turn latency in seconds (repeatable or comma-separated)")
	asJSON := fs.Bool("json", false, "emit JSON output")

	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak doctor telemetry: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	dbPath := *dbFlag
	if dbPath == "" {
		dbPath = discoverOpencodeDB()
	}

	report := trajectory.EvaluateTelemetryHealth(*promptTokens, *baselineTokens, latencies, dbPath)

	return renderJSONOrHuman(stdout, *asJSON, report, writeDoctorTelemetryHuman, report.Findings > 0)
}

func discoverOpencodeDB() string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		p := filepath.Join(home, ".local", "share", "opencode", "opencode.db")
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p
		}
	}
	if userProfile := os.Getenv("USERPROFILE"); userProfile != "" {
		p := filepath.Join(userProfile, ".local", "share", "opencode", "opencode.db")
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p
		}
	}
	return ""
}

func writeDoctorTelemetryHuman(w io.Writer, rep trajectory.TelemetryHealthReport) {
	fmt.Fprintln(w, "== fak doctor: telemetry health ==")
	if rep.PromptTokens > 0 || rep.BaselinePrompt > 0 {
		fmt.Fprintf(w, "prompt tokens: %d (baseline: %d)\n", rep.PromptTokens, rep.BaselinePrompt)
	}
	if rep.CurrentLatency > 0 || rep.MedianLatency > 0 {
		fmt.Fprintf(w, "latency: current=%.2fs median=%.2fs\n", rep.CurrentLatency, rep.MedianLatency)
	}
	if rep.DBPath != "" {
		fmt.Fprintf(w, "database: %s (size=%d bytes, pages=%d, freelist=%d)\n", rep.DBPath, rep.DBBytes, rep.PageCount, rep.FreelistPages)
	}
	for _, alarm := range rep.Alarms {
		tag := "OK  "
		if alarm.Severity == trajectory.SeverityWarn {
			tag = "WARN"
		}
		fmt.Fprintf(w, "[%s] %-18s %s\n", tag, alarm.Type, alarm.Message)
		if alarm.Detail != "" {
			fmt.Fprintf(w, "       detail: %s\n", alarm.Detail)
		}
	}
	if rep.Findings == 0 {
		fmt.Fprintln(w, "doctor: healthy (0 findings)")
	} else {
		fmt.Fprintf(w, "doctor: %d finding(s)\n", rep.Findings)
	}
}
