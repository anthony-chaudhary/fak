package codetools

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/envchild"
)

type bashEngine struct{ t *Toolset }

func (bashEngine) Caps() []abi.Capability { return nil }
func (bashEngine) WeightBearing() bool    { return false }
func (e bashEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, bad := e.t.bash(ctx, in)
	return result(ctx, c, in, out, bad, EngineBash), nil
}

type BashResult struct {
	Stdout          string `json:"stdout"`
	Stderr          string `json:"stderr"`
	ExitCode        int    `json:"exit_code"`
	TimedOut        bool   `json:"timed_out,omitempty"`
	StdoutTruncated bool   `json:"stdout_truncated,omitempty"`
	StderrTruncated bool   `json:"stderr_truncated,omitempty"`
}

type boundedBuffer struct {
	b         bytes.Buffer
	max       int
	truncated bool
}

func (w *boundedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	room := w.max - w.b.Len()
	if room > 0 {
		if room > len(p) {
			room = len(p)
		}
		_, _ = w.b.Write(p[:room])
	}
	if n > room {
		w.truncated = true
	}
	return n, nil
}
func (w *boundedBuffer) String() string { return w.b.String() }

func (t *Toolset) bash(ctx context.Context, body []byte) ([]byte, bool) {
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	var a BashArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if t.exactCommandsOnly && !t.exactCommandAllowed(a.Command) {
		return refuse(CodeCommandDeny, "command is outside the exact command allowlist").JSON(), true
	}
	if !t.exactCommandsOnly && t.focusedCommands && !(focusedCommandAllowed(a.Command) || t.exactCommandAllowed(a.Command)) {
		return refuse(CodeCommandDeny, "command is outside the focused coding allowlist").JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	cwd := t.root
	var cwdWarning string
	if a.Cwd != "" {
		resolved, r := t.resolve(a.Cwd)
		if r != nil {
			return r.JSON(), true
		}
		cwd = filepath.Clean(resolved.Abs)
		if fi, err := os.Stat(cwd); err != nil || !fi.IsDir() {
			if rootFi, rootErr := os.Stat(t.root); rootErr == nil && rootFi.IsDir() {
				cwdWarning = fmt.Sprintf("working directory %s does not exist; executed in %s\n", cwd, t.root)
				cwd = t.root
			} else {
				return refuse(CodeNotFound, fmt.Sprintf("DIRECTORY_NOT_FOUND: working directory %q does not exist", cwd)).JSON(), true
			}
		}
	} else if fi, err := os.Stat(cwd); err != nil || !fi.IsDir() {
		return refuse(CodeNotFound, fmt.Sprintf("DIRECTORY_NOT_FOUND: working directory %q does not exist", cwd)).JSON(), true
	}
	if runtime.GOOS == "windows" {
		// cmd.exe ships neither ls nor pwd, so these are served in-process there.
		if res, ok := t.nativeOrientation(a.Command, cwd); ok {
			res.Stderr = cwdWarning + res.Stderr
			return okJSON(res), false
		}
	}
	timeout := t.limits.MaxCommandTime
	if a.TimeoutMS > 0 && time.Duration(a.TimeoutMS)*time.Millisecond < timeout {
		timeout = time.Duration(a.TimeoutMS) * time.Millisecond
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	gotmpDir := t.ensureGoTmpDir()
	execCommand := a.Command
	if t.focusedCommands && focusedCommandAllowed(execCommand) {
		execCommand = hardenGitArgs(execCommand)
	}
	command := t.rewriteCommandForContainment(execCommand, cwd, gotmpDir)
	name, args := shellCommand(command)
	cmd := exec.CommandContext(runCtx, name, args...)
	cmd.Dir = cwd
	// #14450: the child starts from the allowlisted minimal host environment, so
	// gateway key variable names never reach it; writable Go paths are injected
	// explicitly rather than widening the inherited environment.
	childEnv := enforceGoTmpEnv(envchild.ChildEnv(os.Environ(), envchild.Options{Protected: t.protectedEnv}), gotmpDir)
	if runtime.GOOS == "windows" && isGoCommand(execCommand) {
		childEnv = enforceGoCacheEnv(childEnv, t.ensureWindowsGoCacheDir())
	}
	cmd.Env = noCwdExecutableEnv(childEnv)
	if isGitCommand(a.Command) {
		cmd.Env = gitHardenedEnv(cmd.Env)
	}
	var stdout, stderr boundedBuffer
	stdout.max = t.limits.MaxOutputBytes
	stderr.max = t.limits.MaxOutputBytes
	if cwdWarning != "" {
		_, _ = stderr.Write([]byte(cwdWarning))
	}
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	configureProcessTree(cmd)
	err := cmd.Start()
	if err == nil {
		err = cmd.Wait()
	}
	t.containmentCleanup(gotmpDir)
	timedOut := errors.Is(runCtx.Err(), context.DeadlineExceeded)
	exitCode := 0
	if err != nil {
		exitCode = -1
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			exitCode = ee.ExitCode()
		}
	}
	out := okJSON(BashResult{Stdout: stdout.String(), Stderr: stderr.String(), ExitCode: exitCode, TimedOut: timedOut, StdoutTruncated: stdout.truncated, StderrTruncated: stderr.truncated})
	if runCtx.Err() != nil && !timedOut && errors.Is(runCtx.Err(), context.Canceled) {
		return refuse(CodeCanceled, runCtx.Err().Error()).JSON(), true
	}
	return out, false
}

func shellCommand(command string) (string, []string) {
	if runtime.GOOS == "windows" {
		return "cmd.exe", []string{"/d", "/s", "/c", command}
	}
	return "/bin/sh", []string{"-c", command}
}

// An allowed command may be followed by read-only output filters joined with `|`.
// `2>&1` is the only redirect admitted, and only as a whole field, so stderr
// reaches the same pipe as stdout.
func focusedCommandAllowed(command string) bool {
	if strings.ContainsAny(command, "\r\n") {
		return false
	}
	if !strings.Contains(command, "|") {
		return focusedSimpleCommandAllowed(command)
	}

	// Admit only whitespace that the shell and the validator both tokenize.
	if strings.IndexFunc(command, func(r rune) bool { return r != '\t' && (r < 32 || r > 126) }) >= 0 {
		return false
	}
	segments := strings.Split(command, "|")
	producer := strings.Fields(segments[0])
	if len(producer) > 1 && producer[len(producer)-1] == "2>&1" {
		producer = producer[:len(producer)-1]
	}
	// ls and pwd have in-process implementations on Windows.
	if len(producer) == 0 || producer[0] == "ls" || producer[0] == "pwd" ||
		!focusedSimpleCommandAllowed(strings.Join(producer, " ")) {
		return false
	}
	for _, filter := range segments[1:] {
		if !readOnlyFilterAllowed(filter) {
			return false
		}
	}
	return true
}

// Only filters that cannot write files or read paths outside the pipe: no file
// operands, no pattern files, and no follow mode.
func readOnlyFilterAllowed(segment string) bool {
	fields := strings.Fields(segment)
	if len(fields) == 0 {
		return false
	}
	args := fields[1:]
	switch fields[0] {
	case "head", "tail":
		return lineCountArgsAllowed(args)
	case "wc":
		return len(args) == 1 && args[0] == "-l"
	case "grep":
		return grepPipeArgsAllowed(args)
	default:
		return false
	}
}

var lineCountFlagRe = regexp.MustCompile(`^-n?[0-9]{1,6}$`)

func lineCountArgsAllowed(args []string) bool {
	switch {
	case len(args) == 1:
		return lineCountFlagRe.MatchString(args[0])
	case len(args) == 2:
		return args[0] == "-n" && goDigitsValue(args[1])
	default:
		return false
	}
}

var (
	grepPipePatternRe = regexp.MustCompile(`^[A-Za-z0-9_:=/,-]+$`)
	grepPipeFlags     = map[string]bool{"-i": true, "-v": true, "-n": true, "-c": true, "-F": true, "-w": true}
)

// A single literal-looking pattern and no path operands: grep then reads only
// the piped output, never a file.
func grepPipeArgsAllowed(args []string) bool {
	pattern := ""
	for _, arg := range args {
		switch {
		case strings.HasPrefix(arg, "-"):
			if !grepPipeFlags[arg] {
				return false
			}
		case pattern != "" || !grepPipePatternRe.MatchString(arg):
			return false
		default:
			pattern = arg
		}
	}
	return pattern != ""
}

func focusedSimpleCommandAllowed(command string) bool {
	if strings.ContainsAny(command, ";&|<>`$%") {
		return false
	}
	fields := strings.Fields(command)
	if len(fields) == 0 {
		return false
	}
	for _, field := range fields {
		if strings.Trim(field, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=,@+-") != "" {
			return false
		}
	}
	args := fields[1:]
	switch fields[0] {
	case "pwd":
		return len(args) == 0
	case "ls":
		return orientationArgsAllowed(args)
	case "gofmt":
		return len(args) > 0 && gofmtArgsAllowed(args)
	case "go":
		if len(args) == 0 {
			return false
		}
		switch args[0] {
		case "version":
			return len(args) == 1
		case "build", "vet", "list", "test":
			return goToolchainArgsAllowed(args[1:])
		default:
			return false
		}
	case "git":
		if len(args) == 0 {
			return false
		}
		switch args[0] {
		case "diff":
			return gitArgsAllowed(args[1:])
		case "status":
			return flagsOnly(args[1:]) && gitArgsAllowed(args[1:])
		case "log":
			return flagsOnly(args[1:]) && gitArgsAllowed(args[1:])
		default:
			return false
		}
	default:
		return false
	}
}

// Paths are confined to the workspace by rejecting absolute, drive-qualified and
// parent-relative forms; the charset check above already excludes shell expansion.
func orientationArgsAllowed(args []string) bool {
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") && !workspaceRelativePath(arg) {
			return false
		}
	}
	return true
}

// Only the report modes -l and -d are admitted, with -s and -e as read-only
// modifiers; -w rewrites files, and every other flag is refused by default.
func gofmtArgsAllowed(args []string) bool {
	paths, report := 0, false
	for _, arg := range args {
		switch {
		case arg == "-l" || arg == "-d":
			report = true
		case arg == "-s" || arg == "-e":
		case strings.HasPrefix(arg, "-"):
			return false
		case workspaceRelativePath(arg):
			paths++
		default:
			return false
		}
	}
	return report && paths > 0
}

func workspaceRelativePath(arg string) bool {
	if strings.HasPrefix(arg, "/") || strings.Contains(arg, ":") {
		return false
	}
	for _, part := range strings.Split(arg, "/") {
		if part == ".." {
			return false
		}
	}
	return true
}

func flagsOnly(args []string) bool {
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") {
			return false
		}
	}
	return true
}

// splitFlag normalizes "--name=value", "-name" and "name" to their flag name and
// optional inline value; the go and git flag parsers accept every dash count.
func splitFlag(arg string) (name, value string, hasValue bool) {
	name, value, hasValue = strings.Cut(strings.TrimLeft(arg, "-"), "=")
	return name, value, hasValue
}

// The go toolchain must not write outside the build cache, load code or tooling
// from the workspace, run programs, or reach the module proxy. Go's flag package
// takes every dash count and a space-separated value, and a denylist of names
// keeps leaking (outputdir, cpuprofile, ldflags, compiler), so only the flags
// listed here are admitted and every other name is refused.
var goBoolFlags = map[string]bool{
	"v": true, "race": true, "short": true, "failfast": true, "json": true,
	"cover": true, "benchmem": true, "trimpath": true, "n": true, "e": true,
	"deps": true, "test": true, "find": true,
}

var goValueFlags = map[string]func(string) bool{
	"run": goRegexpValue, "skip": goRegexpValue, "bench": goRegexpValue, "list": goRegexpValue,
	"count": goDigitsValue, "parallel": goDigitsValue,
	"cpu": goDigitsCommasValue, "p": goDigitsCommasValue,
	"timeout": goDurationValue, "benchtime": goBenchtimeValue,
	"tags": goTagsValue, "covermode": goCovermodeValue,
	"vet": goVetValue, "shuffle": goShuffleValue,
}

var (
	goRegexpRe    = regexp.MustCompile(`^[A-Za-z0-9_./:=,+-]*$`)
	goDigitsRe    = regexp.MustCompile(`^[0-9]+$`)
	goDigitsCsvRe = regexp.MustCompile(`^[0-9]+(,[0-9]+)*$`)
	goDurationRe  = regexp.MustCompile(`^(?:[0-9]+(?:\.[0-9]+)?[a-z]+)+$`)
	goBenchtimeRe = regexp.MustCompile(`^[0-9]+(?:\.[0-9]+)?[a-z]*$`)
	goTagsRe      = regexp.MustCompile(`^[A-Za-z0-9_,.]+$`)
	goVetRe       = regexp.MustCompile(`^(?:off|[a-z]+(?:,[a-z]+)*)$`)
	goShuffleRe   = regexp.MustCompile(`^(?:on|off|[0-9]+)$`)
)

func goRegexpValue(v string) bool       { return goRegexpRe.MatchString(v) }
func goDigitsValue(v string) bool       { return goDigitsRe.MatchString(v) }
func goDigitsCommasValue(v string) bool { return goDigitsCsvRe.MatchString(v) }
func goDurationValue(v string) bool     { return goDurationRe.MatchString(v) }
func goBenchtimeValue(v string) bool    { return goBenchtimeRe.MatchString(v) }
func goTagsValue(v string) bool         { return goTagsRe.MatchString(v) }
func goVetValue(v string) bool          { return goVetRe.MatchString(v) }
func goShuffleValue(v string) bool      { return goShuffleRe.MatchString(v) }
func goCovermodeValue(v string) bool    { return v == "set" || v == "count" || v == "atomic" }

// Package arguments are workspace-relative patterns or plain import paths; a
// colon, @ or leading slash would name a path or module query outside the tree.
func goPackagePattern(arg string) bool {
	if arg == "" || strings.HasPrefix(arg, "/") {
		return false
	}
	for _, r := range arg {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("._-/", r)) {
			return false
		}
	}
	for _, part := range strings.Split(arg, "/") {
		if part == ".." {
			return false
		}
	}
	return true
}

func goToolchainArgsAllowed(args []string) bool {
	for _, arg := range args {
		if strings.Contains(arg, "@") {
			return false
		}
	}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "-") {
			if !goPackagePattern(arg) {
				return false
			}
			continue
		}
		name, value, hasValue := splitFlag(arg)
		if goBoolFlags[name] {
			if hasValue {
				return false
			}
			continue
		}
		check, ok := goValueFlags[name]
		if !ok {
			return false
		}
		if !hasValue {
			if i+1 >= len(args) {
				return false
			}
			i++
			value = args[i]
		}
		if !check(value) {
			return false
		}
	}
	return true
}

var deniedGitFlags = []string{"output", "ext-diff", "textconv", "no-index", "orderfile"}

// Git accepts unambiguous long-option prefixes, so --outp=C:/x writes a file
// just as --output does; the three-rune floor keeps --no-color and friends legal.
func hasDeniedGitFlag(args []string) bool {
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") {
			continue
		}
		// -O<file> attaches an orderfile path that the value check never sees.
		if strings.HasPrefix(arg, "-O") && !strings.HasPrefix(arg, "--") {
			return true
		}
		name, _, _ := splitFlag(arg)
		for _, denied := range deniedGitFlags {
			if name == denied || (len(name) >= 3 && strings.HasPrefix(denied, name)) {
				return true
			}
		}
	}
	return false
}

// Positional arguments and inline flag values name revisions or paths, so each
// must stay inside the workspace; git treats an argument after -- as a pathspec.
func gitArgsAllowed(args []string) bool {
	if hasDeniedGitFlag(args) {
		return false
	}
	endOfOpts := false
	for _, arg := range args {
		if arg == "--" {
			endOfOpts = true
			continue
		}
		if endOfOpts || !strings.HasPrefix(arg, "-") {
			if !workspaceRelativePath(arg) {
				return false
			}
			continue
		}
		if _, value, hasValue := splitFlag(arg); hasValue && !workspaceRelativePath(value) {
			return false
		}
	}
	return true
}

// diff and log run the repo's external diff and textconv drivers, which are
// programs named by repo config; the per-invocation flags switch them off.
func hardenGitArgs(command string) string {
	producer, filters, piped := strings.Cut(command, "|")
	fields := strings.Fields(producer)
	if len(fields) < 2 || fields[0] != "git" || (fields[1] != "diff" && fields[1] != "log") {
		return command
	}
	out := append([]string{"git", fields[1], "--no-ext-diff", "--no-textconv"}, fields[2:]...)
	hardened := strings.Join(out, " ")
	if piped {
		hardened += " |" + filters
	}
	return hardened
}

func isGitCommand(command string) bool {
	fields := strings.Fields(command)
	return len(fields) > 0 && fields[0] == "git"
}

func isGoCommand(command string) bool {
	fields := strings.Fields(command)
	return len(fields) > 0 && (strings.EqualFold(fields[0], "go") || strings.EqualFold(fields[0], "go.exe"))
}

// Git reads its pager, fsmonitor hook and diff driver from repo and environment
// config, which would run programs chosen by the workspace. The overrides sit at
// the highest config precedence, and any inherited copies are dropped first.
var gitHardeningConfig = [][2]string{
	{"diff.external", ""},
	{"core.fsmonitor", "false"},
	{"core.pager", "cat"},
}

func gitHardenedEnv(base []string) []string {
	out := make([]string, 0, len(base)+2*len(gitHardeningConfig)+2)
	for _, entry := range base {
		key, _, _ := strings.Cut(entry, "=")
		upper := strings.ToUpper(key)
		if upper == "GIT_EXTERNAL_DIFF" || upper == "GIT_PAGER" || upper == "GIT_CONFIG_COUNT" ||
			upper == "GIT_CONFIG_PARAMETERS" || strings.HasPrefix(upper, "GIT_CONFIG_KEY_") ||
			strings.HasPrefix(upper, "GIT_CONFIG_VALUE_") {
			continue
		}
		out = append(out, entry)
	}
	out = append(out, "GIT_PAGER=cat", fmt.Sprintf("GIT_CONFIG_COUNT=%d", len(gitHardeningConfig)))
	for i, kv := range gitHardeningConfig {
		out = append(out,
			fmt.Sprintf("GIT_CONFIG_KEY_%d=%s", i, kv[0]),
			fmt.Sprintf("GIT_CONFIG_VALUE_%d=%s", i, kv[1]))
	}
	return out
}

// cmd.exe searches the working directory before PATH unless told otherwise, so a
// workspace-planted go.bat or git.bat would shadow the real tool.
func noCwdExecutableEnv(base []string) []string {
	if runtime.GOOS != "windows" {
		return base
	}
	out := make([]string, 0, len(base)+1)
	for _, entry := range base {
		key, _, _ := strings.Cut(entry, "=")
		if strings.EqualFold(key, "NoDefaultCurrentDirectoryInExePath") {
			continue
		}
		out = append(out, entry)
	}
	return append(out, "NoDefaultCurrentDirectoryInExePath=1")
}

// nativeOrientation runs ls and pwd in-process on Windows. The result is built
// from the workspace root so it cannot reach outside it, and it cannot be
// shadowed by a workspace-planted executable. Unsupported ls options are
// refused rather than ignored.
func (t *Toolset) nativeOrientation(command, cwd string) (BashResult, bool) {
	fields := strings.Fields(command)
	if len(fields) == 0 {
		return BashResult{}, false
	}
	var stdout, stderr boundedBuffer
	stdout.max = t.limits.MaxOutputBytes
	stderr.max = t.limits.MaxOutputBytes
	exitCode := 0
	switch fields[0] {
	case "pwd":
		rel, err := filepath.Rel(t.root, cwd)
		if err != nil || rel == ".." || hasDotDotPrefix(rel) {
			return BashResult{}, false
		}
		fmt.Fprintln(&stdout, filepath.ToSlash(rel))
	case "ls":
		exitCode = t.lsNative(cwd, fields[1:], &stdout, &stderr)
	default:
		return BashResult{}, false
	}
	return BashResult{Stdout: stdout.String(), Stderr: stderr.String(), ExitCode: exitCode, StdoutTruncated: stdout.truncated, StderrTruncated: stderr.truncated}, true
}

func (t *Toolset) lsNative(cwd string, args []string, stdout, stderr io.Writer) int {
	long, all, endOfOpts := false, false, false
	var paths []string
	for _, arg := range args {
		if endOfOpts || !strings.HasPrefix(arg, "-") || arg == "-" {
			paths = append(paths, arg)
			continue
		}
		if arg == "--" {
			endOfOpts = true
			continue
		}
		for _, c := range arg[1:] {
			switch c {
			case 'l':
				long = true
			case 'a':
				all = true
			case '1':
			default:
				fmt.Fprintf(stderr, "ls: unsupported option -%c\n", c)
				return 2
			}
		}
	}
	if len(paths) == 0 {
		paths = []string{"."}
	}
	exitCode := 0
	for i, p := range paths {
		target := p
		if !filepath.IsAbs(p) {
			target = filepath.Join(cwd, p)
		}
		res, refused := t.resolve(target)
		if refused != nil {
			fmt.Fprintf(stderr, "ls: %s: %s\n", p, refused.Detail)
			exitCode = 2
			continue
		}
		info, err := os.Stat(res.Abs)
		if err != nil {
			fmt.Fprintf(stderr, "ls: %s: no such file or directory\n", p)
			exitCode = 2
			continue
		}
		if !info.IsDir() {
			writeLsEntry(stdout, long, p, info)
			continue
		}
		if len(paths) > 1 {
			if i > 0 {
				fmt.Fprintln(stdout)
			}
			fmt.Fprintf(stdout, "%s:\n", p)
		}
		entries, err := os.ReadDir(res.Abs)
		if err != nil {
			fmt.Fprintf(stderr, "ls: %s: %v\n", p, err)
			exitCode = 2
			continue
		}
		if all {
			writeLsEntry(stdout, long, ".", info)
			parentInfo := info
			if rel, err := filepath.Rel(t.root, filepath.Dir(res.Abs)); err == nil && !hasDotDotPrefix(rel) && rel != ".." {
				if parent, err := os.Stat(filepath.Dir(res.Abs)); err == nil {
					parentInfo = parent
				}
			}
			writeLsEntry(stdout, long, "..", parentInfo)
		}
		for _, e := range entries {
			if !all && strings.HasPrefix(e.Name(), ".") {
				continue
			}
			ei, err := e.Info()
			if err != nil {
				continue
			}
			writeLsEntry(stdout, long, e.Name(), ei)
		}
	}
	return exitCode
}

func writeLsEntry(w io.Writer, long bool, name string, info fs.FileInfo) {
	if !long {
		fmt.Fprintln(w, name)
		return
	}
	fmt.Fprintf(w, "%s %10d %s %s\n", info.Mode(), info.Size(), info.ModTime().Format("2006-01-02 15:04"), name)
}

func (t *Toolset) exactCommandAllowed(cmd string) bool {
	if strings.TrimSpace(cmd) == "" {
		return false
	}
	for _, allowed := range t.exactAllowedCommands {
		if cmd == allowed {
			return true
		}
	}
	return false
}
