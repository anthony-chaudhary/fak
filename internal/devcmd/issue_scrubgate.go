package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// issueScrubGateModuleRoot is the fak module root the sibling-locator starts from.
// Package var: a test can point it at its own tree instead of the developer's.
// The default is resolved -trimpath-safely (see issueScrubGateDefaultFakModuleRoot):
// it never relies on a compiled-in source path, which -trimpath rewrites to a
// module-qualified string.
var issueScrubGateModuleRoot = issueScrubGateDefaultFakModuleRoot()

// issueScrubGateCwd is the working directory the trimpath-safe private-companion
// walk starts from. Package var: a test can point it at an isolated tree.
var issueScrubGateCwd = func() string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	return cwd
}()

// issueScrubGateDefaultFakModuleRoot resolves the fak module root without relying on
// a compiled-in source path. -trimpath rewrites runtime.Caller paths to
// module-qualified strings, so the primary source is the filesystem: walk up from
// the process cwd to the directory whose go.mod declares the fak module. The legacy
// runtime.Caller value is accepted only when it names a real on-disk directory.
func issueScrubGateDefaultFakModuleRoot() string {
	cwd, err := os.Getwd()
	if err == nil {
		if root, ok := issueScrubGateWalkUpForFakModuleRoot(cwd); ok {
			return root
		}
	}
	if root := issueScrubGateModuleRootFromCaller(); root != "" {
		if info, err := os.Stat(root); err == nil && info.IsDir() {
			return root
		}
	}
	return ""
}

// issueScrubGateModuleRootFromCaller is the legacy runtime.Caller resolution. Under
// -trimpath the recorded path is module-qualified, so the derived root is a module
// path, not a directory; callers must verify it exists on disk before using it.
func issueScrubGateModuleRootFromCaller() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	// The file lives at <fakRoot>/internal/devcmd/, so the module root is its
	// directory's parent's parent: devcmd -> internal -> <fakRoot>.
	dir := filepath.Dir(file)
	return filepath.Dir(filepath.Dir(dir))
}

// issueScrubGateWalkUpForFakModuleRoot walks up from startDir to the nearest
// directory whose go.mod declares the fak module path. Trimpath-safe: pure
// filesystem, no compiled-in paths.
func issueScrubGateWalkUpForFakModuleRoot(startDir string) (string, bool) {
	dir := startDir
	for {
		if issueScrubGateIsFakModuleRoot(dir) {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

// issueScrubGateIsFakModuleRoot reports whether dir is the root of the fak module:
// it holds a go.mod whose module line is the fak module path.
func issueScrubGateIsFakModuleRoot(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, "go.mod"))
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "module ") {
			return strings.TrimSpace(strings.TrimPrefix(line, "module ")) == "github.com/anthony-chaudhary/fak"
		}
	}
	return false
}

// issueScrubGateScriptName is the scrubber the gate shells out to. Public fak
// deliberately ships NO copy of it (redacted-placeholders policy: the REAL needle
// tables stay in the private-guard home), so the gate must be told where the
// private companion repo is.
const issueScrubGateScriptName = "tools/issue_scrub.py"

// issueScrubGateStdin is what the scrubber audits for each create: title text plus
// body text. The scrubber reads one stdin stream, so the title is prepended —
// needles in a title must refuse creation the same as needles in the body.
func issueScrubGateStdin(title, body string) string {
	return title + "\n" + body
}

// issueScrubGateVerdict is the internal-shape result of one scrub-gate run.
type issueScrubGateVerdict struct {
	// Ran is true only when the python interpreter launched, the scrubber emitted
	// the value-free JSON result-contract shape, and its exit code was 0 or 1.
	Ran bool
	// Clean is the scrubber's own verdict: the cleaned text equals the raw text
	// AND no needle survived the scrub. Publish only when Clean is true.
	Clean bool
	// NeedsScrub mirrors the JSON needs_scrub field (scrub would change the text).
	NeedsScrub bool
	// Replacements is the scrubbed-out needle replacement count (value-free).
	Replacements int
	// SurvivorCount is the post-scrub needle survivor count (value-free).
	SurvivorCount int
	// Path is the scrubber script actually used, for the refusal message.
	Path string
	// Err is set when Ran is false; it names probed paths / the failure shape and
	// NEVER contains issue text or scrub stdout/stderr (values could leak).
	Err string
}

// issueScrubGateLocate resolves the private companion root: explicit argument
// first, then env overrides, then the trimpath-safe walk up from the process cwd
// to the nearest private companion (the scrubber lives in the checkout the caller
// is standing in), then the fak-private sibling of the fak module root. Env order:
// FAK_PRIVATE_DIR (the issue-#931 spelling) wins over FAK_PRIVATE_ROOT (the
// pre-existing debtlane/factory spelling).
func issueScrubGateLocate(privateRoot string) string {
	privateRoot = strings.TrimSpace(privateRoot)
	if privateRoot != "" {
		if info, err := os.Stat(privateRoot); err == nil && info.IsDir() {
			return privateRoot
		}
	}
	for _, env := range []string{"FAK_PRIVATE_DIR", "FAK_PRIVATE_ROOT"} {
		if dir := os.Getenv(env); dir != "" {
			if info, err := os.Stat(dir); err == nil && info.IsDir() {
				return dir
			}
		}
	}
	// Trimpath-safe: the scrubber lives in the checkout the caller is standing in.
	// Walk up from the process cwd to the nearest private companion, independent of
	// any compiled-in source path.
	if cwd := issueScrubGateCwd; cwd != "" {
		if root := issueScrubGateLocatePrivateFromCwd(cwd); root != "" {
			return root
		}
	}
	if issueScrubGateModuleRoot != "" {
		candidate := filepath.Join(filepath.Dir(issueScrubGateModuleRoot), "fak-private")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
	}
	return ""
}

// issueScrubGatePrivateCandidates walks up from startDir and returns the private-root
// candidate script paths in check order: at each ancestor, <ancestor>/tools/issue_scrub.py
// (the caller is standing in the private repo) then <ancestor>/fak-private/tools/issue_scrub.py
// (the caller is standing in the public repo or a workspace parent holding both).
// Trimpath-safe: pure filesystem, no compiled-in paths. Bounded to 16 ancestors so the
// refusal list stays finite on deep paths.
func issueScrubGatePrivateCandidates(startDir string) []string {
	var candidates []string
	dir := startDir
	for i := 0; ; i++ {
		candidates = append(candidates,
			filepath.Join(dir, issueScrubGateScriptName),
			filepath.Join(dir, "fak-private", issueScrubGateScriptName))
		parent := filepath.Dir(dir)
		if parent == dir || i >= 16 {
			break
		}
		dir = parent
	}
	return candidates
}

// issueScrubGateLocatePrivateFromCwd returns the private companion root nearest to
// startDir: the first candidate whose script exists on disk. The private root is the
// directory that CONTAINS the tools/ dir (two levels above the script path).
func issueScrubGateLocatePrivateFromCwd(startDir string) string {
	for _, c := range issueScrubGatePrivateCandidates(startDir) {
		if info, err := os.Stat(c); err == nil && !info.IsDir() {
			return filepath.Dir(filepath.Dir(c))
		}
	}
	return ""
}

// issueScrubGateProbe returns every path the gate looked for the scrubber at, in
// precedence order — the fail-closed refusal message names each one so the operator
// can install/point the gate at the right tree instead of guessing. Every named path
// is a real filesystem path: the module-root sibling candidate is named only when the
// module root is a real directory, never a -trimpath module-qualified path.
func issueScrubGateProbe(privateRoot string) []string {
	var probed []string
	if root := strings.TrimSpace(privateRoot); root != "" {
		probed = append(probed, filepath.Join(root, issueScrubGateScriptName))
	}
	for _, env := range []string{"FAK_PRIVATE_DIR", "FAK_PRIVATE_ROOT"} {
		if dir := os.Getenv(env); dir != "" {
			probed = append(probed, filepath.Join(dir, issueScrubGateScriptName))
		}
	}
	if cwd := issueScrubGateCwd; cwd != "" {
		probed = append(probed, issueScrubGatePrivateCandidates(cwd)...)
	}
	if issueScrubGateModuleRoot != "" {
		if info, err := os.Stat(issueScrubGateModuleRoot); err == nil && info.IsDir() {
			probed = append(probed, filepath.Join(
				filepath.Dir(issueScrubGateModuleRoot), "fak-private", issueScrubGateScriptName))
		}
	}
	if len(probed) == 0 {
		probed = append(probed, "<no private root located: pass -private-root or set FAK_PRIVATE_DIR/FAK_PRIVATE_ROOT>")
	}
	return probed
}

// issueScrubGateRefusalPaths renders the probed list for the refusal line: private
// needles never appear here (paths only), and dedup keeps identical candidates from
// printing twice when an env override selects an already-probed default.
func issueScrubGateRefusalPaths(probed []string) string {
	seen := make(map[string]bool, len(probed))
	unique := make([]string, 0, len(probed))
	for _, p := range probed {
		p = strings.TrimSpace(p)
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		unique = append(unique, p)
	}
	if len(unique) == 0 {
		return "(no paths probed)"
	}
	return strings.Join(unique, "; ")
}

// issueScrubGateJsonResult is ONLY the value-free fields the scrubber's --check
// emit contract guarantees. Unknown JSON keys (or anything else on stdout) must
// refuse: a modified scrubber could smuggle needle values into stdout, and the
// gate would then be echoing them through its own diagnostics.
type issueScrubGateJsonResult struct {
	Clean         bool    `json:"clean"`
	NeedsScrub    bool    `json:"needs_scrub"`
	Replacements  int     `json:"replacements"`
	SurvivorCount int     `json:"survivor_count"`
	Audited       *bool   `json:"audited,omitempty"`
	Refused       *string `json:"refused,omitempty"`
}

// issueScrubGateResultIsKnownShape rejects any JSON object carrying a field
// outside the two documented --check contracts. Legacy producers emit four
// fields; newer producers add audited=true, or an explicit empty-input refusal.
// Unknown keys and unaudited success are never accepted or echoed.
func issueScrubGateResultIsKnownShape(raw []byte) bool {
	// Decode fields explicitly: map unmarshalling silently keeps the last value
	// of duplicate keys, which could hide an earlier audited=false declaration.
	dec := json.NewDecoder(bytes.NewReader(raw))
	opening, err := dec.Token()
	if err != nil || opening != json.Delim('{') {
		return false
	}
	fields := make(map[string]json.RawMessage)
	for dec.More() {
		token, err := dec.Token()
		key, ok := token.(string)
		if err != nil || !ok {
			return false
		}
		if _, duplicate := fields[key]; duplicate {
			return false
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return false
		}
		fields[key] = value
	}
	if closing, err := dec.Token(); err != nil || closing != json.Delim('}') {
		return false
	}
	if _, err := dec.Token(); err != io.EOF {
		return false
	}
	if len(fields) < 4 || len(fields) > 6 {
		return false
	}
	for _, key := range []string{"clean", "needs_scrub", "replacements", "survivor_count"} {
		if value, ok := fields[key]; !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return false
		}
	}
	switch len(fields) {
	case 4:
		return true // Legacy contract; the caller independently refuses empty input.
	case 5:
		return bytes.Equal(bytes.TrimSpace(fields["audited"]), []byte("true"))
	case 6:
		var refused string
		return bytes.Equal(bytes.TrimSpace(fields["audited"]), []byte("false")) &&
			json.Unmarshal(fields["refused"], &refused) == nil && refused == "empty-input"
	}
	return false
}

// issueScrubGateHook is the process-wide injection seam behind
// issueCreateRunScrubGate. The shipped implementation runs the real sibling
// python scrubber; tests override it to exercise verdict handling without an
// interpreter. A nil hook selects the real gate.
var issueScrubGateHook func(privateRoot, title, body string) issueScrubGateVerdict

// issueCreateRunScrubGate runs the gate through issueScrubGateHook so the
// runIssueCreateWith wiring and the subprocess gate stay independently testable.
func issueCreateRunScrubGate(privateRoot, title, body string) issueScrubGateVerdict {
	if issueScrubGateHook != nil {
		return issueScrubGateHook(privateRoot, title, body)
	}
	return runIssueScrubGateSubprocess(privateRoot, title, body)
}

// runIssueScrubGateSubprocess is the production arm: locate the scrubber, run
// python issue_scrub.py --check over title+body, and return the value-free verdict.
// The scrubber's --check reads stdin and emits only a value-free JSON result on
// stdout; the gate parses that shape (never the needle tables, never the python
// API) and refuses fail-closed when the scrubber cannot be located or runs.
//
// Refusal posture: publish only when the scrubber reports clean (no survivors AND
// scrub would not change the text). A needs_scrub-without-survivors body still
// refuses: it carries private-adjacent material the scrubber would silently mutate,
// so publishing might leak something the needle scan cannot see.
func runIssueScrubGateSubprocess(privateRoot, title, body string) issueScrubGateVerdict {
	// Enforce anti-vacuity locally as well, including for legacy producers.
	if strings.TrimFunc(issueScrubGateStdin(title, body), func(r rune) bool {
		// Python str.strip also includes the four information separators.
		return unicode.IsSpace(r) || (r >= '\u001c' && r <= '\u001f')
	}) == "" {
		return issueScrubGateVerdict{Err: "issue_scrub.py --check refused empty input"}
	}
	var probed []string
	script, root := "", issueScrubGateLocate(privateRoot)
	if root != "" {
		script = filepath.Join(root, issueScrubGateScriptName)
		if info, err := os.Stat(script); err != nil || info.IsDir() {
			script = ""
		}
	}
	probed = issueScrubGateProbe(privateRoot)
	if script == "" {
		return issueScrubGateVerdict{
			Err: "issue_scrub.py not found (probed: " + issueScrubGateRefusalPaths(probed) + ")",
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "python", script, "--check")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Stdin = strings.NewReader(issueScrubGateStdin(title, body))
	if runErr := cmd.Run(); runErr != nil {
		// Detail is scrubber/interpreter stderr — never echoed into Err: it can
		// contain the needle values the scrub refused to redact (exit-1 arms).
		if ctx.Err() == context.DeadlineExceeded {
			return issueScrubGateVerdict{Path: script,
				Err: "python issue_scrub.py --check timed out after 20s"}
		}
		if exitErr, isExit := runErr.(*exec.ExitError); isExit {
			// Exit 1 is an EXPECTED scrubber outcome (survivors or needs_scrub);
			// the value-free JSON verdict still carries the counts. Only an
			// unusual exit refuses here; the JSON shape gate below covers the rest.
			if exit := exitErr.ExitCode(); exit != 0 && exit != 1 {
				return issueScrubGateVerdict{Path: script,
					Err: fmt.Sprintf("python issue_scrub.py --check exited with unusual code %d", exit)}
			}
		} else {
			return issueScrubGateVerdict{Path: script,
				Err: "python issue_scrub.py --check failed"}
		}
	} else if exit := cmd.ProcessState.ExitCode(); exit != 0 && exit != 1 {
		return issueScrubGateVerdict{Path: script,
			Err: fmt.Sprintf("python issue_scrub.py --check exited with unusual code %d", exit)}
	}
	out := stdout.Bytes()
	if !issueScrubGateResultIsKnownShape(out) {
		return issueScrubGateVerdict{Path: script,
			Err: "issue_scrub.py --check stdout is not the documented value-free JSON shape (refusing without echoing stdout)"}
	}
	var result issueScrubGateJsonResult
	dec := json.NewDecoder(bytes.NewReader(out))
	if err := dec.Decode(&result); err != nil {
		return issueScrubGateVerdict{Path: script, Err: "issue_scrub.py --check returned invalid field types"}
	}
	if result.Refused != nil || (result.Audited != nil && !*result.Audited) {
		return issueScrubGateVerdict{Path: script, Err: "issue_scrub.py --check refused unaudited input"}
	}
	if result.Clean && (cmd.ProcessState.ExitCode() != 0 || result.NeedsScrub || result.Replacements != 0 || result.SurvivorCount != 0) {
		return issueScrubGateVerdict{Path: script, Err: "issue_scrub.py --check returned an inconsistent clean verdict"}
	}
	if result.Replacements < 0 || result.SurvivorCount < 0 {
		return issueScrubGateVerdict{Path: script, Err: "issue_scrub.py --check returned invalid counts"}
	}
	return issueScrubGateVerdict{
		Ran:           true,
		Clean:         result.Clean,
		NeedsScrub:    result.NeedsScrub,
		Replacements:  result.Replacements,
		SurvivorCount: result.SurvivorCount,
		Path:          script,
	}
}

// issueScrubVerdictJSON is the --json shape of the scrub verdict. Counts and the
// script path only — no needle names, never needle values.
type issueScrubVerdictJSON struct {
	Ran           bool   `json:"ran"`
	Clean         bool   `json:"clean"`
	NeedsScrub    bool   `json:"needs_scrub"`
	Replacements  int    `json:"replacements"`
	SurvivorCount int    `json:"survivor_count"`
	Path          string `json:"path,omitempty"`
	Error         string `json:"error,omitempty"`
}

// issueScrubVerdictJSONOf converts an internal verdict to the --json wire shape.
func issueScrubVerdictJSONOf(v issueScrubGateVerdict) issueScrubVerdictJSON {
	return issueScrubVerdictJSON{
		Ran:           v.Ran,
		Clean:         v.Clean,
		NeedsScrub:    v.NeedsScrub,
		Replacements:  v.Replacements,
		SurvivorCount: v.SurvivorCount,
		Path:          v.Path,
		Error:         v.Err,
	}
}

// issueCreateScrubRefused renders the refusal line. The needle report is counts
// only: survivor_count / replacements name HOW MANY private-material matches were
// found, never WHAT matched — echoing needle values would itself re-leak them.
func issueCreateScrubRefused(v issueScrubGateVerdict) string {
	if !v.Ran {
		return fmt.Sprintf("leak scrub unavailable (must refuse fail-closed): %s", v.Err)
	}
	return fmt.Sprintf(
		"leak scrub refused creation (survivor_count=%d replacements=%d) via %s; "+
			"refusing fail-closed unless the scrubber reports clean (needs_scrub-without-survivors "+
			"bodies still refuse: scrubbing would mutate private-adjacent text); "+
			"clean the body: run python %s (without --check) to scrub, or fix the body",
		v.SurvivorCount, v.Replacements, v.Path, v.Path)
}

// runIssueCreateScrubGate is the runIssueCreateWith-side wrapper: it runs the gate,
// renders the refusal to stderr lines on the existing discoverability-refusal
// pattern, fills result.Scrub for --json, and reports whether the create must stop.
// Exit convention: 2 usage, 3 scrub-gate refusal (same exit the discoverability
// audit refuses with).
func runIssueCreateScrubGate(stdout, stderr io.Writer, result *issueCreateResult, privateRoot, title, body string, asJSON bool) (int, bool) {
	verdict := issueCreateRunScrubGate(privateRoot, title, body)
	result.Scrub = issueScrubVerdictJSONOf(verdict)
	if verdict.Ran && verdict.Clean && !verdict.NeedsScrub && verdict.Replacements == 0 && verdict.SurvivorCount == 0 {
		return 0, false
	}
	errMsg := issueCreateScrubRefused(verdict)
	fmt.Fprintf(stderr, "fak-dev issue create: %s\n", errMsg)
	if asJSON {
		result.Title = ""
		result.Args = nil
		result.OK = false
		result.Error = errMsg
		encodeJSONOrFail(stdout, stderr, *result, "fak-dev issue create")
	}
	return 3, true
}
