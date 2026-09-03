package main

// `fak model incumbent` is the typed service-ownership surface for the preserved
// Qwen3.6 incumbent (issue #9714). The exact-model Mac campaigns (#9482/#8325 and
// the #9430 M-queue) must quiesce and restore the incumbent through a launchd job
// that provably owns the port-8090 listener; every drill that found the expected
// `com.fak.qwen36-model` job absent refused with
// alternate_launchd_supervisor_owns_incumbent rather than signal an unknown owner.
//
// This file carries the platform-independent half: the reviewed LaunchAgent
// definition renderer (pure, digest-bound) and the typed preflight classifier.
// The Darwin half (live lsof/ps/launchctl observation and the fail-closed
// bootstrap) lives in model_incumbent_darwin.go.

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	incumbentDefaultLabel = "com.fak.qwen36-model"
	incumbentDefaultPort  = 8090
	incumbentDefaultAlias = "qwen3.6-27b"

	// incumbentPreservedCommandSHA256 binds the preserved incumbent command
	// identity recorded by the #9714 ownership drill
	// (docs/_witnesses/issue-9714-qwen36-launchd-ownership): SHA-256 over the
	// space-joined launchd ProgramArguments. Rendering or observing a different
	// command is a conscious identity migration, not a silent default.
	incumbentPreservedCommandSHA256 = "a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d"
	incumbentRenderSchema           = "fak.model-incumbent-render/1"
	incumbentPreflightSchema        = "fak.model-incumbent-preflight/1"
	incumbentInstallSchema          = "fak.model-incumbent-install/1"
	incumbentOwnerIssueNumber       = 9714
)

// incumbentExpectedCommandSHA256 is the command identity the preflight
// classifier binds. Production default is the preserved #9714 drill identity;
// tests override it to exercise the classifier around synthetic commands.
var incumbentExpectedCommandSHA256 = incumbentPreservedCommandSHA256

const modelIncumbentUsage = "usage: fak model incumbent <render|preflight|install> ...\n" +
	"  fak model incumbent render    render the reviewed com.fak.qwen36-model LaunchAgent definition (digest-bound)\n" +
	"  fak model incumbent preflight read-only typed ownership verdict for the expected launchd job\n" +
	"  fak model incumbent install   fail-closed launchctl bootstrap of a rendered definition (dry-run default)\n"

func cmdModelIncumbent(args []string) {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, modelIncumbentUsage)
		os.Exit(2)
	}
	switch args[0] {
	case "render":
		os.Exit(runModelIncumbentRender(os.Stdout, os.Stderr, args[1:]))
	case "preflight":
		os.Exit(runModelIncumbentPreflight(os.Stdout, os.Stderr, args[1:]))
	case "install":
		os.Exit(runModelIncumbentInstall(os.Stdout, os.Stderr, args[1:]))
	case "-h", "--help", "help":
		fmt.Fprint(os.Stderr, modelIncumbentUsage)
	default:
		fmt.Fprintf(os.Stderr, "fak model incumbent: unknown subcommand %q\n", args[0])
		os.Exit(2)
	}
}

// incumbentCommandDigest binds a launchd command identity the same way the
// preserved #9714 drill receipts do: SHA-256 over the space-joined argv, which is
// what BSD `ps -o command=` renders and what the witness receipts hash.
func incumbentCommandDigest(argv []string) string {
	return digestBytes([]byte(strings.Join(argv, " ")))
}

type incumbentRenderSpec struct {
	Label          string
	Program        []string
	Environment    map[string]string
	StdoutPath     string
	StderrPath     string
	ThrottleInterl int
}

// renderIncumbentPlist renders the reviewed LaunchAgent definition. KeepAlive
// gives the incumbent its exact-restore semantics: a TERM through the owning job
// respawns the identical command, which is exactly the restoration the drill
// receipts require.
func renderIncumbentPlist(spec incumbentRenderSpec) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
	fmt.Fprintf(&b, "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n")
	fmt.Fprintf(&b, "<!-- Written by: fak model incumbent render (#9714) - install: launchctl bootstrap gui/$(id -u) %s -->\n", spec.Label)
	b.WriteString("<plist version=\"1.0\">\n")
	b.WriteString("<dict>\n")
	fmt.Fprintf(&b, "\t<key>Label</key>\n\t<string>%s</string>\n", plistEscape(spec.Label))
	b.WriteString("\t<key>ProgramArguments</key>\n\t<array>\n")
	for _, arg := range spec.Program {
		fmt.Fprintf(&b, "\t\t<string>%s</string>\n", plistEscape(arg))
	}
	b.WriteString("\t</array>\n")
	b.WriteString("\t<key>RunAtLoad</key>\n\t<true/>\n")
	b.WriteString("\t<key>KeepAlive</key>\n\t<true/>\n")
	fmt.Fprintf(&b, "\t<key>ThrottleInterval</key>\n\t<integer>%d</integer>\n", spec.ThrottleInterl)
	if spec.StdoutPath != "" {
		fmt.Fprintf(&b, "\t<key>StandardOutPath</key>\n\t<string>%s</string>\n", plistEscape(spec.StdoutPath))
	}
	if spec.StderrPath != "" {
		fmt.Fprintf(&b, "\t<key>StandardErrorPath</key>\n\t<string>%s</string>\n", plistEscape(spec.StderrPath))
	}
	if len(spec.Environment) > 0 {
		b.WriteString("\t<key>EnvironmentVariables</key>\n\t<dict>\n")
		for _, key := range sortedIncumbentKeys(spec.Environment) {
			fmt.Fprintf(&b, "\t\t<key>%s</key>\n\t\t<string>%s</string>\n", plistEscape(key), plistEscape(spec.Environment[key]))
		}
		b.WriteString("\t</dict>\n")
	}
	b.WriteString("</dict>\n")
	b.WriteString("</plist>\n")
	return b.String()
}

func sortedIncumbentKeys(env map[string]string) []string {
	keys := make([]string, 0, len(env))
	for key := range env {
		keys = append(keys, key)
	}
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

// plistEscape escapes the five XML metacharacters; launchd plists are plain XML
// text nodes, so the escaping is the whole contract.
func plistEscape(s string) string {
	replacer := strings.NewReplacer(
		"&", "&amp;",
		"<", "&lt;",
		">", "&gt;",
		"'", "&apos;",
		"\"", "&quot;",
	)
	return replacer.Replace(s)
}

func runModelIncumbentRender(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model incumbent render", flag.ContinueOnError)
	fs.SetOutput(stderr)
	argvFile := fs.String("argv-file", "", "JSON file holding the exact launchd ProgramArguments string array (private paths stay on this machine)")
	out := fs.String("out", "", "absolute output path for the rendered plist (must be outside this repository)")
	label := fs.String("label", incumbentDefaultLabel, "launchd service label (com.fak.* namespace)")
	expected := fs.String("expect-command-sha256", incumbentPreservedCommandSHA256, "required command digest; a different digest is a conscious identity migration and must be passed explicitly")
	envFile := fs.String("env-file", "", "optional JSON object of environment variables for the job")
	stdoutLog := fs.String("stdout-log", "", "optional absolute StandardOutPath")
	stderrLog := fs.String("stderr-log", "", "optional absolute StandardErrorPath")
	force := fs.Bool("force", false, "overwrite an existing output file")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	summary, err := modelIncumbentRender(incumbentRenderInput{
		ArgvFile:  *argvFile,
		EnvFile:   *envFile,
		Out:       *out,
		Label:     *label,
		Expected:  *expected,
		StdoutLog: *stdoutLog,
		StderrLog: *stderrLog,
		Force:     *force,
		Cwd:       "",
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak model incumbent render: %v\n", err)
		return 2
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(summary)
	return 0
}

type incumbentRenderInput struct {
	ArgvFile  string
	EnvFile   string
	Out       string
	Label     string
	Expected  string
	StdoutLog string
	StderrLog string
	Force     bool
	// Cwd anchors the repository-boundary refusal in tests; empty means process cwd.
	Cwd string
}

type incumbentRenderSummary struct {
	Schema                   string `json:"schema"`
	Issue                    int    `json:"issue"`
	Label                    string `json:"label"`
	PlistPath                string `json:"plist_path"`
	CommandSHA256            string `json:"command_sha256"`
	ExpectedCommandSHA256    string `json:"expected_command_sha256"`
	CommandMatchesExpected   bool   `json:"command_matches_expected_identity"`
	ArgumentCount            int    `json:"argument_count"`
	EnvironmentVariableCount int    `json:"environment_variable_count"`
	KeepAlive                bool   `json:"keep_alive"`
	RunAtLoad                bool   `json:"run_at_load"`
	PlistBytes               int    `json:"plist_bytes"`
	PlistSHA256              string `json:"plist_sha256"`
}

func modelIncumbentRender(input incumbentRenderInput) (incumbentRenderSummary, error) {
	label := strings.TrimSpace(input.Label)
	if label == "" {
		return incumbentRenderSummary{}, errors.New("--label is required")
	}
	if !strings.HasPrefix(label, "com.fak.") || strings.ContainsAny(label, " \t\r\n/") {
		return incumbentRenderSummary{}, fmt.Errorf("--label must stay inside the com.fak.* namespace without whitespace or slashes (got %q)", label)
	}
	if strings.TrimSpace(input.ArgvFile) == "" {
		return incumbentRenderSummary{}, errors.New("--argv-file is required: the exact private command stays operator-supplied, never tracked")
	}
	if !validSHA256(input.Expected) {
		return incumbentRenderSummary{}, errors.New("--expect-command-sha256 must be a SHA-256 digest")
	}
	rawArgv, err := os.ReadFile(input.ArgvFile)
	if err != nil {
		return incumbentRenderSummary{}, fmt.Errorf("read --argv-file: %w", err)
	}
	var argv []string
	if err := json.Unmarshal(rawArgv, &argv); err != nil {
		return incumbentRenderSummary{}, fmt.Errorf("decode --argv-file as a JSON string array: %w", err)
	}
	if len(argv) == 0 {
		return incumbentRenderSummary{}, errors.New("--argv-file holds no ProgramArguments")
	}
	program := make([]string, len(argv))
	for i, arg := range argv {
		program[i] = strings.TrimSpace(arg)
		if program[i] == "" {
			return incumbentRenderSummary{}, fmt.Errorf("argv[%d] is empty after trimming", i)
		}
		if strings.ContainsAny(program[i], "\r\n\x00") {
			return incumbentRenderSummary{}, fmt.Errorf("argv[%d] contains a line break or NUL", i)
		}
	}
	for _, path := range []string{input.StdoutLog, input.StderrLog} {
		if path != "" && !isDarwinAbsolutePath(path) {
			return incumbentRenderSummary{}, fmt.Errorf("log paths must be absolute (got %q)", path)
		}
	}
	env := map[string]string{}
	if strings.TrimSpace(input.EnvFile) != "" {
		rawEnv, err := os.ReadFile(input.EnvFile)
		if err != nil {
			return incumbentRenderSummary{}, fmt.Errorf("read --env-file: %w", err)
		}
		if err := json.Unmarshal(rawEnv, &env); err != nil {
			return incumbentRenderSummary{}, fmt.Errorf("decode --env-file as a JSON object: %w", err)
		}
	}
	out := strings.TrimSpace(input.Out)
	if out == "" {
		return incumbentRenderSummary{}, errors.New("--out is required: rendering never guesses where a plist lands")
	}
	if !filepath.IsAbs(out) {
		return incumbentRenderSummary{}, fmt.Errorf("--out must be an absolute path (got %q)", out)
	}
	if err := refuseIncumbentOutputInsideRepository(out, input.Cwd); err != nil {
		return incumbentRenderSummary{}, err
	}
	commandDigest := incumbentCommandDigest(program)
	matches := sameModelCanaryDigest(commandDigest, input.Expected)
	if !matches {
		return incumbentRenderSummary{}, fmt.Errorf("rendered command digest %s does not match the expected identity %s; a different command is a conscious identity migration and must pass its own --expect-command-sha256", commandDigest, input.Expected)
	}
	if _, err := os.Stat(out); err == nil && !input.Force {
		return incumbentRenderSummary{}, fmt.Errorf("--out %s already exists; pass --force to overwrite a reviewed definition", out)
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return incumbentRenderSummary{}, fmt.Errorf("create output directory: %w", err)
	}
	body := renderIncumbentPlist(incumbentRenderSpec{
		Label:          label,
		Program:        program,
		Environment:    env,
		StdoutPath:     input.StdoutLog,
		StderrPath:     input.StderrLog,
		ThrottleInterl: 10,
	})
	if err := atomicWriteFile(out, []byte(body), 0o644); err != nil {
		return incumbentRenderSummary{}, fmt.Errorf("write rendered plist: %w", err)
	}
	return incumbentRenderSummary{
		Schema:                   incumbentRenderSchema,
		Issue:                    incumbentOwnerIssueNumber,
		Label:                    label,
		PlistPath:                out,
		CommandSHA256:            commandDigest,
		ExpectedCommandSHA256:    digestKey(input.Expected),
		CommandMatchesExpected:   matches,
		ArgumentCount:            len(program),
		EnvironmentVariableCount: len(env),
		KeepAlive:                true,
		RunAtLoad:                true,
		PlistBytes:               len(body),
		PlistSHA256:              digestBytes([]byte(body)),
	}, nil
}

// digestKey normalizes a digest for reporting: the identity keeps its sha256:
// prefix only when the caller supplied one.
func digestKey(digest string) string {
	trimmed := strings.ToLower(strings.TrimSpace(digest))
	if !strings.HasPrefix(trimmed, "sha256:") {
		return "sha256:" + trimmed
	}
	return trimmed
}

// refuseIncumbentOutputInsideRepository keeps rendered plists out of the public
// tree: they embed the operator's private model/binary paths by design, and the
// public-leak guard must never have to catch one.
func refuseIncumbentOutputInsideRepository(out, cwd string) error {
	root := cwd
	if root == "" {
		root, _ = os.Getwd()
	}
	absOut, err := filepath.Abs(out)
	if err != nil {
		return fmt.Errorf("resolve --out: %w", err)
	}
	dir := root
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			if absOut == dir || strings.HasPrefix(absOut, dir+string(filepath.Separator)) {
				return fmt.Errorf("--out %s is inside this repository; rendered plists embed private paths and must live outside the repo (for example ~/Library/LaunchAgents)", out)
			}
			return nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return nil
		}
		dir = parent
	}
}

func atomicWriteFile(path string, raw []byte, mode os.FileMode) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	closed := false
	defer func() {
		if !closed {
			_ = tmp.Close()
		}
		_ = os.Remove(tmpPath)
	}()
	if _, err := tmp.Write(raw); err != nil {
		return err
	}
	if err := tmp.Chmod(mode); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	closed = true
	return os.Rename(tmpPath, path)
}

// incumbentPreflightReceipt is the read-only verdict the drill gate reads. Every
// field is an observation; nothing is inferred from wall time or absence of a
// probe failure.
type incumbentPreflightReceipt struct {
	Schema        string                          `json:"schema"`
	Issue         int                             `json:"issue"`
	ObservedAt    string                          `json:"observed_at"`
	ReadOnly      bool                            `json:"read_only"`
	LaunchdTarget string                          `json:"launchd_target"`
	ListenerPort  int                             `json:"listener_port"`
	ExpectedOwner incumbentPreflightExpectedOwner `json:"expected_owner"`
	Incumbent     incumbentPreflightIncumbent     `json:"incumbent"`
	Verdict       string                          `json:"verdict"`
	Reason        string                          `json:"reason,omitempty"`
}

type incumbentPreflightExpectedOwner struct {
	Label            string `json:"service_label"`
	JobPresent       bool   `json:"launchd_job_present"`
	JobPID           int    `json:"launchd_job_pid"`
	PlistPath        string `json:"launchd_plist_path"`
	JobBindsListener bool   `json:"launchd_job_binds_listener"`
}

type incumbentPreflightIncumbent struct {
	ListenerPresent         bool   `json:"listener_present"`
	ListenerPID             int    `json:"listener_pid"`
	CommandSHA256           string `json:"command_sha256"`
	CommandMatchesPreserved bool   `json:"command_matches_preserved_identity"`
	OwnerLabelSHA256        string `json:"owner_label_sha256,omitempty"`
	OwnerResolved           bool   `json:"owner_resolved"`
	HealthStatus            int    `json:"health_status"`
	ModelsStatus            int    `json:"models_status"`
	ModelAlias              string `json:"model_alias,omitempty"`
	AliasMatches            bool   `json:"alias_matches"`
}

const (
	incumbentVerdictOwned = "OWNED_EXPECTED_JOB"
	// incumbentVerdictJobAbsent covers the observed #9714 state: the expected job
	// does not exist in the user domain. Whether the healthy incumbent listener
	// is ownerless or held by an alternate supervisor is reported through
	// owner_resolved/owner_label_sha256; the drill gate refuses either way.
	incumbentVerdictJobAbsent = "EXPECTED_JOB_ABSENT"
	incumbentVerdictUnhealthy = "INCUMBENT_UNHEALTHY"
	incumbentVerdictMismatch  = "COMMAND_IDENTITY_MISMATCH"
	incumbentVerdictFailed    = "OBSERVATION_FAILED"
)

// classifyIncumbentPreflight turns observations into the typed verdict. The
// classifier is pure so the fail-closed boundary is testable without launchd.
func classifyIncumbentPreflight(obs incumbentPreflightReceipt) incumbentPreflightReceipt {
	obs.Schema = incumbentPreflightSchema
	obs.ReadOnly = true
	obs.Issue = incumbentOwnerIssueNumber
	obs.Verdict = incumbentVerdictFailed
	inc := &obs.Incumbent
	inc.CommandMatchesPreserved = sameModelCanaryDigest(inc.CommandSHA256, incumbentExpectedCommandSHA256)
	inc.AliasMatches = inc.ModelAlias == incumbentDefaultAlias
	owner := &obs.ExpectedOwner
	owner.JobBindsListener = owner.JobPresent && inc.ListenerPresent &&
		owner.JobPID > 0 && owner.JobPID == inc.ListenerPID
	switch {
	case obs.Reason != "":
		// An observation failure was already recorded; keep the typed
		// OBSERVATION_FAILED verdict and the specific reason.
	case !owner.JobPresent && !inc.ListenerPresent:
		obs.Verdict = incumbentVerdictJobAbsent
		obs.Reason = "expected job absent and no listener observed on the port"
	case !owner.JobPresent:
		obs.Verdict = incumbentVerdictJobAbsent
		obs.Reason = "alternate_launchd_supervisor_owns_incumbent"
		if !inc.ListenerPresent {
			obs.Reason = "expected job absent"
		}
	case !owner.JobBindsListener:
		obs.Verdict = incumbentVerdictMismatch
		obs.Reason = "expected job exists but does not bind the exact listener PID"
	case !inc.CommandMatchesPreserved:
		obs.Verdict = incumbentVerdictMismatch
		obs.Reason = "listener command digest does not match the preserved incumbent identity"
	case inc.HealthStatus != 200 || inc.ModelsStatus != 200:
		obs.Verdict = incumbentVerdictUnhealthy
		obs.Reason = fmt.Sprintf("incumbent endpoints returned health=%d models=%d; drill requires both 200", inc.HealthStatus, inc.ModelsStatus)
	case !inc.AliasMatches:
		obs.Verdict = incumbentVerdictUnhealthy
		obs.Reason = fmt.Sprintf("model inventory advertises %q; drill requires %q", inc.ModelAlias, incumbentDefaultAlias)
	default:
		obs.Verdict = incumbentVerdictOwned
	}
	if obs.Verdict == incumbentVerdictOwned && obs.ExpectedOwner.PlistPath == "" {
		obs.Verdict = incumbentVerdictMismatch
		obs.Reason = "expected job binds the listener but launchctl print omitted its plist path"
	}
	return obs
}

// incumbentVerdictExit maps a typed verdict onto the drill gate's contract: only
// OWNED_EXPECTED_JOB admits lifecycle work, everything else refuses with exit 2.
func incumbentVerdictExit(verdict string) int {
	if verdict == incumbentVerdictOwned {
		return 0
	}
	return 2
}
