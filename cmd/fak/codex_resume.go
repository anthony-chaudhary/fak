// `fak codex-resume` runs a fresh headless Codex continuation to the rollout's
// typed terminal, even when the upstream process fails to exit (#6414).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/codexresume"
)

type codexResumeThreads []string

type codexResumeAccountPlan struct {
	Account          string
	TargetHome       string
	SourceHome       string
	ThreadID         string
	RolloutPath      string
	CheckRolloutPath string
	Transition       codexresume.RehomeClass
	Binding          codexresume.ThreadBinding
	Store            codexresume.BindingStore
}

var codexResumeRecover = codexresume.Recover

func (t *codexResumeThreads) String() string { return strings.Join(*t, ",") }
func (t *codexResumeThreads) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return fmt.Errorf("thread id must not be empty")
	}
	*t = append(*t, value)
	return nil
}

type codexResumeBatchResult struct {
	Schema    string               `json:"schema"`
	Results   []codexresume.Result `json:"results"`
	Succeeded int                  `json:"succeeded"`
	Failed    int                  `json:"failed"`
	ExitCode  int                  `json:"exit_code"`
}

func runCodexResume(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "recover" {
		return runSessionRecover(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("codex-resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rollout := fs.String("rollout", "", "existing rollout JSONL for one resumed thread (otherwise discovered under CODEX_HOME/sessions)")
	codexHome := fs.String("codex-home", "", "Codex state home (default CODEX_HOME or ~/.codex)")
	account := fs.String("account", "", "exact discovered Codex account name (never rotates or falls back)")
	defaultHome, _ := os.UserHomeDir()
	defaultRegistry := os.Getenv("FAK_ACCOUNTS_REGISTRY")
	if defaultRegistry == "" && defaultHome != "" {
		defaultRegistry = filepath.Join(defaultHome, ".claude-accounts", "registry.json")
	}
	registryPath := fs.String("registry", defaultRegistry, "account registry path used to exclude persisted Claude-name collisions")
	codexExe := fs.String("codex", "codex", "Codex executable")
	cwd := fs.String("cwd", "", "child working directory (default current directory)")
	deadline := fs.Duration("deadline", 10*time.Minute, "maximum time before pre-terminal stall")
	drain := fs.Duration("drain", time.Second, "process drain time after rollout task_complete")
	promptFile := fs.String("prompt-file", "", "read the continuation prompt from a file (keeps it out of argv and ledgers)")
	resultFile := fs.String("result-file", "", "atomically persist the scrub-safe typed result")
	targetProvider := fs.String("target-provider", "", "target provider id (default persisted session provider)")
	targetWire := fs.String("target-wire", "responses", "target wire format")
	requiredCallPrefix := fs.String("required-call-prefix", "fc_", "required response_item.function_call.id prefix")
	checkOnly := fs.Bool("preflight-only", false, "report typed compatibility/admission without spawning")
	asJSON := fs.Bool("json", false, "emit the typed result as JSON")
	var threadFlags codexResumeThreads
	fs.Var(&threadFlags, "thread", "candidate thread id (repeat for bulk recovery; use --prompt-file when launching)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	threadIDs, code := validateCodexResumeArgs(stderr, fs, threadFlags, *account, *rollout, *codexHome, *promptFile, *checkOnly)
	if code != 0 {
		return code
	}
	dir := *cwd
	if dir == "" {
		var err error
		dir, err = os.Getwd()
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	absRollout := ""
	if *rollout != "" {
		var err error
		absRollout, err = filepath.Abs(*rollout)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	prompt := ""
	if *promptFile != "" {
		b, readErr := os.ReadFile(*promptFile)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak codex-resume: read prompt file: %v\n", readErr)
			return 1
		}
		prompt = string(b)
	} else if !*checkOnly {
		prompt = fs.Arg(1)
	}

	var accountPlan *codexResumeAccountPlan
	if *account != "" {
		reg, err := loadOrDiscover(*registryPath, defaultHome)
		if err != nil {
			fmt.Fprintf(stderr, "fak codex-resume: load account registry: %v\n", err)
			return 1
		}
		stateRoot := codexResumeBindingRoot(*registryPath, defaultHome)
		plan, err := planCodexResumeAccount(reg, discoveredCodexHomes(), stateRoot, *account, threadIDs[0], !*checkOnly)
		if err != nil {
			fmt.Fprintf(stderr, "fak codex-resume: account %q: %v\n", *account, err)
			return 1
		}
		accountPlan = &plan
	}

	results := make([]codexresume.Result, 0, len(threadIDs))
	exitCode := 0
	for _, threadID := range threadIDs {
		checkRollout := absRollout
		checkHome := *codexHome
		launchThreadID := threadID
		launchEnv := os.Environ()
		if accountPlan != nil {
			launchThreadID = accountPlan.ThreadID
			threadID = accountPlan.ThreadID
			checkRollout = accountPlan.CheckRolloutPath
			checkHome = accountPlan.TargetHome
			launchEnv = launchCodexEnv(launchEnv, accountPlan.TargetHome)
		}
		checkConfig := codexresume.CheckConfig{
			ThreadID:                       threadID,
			RolloutPath:                    checkRollout,
			CodexHome:                      checkHome,
			TargetProvider:                 *targetProvider,
			TargetWire:                     *targetWire,
			RequiredFunctionCallItemPrefix: *requiredCallPrefix,
		}
		var result codexresume.Result
		if *checkOnly {
			preflight := codexresume.Preflight(checkConfig)
			result = codexresume.Result{ThreadID: threadID, Preflight: &preflight, LaunchState: codexresume.LaunchNotAttempted}
			if preflight.Verdict == codexresume.VerdictResumable {
				result.Outcome = codexresume.OutcomeCheckOnly
			} else {
				result.Outcome = codexresume.OutcomeRefused
			}
		} else {
			cmd, commandEnv := buildCodexResumeLaunch(*codexExe, launchThreadID, prompt, launchEnv, "")
			if accountPlan != nil {
				cmd, commandEnv = buildCodexResumeLaunch(*codexExe, launchThreadID, prompt, os.Environ(), accountPlan.TargetHome)
			}
			var captured io.Writer = io.Discard
			if !*asJSON {
				captured = stderr
			}
			var err error
			result, err = codexResumeRecover(context.Background(), checkConfig, codexresume.Config{
				Command:      cmd,
				Dir:          dir,
				Env:          commandEnv,
				Deadline:     *deadline,
				Drain:        *drain,
				Stdout:       captured,
				Stderr:       stderr,
				PollInterval: 100 * time.Millisecond,
			})
			if err != nil {
				fmt.Fprintf(stderr, "fak codex-resume %s: %v\n", threadID, err)
				result.ThreadID = threadID
				if result.Outcome == "" {
					result.Outcome = codexresume.OutcomeExited
				}
			}
		}
		if accountPlan != nil {
			if err := saveCodexResumeBindingOnSuccess(accountPlan.Store, accountPlan.Binding, result.Outcome); err != nil {
				fmt.Fprintf(stderr, "fak codex-resume %s: save account binding: %v\n", threadID, err)
				return 1
			}
		}
		results = append(results, result)
		if code := codexResumeResultExitCode(result); code != 0 {
			if code == 130 {
				exitCode = 130
			} else if exitCode == 0 {
				exitCode = code
			}
		}
	}

	var output any
	if len(results) == 1 {
		output = results[0]
	} else {
		output = summarizeCodexResumeBatch(results)
	}
	if *resultFile != "" {
		if err := writeCodexResumeJSON(*resultFile, output); err != nil {
			fmt.Fprintf(stderr, "fak codex-resume: write result: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(output)
	} else {
		printCodexResumeResults(stdout, results)
	}
	return exitCode
}

func validateCodexResumeArgs(stderr io.Writer, fs *flag.FlagSet, threadFlags codexResumeThreads, account, rollout, codexHome, promptFile string, checkOnly bool) ([]string, int) {
	threadIDs := append(codexResumeThreads(nil), threadFlags...)
	if len(threadIDs) == 0 && fs.NArg() >= 1 {
		threadIDs = append(threadIDs, fs.Arg(0))
	}
	if len(threadIDs) == 0 {
		fmt.Fprintln(stderr, "usage: fak codex-resume [--rollout FILE] [--prompt-file FILE] SESSION_ID [PROMPT]")
		fmt.Fprintln(stderr, "       fak codex-resume --thread ID [--thread ID...] --prompt-file FILE")
		return nil, 2
	}
	if len(threadFlags) > 0 && fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak codex-resume: positional arguments cannot be combined with --thread")
		return nil, 2
	}
	if account != "" && len(threadIDs) != 1 {
		fmt.Fprintln(stderr, "fak codex-resume: --account supports exactly one candidate")
		return nil, 2
	}
	if account != "" && rollout != "" {
		fmt.Fprintln(stderr, "fak codex-resume: --rollout cannot be combined with --account")
		return nil, 2
	}
	if account != "" && codexHome != "" {
		fmt.Fprintln(stderr, "fak codex-resume: --codex-home cannot be combined with --account")
		return nil, 2
	}
	if rollout != "" && len(threadIDs) != 1 {
		fmt.Fprintln(stderr, "fak codex-resume: --rollout is valid only for one candidate")
		return nil, 2
	}
	if !checkOnly && promptFile == "" && (len(threadFlags) > 0 || fs.NArg() < 2) {
		fmt.Fprintln(stderr, "fak codex-resume: a prompt or --prompt-file is required for launch")
		return nil, 2
	}
	return threadIDs, 0
}

func printCodexResumeResults(stdout io.Writer, results []codexresume.Result) {
	for _, result := range results {
		preflightVerdict := codexresume.PreflightVerdict("")
		codexHome, sessionRoot, lookup, recoveryAction := "", "", codexresume.LookupState(""), ""
		if result.Preflight != nil {
			preflightVerdict = result.Preflight.Verdict
			codexHome = result.Preflight.CodexHome
			sessionRoot = result.Preflight.SessionRoot
			lookup = result.Preflight.LookupState
			recoveryAction = result.Preflight.RecoveryAction
		}
		launch := "launch state unknown"
		switch result.LaunchState {
		case codexresume.LaunchNotAttempted:
			launch = fmt.Sprintf("launch not attempted (verdict=%s recovery_action=%q)", preflightVerdict, recoveryAction)
		case codexresume.LaunchStartFailed:
			launch = "fresh Codex process start_failed"
		case codexresume.LaunchStarted:
			launch = fmt.Sprintf("fresh Codex process started (pid=%d)", result.LaunchPID)
		case codexresume.LaunchCompleted:
			launch = fmt.Sprintf("fresh Codex process completed (pid=%d)", result.LaunchPID)
		}
		fmt.Fprintf(stdout, "%s codex_home=%q session_root=%q lookup=%s launch=%q preflight=%s outcome=%s turn_status=%s useful_work=%t task_completed=%t process_exit=%t reclaimed=%t duration_ms=%d\n",
			result.ThreadID, codexHome, sessionRoot, lookup, launch, preflightVerdict, result.Outcome, result.TurnStatus, result.UsefulWork, result.TaskCompleted, result.ProcessExit, result.ForcedReclaim, result.DurationMS)
	}
}

func buildCodexResumeLaunch(codexExe, threadID, prompt string, baseEnv []string, targetHome string) ([]string, []string) {
	env := baseEnv
	if targetHome != "" {
		env = launchCodexEnv(baseEnv, targetHome)
	}
	return []string{codexExe, "exec", "resume", "--json", "--skip-git-repo-check", threadID, prompt}, env
}

func canonicalCodexResumeHome(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	if resolved, resolveErr := filepath.EvalSymlinks(abs); resolveErr == nil {
		abs = resolved
	}
	return filepath.Clean(abs), nil
}

func codexResumeFilesEqual(left, right string) (bool, error) {
	a, err := os.ReadFile(left)
	if err != nil {
		return false, err
	}
	b, err := os.ReadFile(right)
	if err != nil {
		return false, err
	}
	return string(a) == string(b), nil
}

func saveCodexResumeBindingOnSuccess(store codexresume.BindingStore, binding codexresume.ThreadBinding, outcome codexresume.Outcome) error {
	if outcome != codexresume.OutcomeCompleted && outcome != codexresume.OutcomeCompletedReclaimed {
		return nil
	}
	return store.Save(binding)
}

func codexResumeBindingRoot(registryPath, home string) string {
	if strings.TrimSpace(registryPath) != "" {
		return filepath.Join(filepath.Dir(registryPath), "codex-resume-bindings")
	}
	return filepath.Join(home, ".fak", "codex-resume-bindings")
}

func eligibleCodexResumeHomes(reg accounts.Registry, discovered []accounts.Home) []accounts.Home {
	persisted := make(map[string]struct{}, len(reg.Homes))
	for _, home := range reg.Homes {
		persisted[home.Name] = struct{}{}
	}
	out := make([]accounts.Home, 0, len(discovered))
	for _, home := range discovered {
		if _, collision := persisted[home.Name]; collision {
			continue
		}
		out = append(out, home)
	}
	return out
}

func planCodexResumeAccount(reg accounts.Registry, discovered []accounts.Home, stateRoot, account, query string, copyRollout bool) (codexResumeAccountPlan, error) {
	homes := eligibleCodexResumeHomes(reg, discovered)
	var target *accounts.Home
	for i := range homes {
		if homes[i].Name != account {
			continue
		}
		if target != nil {
			return codexResumeAccountPlan{}, errors.New("discovered Codex account name is ambiguous")
		}
		target = &homes[i]
	}
	if target == nil {
		return codexResumeAccountPlan{}, errors.New("not an exact discovered Codex account (or its name collides with the persisted registry)")
	}
	if !target.CanServe() {
		return codexResumeAccountPlan{}, errors.New("discovered Codex account is not ready to serve")
	}
	if strings.TrimSpace(target.Dir) == "" {
		return codexResumeAccountPlan{}, errors.New("discovered Codex account has no usable home")
	}
	info, err := os.Stat(target.Dir)
	if err != nil || !info.IsDir() {
		return codexResumeAccountPlan{}, errors.New("discovered Codex account home is unavailable")
	}
	identity, err := accounts.ReadCodexHomeIdentity(target.Dir)
	if err != nil {
		return codexResumeAccountPlan{}, fmt.Errorf("read target identity: %w", err)
	}
	if !identity.AuthPresent || identity.AccountDigest == "" {
		return codexResumeAccountPlan{}, errors.New("discovered Codex account has no recognized OAuth identity")
	}

	type ownedMatch struct {
		home  accounts.Home
		match codexresume.RolloutMatch
	}
	var matches []ownedMatch
	for _, home := range homes {
		match, findErr := codexresume.FindRollout(home.Dir, query)
		if errors.Is(findErr, codexresume.ErrRolloutNotFound) {
			continue
		}
		if findErr != nil {
			return codexResumeAccountPlan{}, fmt.Errorf("locate rollout in %q: %w", home.Name, findErr)
		}
		matches = append(matches, ownedMatch{home: home, match: match})
	}
	if len(matches) == 0 {
		return codexResumeAccountPlan{}, codexresume.ErrRolloutNotFound
	}
	threadID := matches[0].match.ThreadID
	for _, candidate := range matches[1:] {
		if candidate.match.ThreadID != threadID {
			return codexResumeAccountPlan{}, codexresume.ErrRolloutAmbiguous
		}
	}

	store := codexresume.BindingStore{Root: stateRoot}
	var binding *codexresume.ThreadBinding
	loaded, loadErr := store.Load(threadID)
	if loadErr == nil {
		binding = &loaded
	} else if !errors.Is(loadErr, codexresume.ErrBindingNotFound) {
		return codexResumeAccountPlan{}, fmt.Errorf("load thread binding: %w", loadErr)
	}
	selected := -1
	if len(matches) == 1 {
		selected = 0
	} else if binding != nil {
		for i := range matches {
			canonical, absErr := canonicalCodexResumeHome(matches[i].home.Dir)
			if absErr == nil && filepath.Clean(canonical) == filepath.Clean(binding.CanonicalHome) {
				if selected != -1 {
					return codexResumeAccountPlan{}, codexresume.ErrRolloutAmbiguous
				}
				selected = i
			}
		}
	} else {
		// A failed first launch may leave an identical idempotent destination copy but no
		// authoritative binding. Treat that selected target copy as a destination only when
		// exactly one other home owns byte-identical history; multiple non-target owners refuse.
		targetCanonical, _ := canonicalCodexResumeHome(target.Dir)
		targetMatch := -1
		nonTarget := -1
		for i := range matches {
			homeCanonical, _ := canonicalCodexResumeHome(matches[i].home.Dir)
			if filepath.Clean(homeCanonical) == filepath.Clean(targetCanonical) {
				targetMatch = i
				continue
			}
			if nonTarget != -1 {
				nonTarget = -2
				break
			}
			nonTarget = i
		}
		if targetMatch >= 0 && nonTarget >= 0 {
			equal, equalErr := codexResumeFilesEqual(matches[targetMatch].match.Path, matches[nonTarget].match.Path)
			if equalErr != nil {
				return codexResumeAccountPlan{}, fmt.Errorf("compare target rollout copy: %w", equalErr)
			}
			if equal {
				selected = nonTarget
			}
		}
	}
	if selected == -1 {
		return codexResumeAccountPlan{}, errors.New("rollout source ownership is ambiguous")
	}
	source := matches[selected]
	transition := codexresume.ClassifyRehome(binding, target.Dir, identity.AccountDigest, false)
	if transition == codexresume.RehomeUnknown || transition == codexresume.RehomeAmbiguous {
		return codexResumeAccountPlan{}, fmt.Errorf("cannot safely classify rehome transition %q", transition)
	}
	checkRolloutPath := source.match.Path
	rolloutPath := source.match.Path
	sourceCanonical, _ := canonicalCodexResumeHome(source.home.Dir)
	targetCanonical, _ := canonicalCodexResumeHome(target.Dir)
	if filepath.Clean(sourceCanonical) != filepath.Clean(targetCanonical) {
		rolloutPath = filepath.Join(target.Dir, filepath.FromSlash(source.match.RelativePath))
		if copyRollout {
			copied, copyErr := codexresume.CopyRollout(source.home.Dir, target.Dir, source.match.Path)
			if copyErr != nil {
				return codexResumeAccountPlan{}, fmt.Errorf("copy selected rollout: %w", copyErr)
			}
			rolloutPath = copied.Path
			checkRolloutPath = copied.Path
		}
	}
	newBinding, err := codexresume.NewThreadBinding(threadID, target.Dir, identity.AccountDigest, rolloutPath, time.Now().UTC())
	if err != nil {
		return codexResumeAccountPlan{}, fmt.Errorf("prepare thread binding: %w", err)
	}
	return codexResumeAccountPlan{
		Account: account, TargetHome: target.Dir, SourceHome: source.home.Dir,
		ThreadID: threadID, RolloutPath: rolloutPath, CheckRolloutPath: checkRolloutPath, Transition: transition,
		Binding: newBinding, Store: store,
	}, nil
}

func writeCodexResumeResult(path string, result codexresume.Result) error {
	return writeCodexResumeJSON(path, result)
}

func writeCodexResumeJSON(path string, value any) error {
	b, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + fmt.Sprintf(".tmp-%d", os.Getpid())
	if err := os.WriteFile(tmp, append(b, '\n'), 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func summarizeCodexResumeBatch(results []codexresume.Result) codexResumeBatchResult {
	batch := codexResumeBatchResult{Schema: "fak-codex-resume-batch/1", Results: results}
	for _, result := range results {
		code := codexResumeResultExitCode(result)
		if code == 0 {
			batch.Succeeded++
		} else {
			batch.Failed++
		}
		if code == 130 {
			batch.ExitCode = 130
		} else if code != 0 && batch.ExitCode == 0 {
			batch.ExitCode = code
		}
	}
	return batch
}

func codexResumeResultExitCode(result codexresume.Result) int {
	switch result.Outcome {
	case codexresume.OutcomeCompleted, codexresume.OutcomeCompletedReclaimed, codexresume.OutcomeCheckOnly:
		return 0
	case codexresume.OutcomeExplicitCancelled:
		return 130
	default:
		return 1
	}
}

func cmdCodexResume(args []string) { os.Exit(runCodexResume(os.Stdout, os.Stderr, args)) }
