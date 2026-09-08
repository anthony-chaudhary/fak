package goalrunner

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// FormatGoalPrompt prepends "/goal " if not present and enforces the 4000 char ceiling.
func FormatGoalPrompt(pointerContent string) (string, error) {
	cond := pointerContent
	if !strings.HasPrefix(cond, "/goal ") {
		cond = "/goal " + cond
	}
	if len(cond) > 4000 {
		return "", fmt.Errorf("goal condition is %d chars (>4000 cap) -- shrink the pointer", len(cond))
	}
	return cond, nil
}

// ScrubParentAuthEnv strips ANTHROPIC_*, CLAUDE_CODE_SESSION_ID, and CLAUDE_CODE_CHILD_SESSION
// from environment variables to prevent child processes from inheriting parent gateway auth.
func ScrubParentAuthEnv(environ []string) []string {
	scrubbed := make([]string, 0, len(environ))
	for _, entry := range environ {
		parts := strings.SplitN(entry, "=", 2)
		if len(parts) == 0 {
			continue
		}
		upper := strings.ToUpper(parts[0])
		if strings.HasPrefix(upper, "ANTHROPIC_") {
			continue
		}
		if upper == "CLAUDE_CODE_SESSION_ID" || upper == "CLAUDE_CODE_CHILD_SESSION" {
			continue
		}
		scrubbed = append(scrubbed, entry)
	}
	return scrubbed
}

// ResolveFakExe probes tools\.bin, repo root, and system PATH for the fak executable.
func ResolveFakExe(repoRoot, explicit string) (string, error) {
	if explicit != "" {
		st, err := os.Stat(explicit)
		if err != nil {
			return "", fmt.Errorf("fak binary not found: %s", explicit)
		}
		if st.IsDir() {
			return "", fmt.Errorf("fak binary path is a directory: %s", explicit)
		}
		return filepath.Abs(explicit)
	}

	candidates := []string{
		filepath.Join(repoRoot, "tools", ".bin", "fak.exe"),
		filepath.Join(repoRoot, "tools", ".bin", "fak"),
		filepath.Join(repoRoot, "fak.exe"),
		filepath.Join(repoRoot, "fak"),
	}

	for _, cand := range candidates {
		st, err := os.Stat(cand)
		if err == nil && !st.IsDir() {
			return filepath.Abs(cand)
		}
	}

	if path, err := exec.LookPath("fak"); err == nil {
		return filepath.Abs(path)
	}

	return "", fmt.Errorf("no fak binary found (looked in %s/tools/.bin, repo root, and PATH; pass explicit path)", repoRoot)
}

// LaunchDetachedWorker launches a headless /goal worker fully detached from the current process.
func LaunchDetachedWorker(opt LaunchOptions) (*LaunchResult, error) {
	if opt.Workspace == "" {
		wd, err := os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("resolve workspace: %w", err)
		}
		opt.Workspace = wd
	}

	if opt.LogDir == "" {
		opt.LogDir = filepath.Join(opt.Workspace, ".goal-runs")
	}
	if err := os.MkdirAll(opt.LogDir, 0755); err != nil {
		return nil, fmt.Errorf("create log dir: %w", err)
	}

	// 1. Preflight / PID cleanup
	_, _ = SweepDeadPidBreadcrumbs(opt.Workspace)

	// 2. Resolve pointer content and tag
	tag := opt.Tag
	content := opt.PointerContent
	if content == "" {
		if opt.PointerFile == "" {
			return nil, fmt.Errorf("no pointer file or pointer content provided")
		}
		pPath := opt.PointerFile
		if !filepath.IsAbs(pPath) {
			pPath = filepath.Join(opt.Workspace, pPath)
		}
		b, err := os.ReadFile(pPath)
		if err != nil {
			return nil, fmt.Errorf("pointer file not found: %w", err)
		}
		content = string(b)
		if tag == "" {
			base := filepath.Base(opt.PointerFile)
			tag = strings.TrimSuffix(base, filepath.Ext(base))
		}
	}
	if tag == "" {
		tag = "goal"
	}

	// 3. Format goal prompt
	prompt, err := FormatGoalPrompt(content)
	if err != nil {
		return nil, err
	}

	// 4. File paths
	stamp := time.Now().UTC().Format("20060102-150405")
	runID := fmt.Sprintf("%s-%s", tag, stamp)
	inF := filepath.Join(opt.LogDir, fmt.Sprintf("%s.in.txt", runID))
	outF := filepath.Join(opt.LogDir, fmt.Sprintf("%s.out.log", runID))
	errF := filepath.Join(opt.LogDir, fmt.Sprintf("%s.err.log", runID))
	pidF := filepath.Join(opt.LogDir, fmt.Sprintf("%s.pid", runID))
	seedDir := filepath.Join(opt.LogDir, fmt.Sprintf("seed-%s", runID))

	// Write prompt file (UTF-8, no BOM)
	if err := os.WriteFile(inF, []byte(prompt), 0644); err != nil {
		return nil, fmt.Errorf("write prompt file: %w", err)
	}

	// PlanOnly mode: format, write input, and return without spawning
	if opt.PlanOnly {
		launchWitness := fmt.Sprintf("PLAN_ONLY tag=%s run_id=%s", tag, runID)
		return &LaunchResult{
			PID:           0,
			Tag:           tag,
			RunID:         runID,
			LaunchWitness: launchWitness,
			PromptFile:    inF,
			OutLog:        outF,
			ErrLog:        errF,
			PIDFile:       pidF,
			SeedDir:       seedDir,
			PlanOnly:      true,
		}, nil
	}

	// Apply defaults
	contextBudgetTokens := opt.ContextBudgetTokens
	if contextBudgetTokens <= 0 {
		contextBudgetTokens = 2016000
	}
	restartLimit := opt.RestartLimit
	if restartLimit <= 0 {
		restartLimit = 16
	}
	maxDuration := opt.MaxDuration
	if maxDuration == "" {
		maxDuration = "45m"
	}
	exposeProfile := opt.ExposeProfile
	if exposeProfile == "" {
		exposeProfile = "headless"
	}

	claudeExe := opt.ClaudeExe
	if claudeExe == "" {
		claudeExe = "claude"
	}

	claudeArgs := []string{"-p"}
	if opt.Model != "" {
		claudeArgs = append(claudeArgs, "--model", opt.Model)
	}
	claudeArgs = append(claudeArgs, "--permission-mode", "bypassPermissions")

	// Determine if guarded (default true unless RawSpawn is set)
	guarded := true
	if opt.RawSpawn {
		guarded = false
	}

	var spawnCmd string
	var spawnArgs []string

	if guarded {
		fakExe, err := ResolveFakExe(opt.Workspace, opt.FakExe)
		if err != nil {
			return nil, err
		}
		if err := os.MkdirAll(seedDir, 0755); err != nil {
			return nil, fmt.Errorf("create seed dir: %w", err)
		}
		spawnCmd = fakExe
		spawnArgs = []string{
			"guard",
			"--context-budget-tokens", strconv.Itoa(contextBudgetTokens),
			"--restart-on-budget",
			"--restart-limit", strconv.Itoa(restartLimit),
			"--restart-seed-dir", seedDir,
			"--max-duration", maxDuration,
			"--expose-profile", exposeProfile,
			"--",
			claudeExe,
		}
		spawnArgs = append(spawnArgs, claudeArgs...)
	} else {
		spawnCmd = claudeExe
		spawnArgs = claudeArgs
	}

	// Open redirect stream files
	inFile, err := os.Open(inF)
	if err != nil {
		return nil, fmt.Errorf("open stdin file: %w", err)
	}
	defer inFile.Close()

	outFile, err := os.OpenFile(outF, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return nil, fmt.Errorf("open stdout log: %w", err)
	}
	defer outFile.Close()

	errFile, err := os.OpenFile(errF, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return nil, fmt.Errorf("open stderr log: %w", err)
	}
	defer errFile.Close()

	// Environment setup with seat hygiene
	baseEnv := opt.Environment
	if len(baseEnv) == 0 {
		baseEnv = os.Environ()
	}
	env := ScrubParentAuthEnv(baseEnv)
	if opt.ConfigDir != "" {
		env = append(env, "CLAUDE_CONFIG_DIR="+opt.ConfigDir)
	}
	if opt.OAuthToken != "" {
		env = append(env, "CLAUDE_CODE_OAUTH_TOKEN="+opt.OAuthToken)
	}

	cmd := exec.Command(spawnCmd, spawnArgs...)
	cmd.Dir = opt.Workspace
	cmd.Env = env
	cmd.Stdin = inFile
	cmd.Stdout = outFile
	cmd.Stderr = errFile
	cmd.SysProcAttr = detachedSysProcAttr()

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start detached worker: %w", err)
	}

	pid := cmd.Process.Pid

	// Write PID breadcrumb file
	pidContent := fmt.Sprintf("%d\n", pid)
	if err := os.WriteFile(pidF, []byte(pidContent), 0644); err != nil {
		return nil, fmt.Errorf("write pid file: %w", err)
	}

	// Release handle to detach process independently
	_ = cmd.Process.Release()

	accountTag := opt.AccountTag
	if accountTag == "" {
		accountTag = opt.Account
	}
	if accountTag == "" {
		accountTag = tag
	}

	// Emit machine-readable launch witness
	launchWitness := fmt.Sprintf("LAUNCH_WITNESS pid=%d tag=%s run_id=%s", pid, accountTag, runID)
	fmt.Println(launchWitness)

	return &LaunchResult{
		PID:           pid,
		Tag:           tag,
		RunID:         runID,
		LaunchWitness: launchWitness,
		PromptFile:    inF,
		OutLog:        outF,
		ErrLog:        errF,
		PIDFile:       pidF,
		SeedDir:       seedDir,
		Command:       spawnCmd,
		Args:          spawnArgs,
		PlanOnly:      false,
	}, nil
}
