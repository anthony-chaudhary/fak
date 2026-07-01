package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// guard_stophook.go — the harness half of the deny-all false-stop fix.
//
// The WIRE half is unchanged and correct: when the capability floor refuses EVERY tool call
// in a turn, the gateway must report stop_reason=end_turn (else the client hangs hunting for
// a tool_use block that was dropped — the v0.15.0 contract). But end_turn halts the agent
// loop though the model wanted to act, so the turn ends with the task unfinished — a STOP the
// agent did not choose. This actuator catches that stop and, in enforce mode, RESUMES the
// agent: it is a Claude Code `Stop` hook (installed by `fak guard` into the same --settings
// file as the PreCompact hook) that reads the gateway's fak_guard_deny_all_consecutive gauge
// and, when the most recent turn was a deny-all (and we are under the retry bound), exits 2
// with a continuation instruction on stderr — which Claude Code feeds back to the model so it
// picks an allowed alternative instead of stopping. The bound (max consecutive auto-continues)
// makes it impossible to loop forever on a model that keeps re-proposing refused calls.
//
// It mirrors guard-precompact exactly: a hidden subcommand polling /metrics, fail-open on any
// unavailability (a Stop hook that cannot reach the gateway must never wedge the agent), and a
// shadow mode that logs the would-be decision while always allowing the stop.

const (
	guardStopHookEnvMode       = "FAK_GUARD_DENYALL_MODE"
	guardStopHookEnvMetricsURL = "FAK_GUARD_DENYALL_METRICS_URL"
	guardStopHookEnvMax        = "FAK_GUARD_DENYALL_MAX"
	guardStopHookEnvWarn       = "FAK_GUARD_DENYALL_WARN"
	guardStopHookEnvFinal      = "FAK_GUARD_DENYALL_FINAL"

	guardTaskHandoffEnvMode = "FAK_GUARD_TASK_HANDOFF_MODE"
	guardTaskHandoffEnvFile = "FAK_GUARD_TASK_HANDOFF_FILE"
	guardTaskHandoffEnvRepo = "FAK_GUARD_TASK_HANDOFF_REPO"
	guardTaskHandoffEnvLive = "FAK_GUARD_TASK_HANDOFF_LIVE"

	// guardTaskHandoffFileEnv is the short, model-facing alias the wrapped agent sees. The
	// hook reads the guard-prefixed env above, but the continuation instruction points agents
	// at this stable name so they do not need to learn the hook's private wiring names.
	guardTaskHandoffFileEnv = "FAK_TASK_HANDOFF_FILE"

	// guardStopHookMetricName is the gateway gauge this hook polls: the count of consecutive
	// deny-all turns ending the most recent served turn (0 on a healthy completion).
	guardStopHookMetricName = "fak_guard_deny_all_consecutive"

	// The graduated back-off ladder. Rather than a single cliff (continue N times, then a hard
	// stop), the auto-continue guidance ESCALATES with the consecutive deny-all depth, and the
	// hard give-up moves much later. This gives a confused-but-capable model more room while a
	// genuinely-blocked one is told, early and explicitly, to declare BLOCKED and stop cleanly:
	//
	//	consecutive 0	-> ALLOW   (a clean completion is a real stop)
	//	1 .. Warn-1	-> NUDGE   (gentle "pick an allowed alternative")
	//	Warn .. Final-1	-> WARN    (force a relevance decision; name the BLOCKED exit)
	//	Final .. Max	-> FINAL   (last attempts; act now or declare BLOCKED)
	//	> Max		-> GIVE-UP (allow the stop, LOUDLY, so it is no longer invisible)
	guardStopHookDefaultWarn  = 3
	guardStopHookDefaultFinal = 7
	guardStopHookDefaultMax   = 9
)

// guardStopHookStage is the rung of the graduated deny-all back-off ladder the current
// consecutive count falls in. It drives BOTH the decision (allow / continue / give-up) and the
// firmness of the guidance fed back to the model.
type guardStopHookStage int

const (
	guardStopHookAllow  guardStopHookStage = iota // consecutive 0: a clean completion — allow the stop
	guardStopHookNudge                            // 1 .. Warn-1: gentle "pick an allowed alternative"
	guardStopHookWarn                             // Warn .. Final-1: force a relevance decision
	guardStopHookFinal                            // Final .. Max: last attempts before give-up
	guardStopHookGiveUp                           // > Max: bounded give-up — allow the stop, loudly
)

func (s guardStopHookStage) String() string {
	switch s {
	case guardStopHookAllow:
		return "allow"
	case guardStopHookNudge:
		return "nudge"
	case guardStopHookWarn:
		return "warn"
	case guardStopHookFinal:
		return "final"
	case guardStopHookGiveUp:
		return "give-up"
	default:
		return "unknown"
	}
}

// guardStopHookContinueReason is the NUDGE-rung instruction fed back to the model (via the Stop
// hook's exit-2 stderr) when fak first resumes the agent past a deny-all stop. The per-call
// refusal detail is already in the transcript (the in-band `[fak] refused …` note on the ended
// turn); this is the gentle nudge to act on it rather than stop. Later rungs of the ladder
// (guardStopHookStageMessage) escalate the firmness and name the sanctioned clean exit.
const guardStopHookContinueReason = "fak guard: your previous turn proposed only tool call(s) the capability floor refused, so it ended without action (reported upstream as end_turn). Do NOT re-propose a refused call unchanged — pick an ALLOWED alternative and continue the task. Most refusals are MODEL-FIXABLE by RESHAPING, not a dead end: a SELF_MODIFY deny means the floor saw a guarded write target (e.g. VERSION, .dos/, internal/…), so re-issue the command with the write aimed at an unguarded path, split compound commands to isolate the intended write, or drop the guarded write. If there is genuinely no allowed way to make progress, say so explicitly and then stop."

// normalizeDenyAllThresholds makes the ladder a TOTAL, deterministic function of its three
// knobs: it clamps any operator/env misconfiguration into the invariant 1 <= warn <= final <=
// max so a bad flag can never invert the ladder or wedge the hook. A non-positive max falls
// back to the default; warn floors at 1; final is pulled into [warn, max].
func normalizeDenyAllThresholds(warnAt, finalAt, maxN int) (int, int, int) {
	if maxN <= 0 {
		maxN = guardStopHookDefaultMax
	}
	if warnAt < 1 {
		warnAt = 1
	}
	if warnAt > maxN {
		warnAt = maxN
	}
	if finalAt < warnAt {
		finalAt = warnAt
	}
	if finalAt > maxN {
		finalAt = maxN
	}
	return warnAt, finalAt, maxN
}

// guardStopHookStageFor maps a consecutive deny-all count onto its ladder rung. Pure + total;
// thresholds are normalized first so the rung order can never invert.
func guardStopHookStageFor(consecutive, warnAt, finalAt, maxN int) guardStopHookStage {
	warnAt, finalAt, maxN = normalizeDenyAllThresholds(warnAt, finalAt, maxN)
	switch {
	case consecutive <= 0:
		return guardStopHookAllow
	case consecutive > maxN:
		return guardStopHookGiveUp
	case consecutive >= finalAt:
		return guardStopHookFinal
	case consecutive >= warnAt:
		return guardStopHookWarn
	default:
		return guardStopHookNudge
	}
}

// guardStopHookStageMessage is the exact stderr text fed back to the model when the hook BLOCKS
// the stop (exit 2) at a continue rung. Each rung is firmer than the last, and the WARN/FINAL
// rungs name the sanctioned clean exit: reply `BLOCKED: <reason>` and stop. (A pure-text turn
// resets the gateway's consecutive gauge to 0, so that declaration genuinely lets the next stop
// through — the agent's own choice, not a fak-forced halt.)
func guardStopHookStageMessage(stage guardStopHookStage, consecutive, maxN int) string {
	switch stage {
	case guardStopHookWarn:
		return fmt.Sprintf("fak guard: %d turns in a row now ended with EVERY proposed tool call refused by the capability floor — you are repeating calls it will not allow. STOP and DECIDE: is the remaining work actually reachable under this floor? If YES, take a DIFFERENT, allowed action now (a different tool, a narrower command, or a path the floor permits). If NO, reply on one line `BLOCKED: <reason>` and then stop — declaring blocked is a clean, expected outcome, not a failure. (Auto-continue %d of %d before fak lets the turn end.)", consecutive, consecutive, maxN)
	case guardStopHookFinal:
		return fmt.Sprintf("fak guard: FINAL auto-continue (%d of %d). After %d consecutive refused-everything turns fak will let the session stop with work possibly unfinished. If you have ANY allowed way to make progress, take it on THIS turn; otherwise reply on one line `BLOCKED: <reason>` and stop now, so the stop is your decision, not fak's.", consecutive, maxN, maxN)
	default:
		return guardStopHookContinueReason
	}
}

// guardStopHookGiveUpMessage is the OPERATOR-facing line printed when the hook gives up and
// allows the stop (exit 0, so it is NOT fed to the model). It makes the previously-invisible
// give-up legible: the residual false-stop the audit named.
func guardStopHookGiveUpMessage(consecutive, maxN int) string {
	return fmt.Sprintf("fak guard Stop: GIVE-UP after %d consecutive deny-all turns (every proposed tool call refused; %d > max %d) — allowing the stop with work possibly unfinished. Inspect why the floor refuses everything (fak guard --dump-policy) or raise --deny-all-max; --deny-all-continue off disables this layer.", consecutive, consecutive, maxN)
}

type guardStopHookInstall struct {
	Applied      bool
	Mode         string
	SettingsPath string
	MetricsURL   string
	WarnAt       int
	FinalAt      int
	Max          int
	Reason       string
}

type guardTaskHandoffConfig struct {
	Mode string
	File string
	Repo string
	Live bool
}

func cmdGuardStopHook(argv []string) {
	os.Exit(runGuardStopHook(os.Stderr, os.Stdin, argv))
}

// runGuardStopHook is the Stop-hook actuator. It returns the process exit code: 2 to BLOCK the
// stop (Claude Code continues the agent with the stderr text as guidance), 0 to allow it. It
// fails OPEN — any bad args, unknown mode, missing metrics URL, or unreachable gateway returns
// 0 so the hook can never wedge the agent — exactly the posture guard-precompact takes.
func runGuardStopHook(stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("guard-stophook", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modeFlag := fs.String("mode", os.Getenv(guardStopHookEnvMode), "off|shadow|enforce")
	metricsURLFlag := fs.String("metrics-url", os.Getenv(guardStopHookEnvMetricsURL), "gateway /metrics URL")
	maxFlag := fs.Int("max", guardStopHookMaxFromEnv(), "hard give-up: max consecutive deny-all turns to auto-continue past before letting the turn end")
	warnFlag := fs.Int("warn", guardStopHookWarnFromEnv(), "escalate the continue guidance to a relevance-decision warning at this consecutive deny-all depth")
	finalFlag := fs.Int("final", guardStopHookFinalFromEnv(), "escalate the continue guidance to a final warning at this consecutive deny-all depth")
	handoffModeFlag := fs.String("task-handoff-mode", os.Getenv(guardTaskHandoffEnvMode), "completion handoff gate: off|shadow|enforce")
	handoffFileFlag := fs.String("task-handoff-file", os.Getenv(guardTaskHandoffEnvFile), "path to fak.task-handoff.v1 JSON the agent must write before a clean stop")
	handoffRepoFlag := fs.String("task-handoff-repo", os.Getenv(guardTaskHandoffEnvRepo), "owner/repo passed to fak task handoff --live")
	handoffLiveFlag := fs.Bool("task-handoff-live", guardTaskHandoffLiveFromEnv(), "when true, sync valid next steps to GitHub with fak task handoff --live")
	timeout := fs.Duration("timeout", 500*time.Millisecond, "maximum time to wait for the gateway gauge")
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; bad hook args: %v\n", err)
		return 0
	}
	mode, err := normalizeGuardStopHookMode(*modeFlag)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; %v\n", err)
		return 0
	}
	if mode == guardPreCompactModeOff {
		return 0
	}
	// Best-effort read of Claude's Stop-hook stdin payload for stop_hook_active — used for the
	// shadow log and as a defensive signal; the gateway consecutive bound is the authoritative
	// loop guard, so a parse miss never changes the decision.
	active := readStopHookActive(stdin)
	metricsURL := strings.TrimSpace(*metricsURLFlag)
	if metricsURL == "" {
		metricsURL = guardPreCompactMetricsURLFromBase(os.Getenv("ANTHROPIC_BASE_URL"))
	}
	if metricsURL == "" {
		fmt.Fprintln(stderr, "fak guard Stop: allowing stop; no metrics URL configured")
		return 0
	}
	consecutive, err := fetchGuardStopHookConsecutive(context.Background(), metricsURL, *timeout)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; deny-all gauge unavailable: %v\n", err)
		return 0
	}
	warnAt, finalAt, maxN := normalizeDenyAllThresholds(*warnFlag, *finalFlag, *maxFlag)
	exit, block, stage := guardStopHookDecision(consecutive, warnAt, finalAt, maxN, mode)
	if mode == guardPreCompactModeShadow {
		action := "allow stop"
		switch {
		case block:
			action = "auto-continue (block stop)"
		case stage == guardStopHookGiveUp:
			action = "give up and allow stop"
		}
		fmt.Fprintf(stderr, "fak guard Stop: shadow would %s (stage=%s deny_all_consecutive=%d warn=%d final=%d max=%d stop_hook_active=%v)\n", action, stage, consecutive, warnAt, finalAt, maxN, active)
		return 0
	}
	if exit == 2 {
		// Exit 2 blocks the stop; stderr is shown to Claude as the reason to continue. The text
		// escalates with the ladder rung (nudge -> warn -> final).
		fmt.Fprintln(stderr, guardStopHookStageMessage(stage, consecutive, maxN))
		return 2
	}
	// Allowed (exit 0). A clean completion (stage allow) is silent; a bounded GIVE-UP is the
	// residual false-stop — make it loud and operator-visible (it is NOT fed to the model).
	if stage == guardStopHookGiveUp {
		fmt.Fprintln(stderr, guardStopHookGiveUpMessage(consecutive, maxN))
		return 0
	}
	if stage == guardStopHookAllow {
		return runGuardTaskHandoffGate(stderr, guardTaskHandoffConfig{
			Mode: *handoffModeFlag,
			File: *handoffFileFlag,
			Repo: *handoffRepoFlag,
			Live: *handoffLiveFlag,
		})
	}
	return 0
}

func runGuardTaskHandoffGate(stderr io.Writer, cfg guardTaskHandoffConfig) int {
	mode, err := normalizeGuardTaskHandoffMode(cfg.Mode)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard Stop: allowing stop; %v\n", err)
		return 0
	}
	if mode == guardPreCompactModeOff {
		return 0
	}
	file := strings.TrimSpace(cfg.File)
	if file == "" {
		fmt.Fprintln(stderr, "fak guard Stop: allowing stop; task handoff gate enabled but no handoff file configured")
		return 0
	}
	handoff, review, err := readAndReviewGuardTaskHandoff(file)
	if err != nil || !review.OK {
		msg := guardTaskHandoffRequiredMessage(file, review, err, cfg.Live, cfg.Repo)
		if mode == guardPreCompactModeShadow {
			fmt.Fprintf(stderr, "fak guard Stop: shadow would block clean stop for task handoff: %s\n", strings.TrimSpace(msg))
			return 0
		}
		fmt.Fprintln(stderr, msg)
		return 2
	}
	if cfg.Live && len(handoff.NextSteps) > 0 {
		var out, errb bytes.Buffer
		args := []string{"--file", file, "--live", "--json"}
		if repo := strings.TrimSpace(cfg.Repo); repo != "" {
			args = append(args, "--repo", repo)
		}
		code := runTaskHandoff(&out, &errb, args)
		if code != 0 {
			msg := fmt.Sprintf("fak guard Stop: task handoff is valid, but live GitHub issue sync failed (exit %d): %s", code, strings.TrimSpace(errb.String()))
			if mode == guardPreCompactModeShadow {
				fmt.Fprintln(stderr, "fak guard Stop: shadow would block clean stop: "+msg)
				return 0
			}
			fmt.Fprintln(stderr, msg)
			fmt.Fprintln(stderr, "Fix the handoff or GitHub sync, then stop again; use --task-handoff-live=false to require only the validated handoff artifact.")
			return 2
		}
	}
	return 0
}

func readAndReviewGuardTaskHandoff(file string) (taskmgr.Handoff, taskmgr.HandoffReview, error) {
	b, err := os.ReadFile(file)
	if err != nil {
		return taskmgr.Handoff{}, taskmgr.HandoffReview{}, err
	}
	var h taskmgr.Handoff
	if err := json.Unmarshal(b, &h); err != nil {
		return taskmgr.Handoff{}, taskmgr.HandoffReview{}, err
	}
	return h, taskmgr.ReviewHandoff(h), nil
}

func guardTaskHandoffRequiredMessage(file string, review taskmgr.HandoffReview, err error, live bool, repo string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard Stop: task handoff required before a clean stop. Write a valid `%s` JSON record to `%s` and stop again.\n", taskmgr.SchemaHandoff, file)
	if err != nil {
		fmt.Fprintf(&b, "Current handoff read failed: %v\n", err)
	} else if len(review.Reasons) > 0 {
		fmt.Fprintf(&b, "Current handoff was refused: %s\n", strings.Join(review.Reasons, ", "))
	}
	fmt.Fprintf(&b, "Required fields: task.state=`%s`, task.witness.verified_state=`%s`, current_state, and either 1-2 next_steps or no_next_step_reason.\n", taskmgr.StateDone, taskmgr.VerifiedDone)
	fmt.Fprintln(&b, "When follow-up work is reasonable, prefer 1-2 next_steps with stable key/title/body/reason so `fak task handoff` can create or update GitHub issues.")
	if live {
		fmt.Fprintln(&b, "This hook is in live mode: after the JSON validates it will run `fak task handoff --live` before allowing the stop.")
	} else {
		cmd := "fak task handoff --file \"" + file + "\" --live"
		if strings.TrimSpace(repo) != "" {
			cmd += " --repo " + strings.TrimSpace(repo)
		}
		fmt.Fprintf(&b, "To sync issues yourself before stopping, run `%s`; otherwise the validated handoff artifact is the stop witness.\n", cmd)
	}
	fmt.Fprintf(&b, "The path is also exposed to the agent as `$%s`.\n", guardTaskHandoffFileEnv)
	return strings.TrimRight(b.String(), "\n")
}

// guardStopHookDecision is the PURE decision behind the hook: given the gateway's consecutive
// deny-all count, the graduated thresholds, and the mode, return the exit code, whether it
// WOULD block, and the ladder rung (drives the guidance text + the shadow log). Side-effect-free
// so the policy is unit-tested without an HTTP gateway. The continue rungs (nudge/warn/final)
// block the stop (1..max); rung 0 is a clean completion and rung > max is the bounded give-up —
// both ALLOW the stop, so a stuck model cannot loop forever.
func guardStopHookDecision(consecutive, warnAt, finalAt, maxN int, mode string) (exit int, block bool, stage guardStopHookStage) {
	stage = guardStopHookStageFor(consecutive, warnAt, finalAt, maxN)
	if mode == guardPreCompactModeOff {
		return 0, false, stage
	}
	block = stage == guardStopHookNudge || stage == guardStopHookWarn || stage == guardStopHookFinal
	if mode == guardPreCompactModeShadow {
		return 0, block, stage // shadow always allows the stop (exit 0) but reports the would-be block
	}
	if block {
		return 2, true, stage
	}
	return 0, false, stage
}

// readStopHookActive parses the stop_hook_active flag from Claude's Stop-hook stdin JSON. A nil
// reader, an empty body, or a parse miss returns false — it is advisory only.
func readStopHookActive(stdin io.Reader) bool {
	if stdin == nil {
		return false
	}
	b, err := io.ReadAll(io.LimitReader(stdin, 1<<20))
	if err != nil || len(b) == 0 {
		return false
	}
	var payload struct {
		StopHookActive bool `json:"stop_hook_active"`
	}
	if json.Unmarshal(b, &payload) != nil {
		return false
	}
	return payload.StopHookActive
}

func normalizeGuardStopHookMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardPreCompactModeEnforce:
		// Default ENFORCE: the false stop is a real defect, so the fix is on by default. It is
		// bounded (guardStopHookDefaultMax) and fully observable (the deny-all metrics + exit
		// summary), and `--deny-all-continue off` opts out. This differs deliberately from the
		// PreCompact hook's shadow default, whose enforce path can break harness context management.
		return guardPreCompactModeEnforce, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	default:
		return "", fmt.Errorf("invalid --deny-all-continue mode %q (want off, shadow, or enforce)", mode)
	}
}

func normalizeGuardTaskHandoffMode(mode string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", guardPreCompactModeOff:
		return guardPreCompactModeOff, nil
	case guardPreCompactModeShadow:
		return guardPreCompactModeShadow, nil
	case guardPreCompactModeEnforce:
		return guardPreCompactModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid --task-handoff mode %q (want off, shadow, or enforce)", mode)
	}
}

func guardStopHookMaxFromEnv() int {
	return guardStopHookIntFromEnv(guardStopHookEnvMax, guardStopHookDefaultMax)
}

func guardStopHookWarnFromEnv() int {
	return guardStopHookIntFromEnv(guardStopHookEnvWarn, guardStopHookDefaultWarn)
}

func guardStopHookFinalFromEnv() int {
	return guardStopHookIntFromEnv(guardStopHookEnvFinal, guardStopHookDefaultFinal)
}

// guardStopHookIntFromEnv reads a positive int env override, falling back to def on any unset,
// blank, unparseable, or non-positive value (normalization clamps the rest).
func guardStopHookIntFromEnv(name string, def int) int {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func guardTaskHandoffLiveFromEnv() bool {
	v := strings.TrimSpace(os.Getenv(guardTaskHandoffEnvLive))
	return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes")
}

func guardTaskHandoffEnv(cfg guardTaskHandoffConfig) [][2]string {
	mode, err := normalizeGuardTaskHandoffMode(cfg.Mode)
	if err != nil || mode == guardPreCompactModeOff {
		return nil
	}
	file := strings.TrimSpace(cfg.File)
	if file == "" {
		return nil
	}
	env := [][2]string{
		{guardTaskHandoffEnvMode, mode},
		{guardTaskHandoffEnvFile, file},
		{guardTaskHandoffFileEnv, file},
	}
	if repo := strings.TrimSpace(cfg.Repo); repo != "" {
		env = append(env, [2]string{guardTaskHandoffEnvRepo, repo})
	}
	if cfg.Live {
		env = append(env, [2]string{guardTaskHandoffEnvLive, "1"})
	}
	return env
}

func guardTaskHandoffConfigOrZero(configs []guardTaskHandoffConfig) guardTaskHandoffConfig {
	if len(configs) == 0 {
		return guardTaskHandoffConfig{}
	}
	return configs[0]
}

// installGuardStopHook installs the Claude Code Stop hook for a guard session. When the
// PreCompact hook already wrote a --settings file (existingSettingsPath non-empty), the Stop
// hook is MERGED into it so a single --settings carries both (--settings is a single-value flag;
// injecting it twice clobbers rather than merges). Otherwise it writes its own settings file and
// injects --settings. Off mode or a non-claude child is a no-op (command returned unchanged).
func installGuardStopHook(command []string, mode, gwURL, existingSettingsPath string, warnAt, finalAt, maxN int, handoffConfig ...guardTaskHandoffConfig) ([]string, [][2]string, guardStopHookInstall, error) {
	normalized, err := normalizeGuardStopHookMode(mode)
	if err != nil {
		return command, nil, guardStopHookInstall{}, err
	}
	install := guardStopHookInstall{Mode: normalized, WarnAt: warnAt, FinalAt: finalAt, Max: maxN}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak"
	}
	dir := ""
	if strings.TrimSpace(existingSettingsPath) == "" {
		dir, err = os.MkdirTemp("", "fak-guard-stophook-*")
		if err != nil {
			return command, nil, guardStopHookInstall{}, err
		}
	}
	return installGuardStopHookAt(command, mode, gwURL, fakBin, dir, existingSettingsPath, warnAt, finalAt, maxN, handoffConfig...)
}

func installGuardStopHookAt(command []string, mode, gwURL, fakBin, dir, existingSettingsPath string, warnAt, finalAt, maxN int, handoffConfig ...guardTaskHandoffConfig) ([]string, [][2]string, guardStopHookInstall, error) {
	normalized, err := normalizeGuardStopHookMode(mode)
	if err != nil {
		return command, nil, guardStopHookInstall{}, err
	}
	// Normalize once so the install record, the banner, and the injected env all carry the SAME
	// effective ladder the hook will use — a misconfigured flag can never present one ladder and
	// run another.
	warnAt, finalAt, maxN = normalizeDenyAllThresholds(warnAt, finalAt, maxN)
	install := guardStopHookInstall{Mode: normalized, WarnAt: warnAt, FinalAt: finalAt, Max: maxN}
	if normalized == guardPreCompactModeOff {
		install.Reason = "disabled"
		return command, nil, install, nil
	}
	if !guardPreCompactIsClaudeCommand(command) {
		install.Reason = "non-claude-child"
		return command, nil, install, nil
	}

	var settingsPath string
	if strings.TrimSpace(existingSettingsPath) != "" {
		// Merge the Stop hook INTO the file the PreCompact hook already wrote + injected.
		if err := mergeGuardStopHookIntoSettings(existingSettingsPath, fakBin); err != nil {
			return command, nil, install, err
		}
		settingsPath = existingSettingsPath
		// command already carries --settings; do NOT inject it again.
	} else {
		if strings.TrimSpace(dir) == "" {
			return command, nil, install, errors.New("empty Stop hook settings directory")
		}
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return command, nil, install, err
		}
		settingsPath = filepath.Join(dir, "claude-stophook-settings.json")
		if err := writeGuardStopHookSettings(settingsPath, fakBin); err != nil {
			return command, nil, install, err
		}
		command = appendClaudeSettingsArg(command, settingsPath)
	}
	metricsURL := guardPreCompactMetricsURLFromBase(gwURL)
	install.Applied = true
	install.SettingsPath = settingsPath
	install.MetricsURL = metricsURL
	env := [][2]string{
		{guardStopHookEnvMode, normalized},
		{guardStopHookEnvMetricsURL, metricsURL},
		{guardStopHookEnvWarn, strconv.Itoa(warnAt)},
		{guardStopHookEnvFinal, strconv.Itoa(finalAt)},
		{guardStopHookEnvMax, strconv.Itoa(maxN)},
	}
	env = append(env, guardTaskHandoffEnv(guardTaskHandoffConfigOrZero(handoffConfig))...)
	return command, env, install, nil
}

// guardStopHookMatchers builds the Stop-hook settings entry. The Stop event takes NO matcher
// (matchers are for tool-scoped events), so the Matcher field is left empty (omitempty drops it).
func guardStopHookMatchers(fakBin string) []guardPreCompactClaudeMatcher {
	return []guardPreCompactClaudeMatcher{{
		Hooks: []guardPreCompactClaudeCommand{{
			Type:    "command",
			Command: guardPreCompactHookCommand(fakBin),
			Args:    []string{"guard-stophook"},
		}},
	}}
}

func writeGuardStopHookSettings(path, fakBin string) error {
	settings := guardPreCompactClaudeSettings{
		Hooks: map[string][]guardPreCompactClaudeMatcher{
			"Stop": guardStopHookMatchers(fakBin),
		},
	}
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

// mergeGuardStopHookIntoSettings adds (or replaces) the Stop hook in an existing guard settings
// file, preserving every other key (e.g. the PreCompact hook), so a single --settings carries both.
func mergeGuardStopHookIntoSettings(path, fakBin string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var settings guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &settings); err != nil {
		return fmt.Errorf("parse existing hook settings %s: %w", path, err)
	}
	if settings.Hooks == nil {
		settings.Hooks = map[string][]guardPreCompactClaudeMatcher{}
	}
	settings.Hooks["Stop"] = guardStopHookMatchers(fakBin)
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

func fetchGuardStopHookConsecutive(ctx context.Context, metricsURL string, timeout time.Duration) (int, error) {
	if timeout <= 0 {
		timeout = 500 * time.Millisecond
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return 0, err
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("metrics returned HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return 0, err
	}
	return parseGuardStopHookConsecutive(string(body))
}

// parseGuardStopHookConsecutive extracts the unlabeled fak_guard_deny_all_consecutive gauge
// value from a Prometheus scrape. Not-found is an error so the caller fails open rather than
// silently treating a missing gauge as 0 (which would never auto-continue).
func parseGuardStopHookConsecutive(metrics string) (int, error) {
	for _, line := range strings.Split(metrics, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 || fields[0] != guardStopHookMetricName {
			continue
		}
		value, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return 0, fmt.Errorf("parse %s value %q: %w", guardStopHookMetricName, fields[1], err)
		}
		return int(value), nil
	}
	return 0, fmt.Errorf("metric %s not found", guardStopHookMetricName)
}
