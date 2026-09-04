package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/perfrsiscore"
	"github.com/anthony-chaudhary/fak/internal/repoguard"
)

// loopRunAdmit evaluates guard containment or disables guard, emits the admit (or refuse/end)
// ledger event, and returns the child argv to run. If admission fails or guard is unavailable,
// it returns exitCode != 0 and ok == false.
func loopRunAdmit(
	stdout, stderr io.Writer,
	ledger, loopID, runID string,
	cmdArgs []string,
	guardEnabled bool,
	baseMetrics map[string]int64,
	loopEvent func(loopmgr.Event) loopmgr.Event,
	asJSON bool,
) (childArgv []string, exitCode int, ok bool) {
	childArgv = append([]string(nil), cmdArgs...)
	admitReason := "GUARD_ADMITTED"
	admitSummary := "loop wrapper admitted command under fak guard"
	if guardEnabled {
		if violations := loopContainmentViolations(cmdArgs); len(violations) > 0 {
			m := cloneLoopMetrics(baseMetrics)
			m["violations"] = int64(len(violations))
			summary := repoguard.RenderReason(violations)
			if err := appendLoopRunEvent(ledger, loopEvent(loopmgr.Event{
				Kind:    loopmgr.EventAdmit,
				Status:  loopmgr.StatusRefused,
				Reason:  repoguard.Reason,
				Summary: summary,
				Metrics: m,
			})); err != nil {
				fmt.Fprintf(stderr, "fak loop run: %v\n", err)
				return nil, 1, false
			}
			fmt.Fprintf(stderr, "fak loop run: containment refused command: %s\n", summary)
			if asJSON && !writeLoopRunReport(stdout, stderr, ledger, loopID, runID, map[string]any{
				"status":    "refused",
				"reason":    repoguard.Reason,
				"exit_code": 3,
			}) {
				return nil, 1, false
			}
			return nil, 3, false
		}
		fakBin, err := loopExecutable()
		if err != nil {
			m := cloneLoopMetrics(baseMetrics)
			m["exit_code"] = 127
			_ = appendLoopRunEvent(ledger, loopEvent(loopmgr.Event{
				Kind:    loopmgr.EventEnd,
				Status:  loopmgr.StatusFailed,
				Reason:  "GUARD_UNAVAILABLE",
				Summary: err.Error(),
				Metrics: m,
			}))
			fmt.Fprintf(stderr, "fak loop run: resolve fak guard binary: %v\n", err)
			return nil, 127, false
		}
		childArgv = loopGuardArgv(fakBin, cmdArgs)
	} else {
		admitReason = "GUARD_DISABLED"
		admitSummary = "--no-guard disabled fak guard containment"
		fmt.Fprintln(stderr, "fak loop run: WARNING --no-guard disables fak guard containment for this run")
	}
	if err := appendLoopRunEvent(ledger, loopEvent(loopmgr.Event{
		Kind:    loopmgr.EventAdmit,
		Status:  loopmgr.StatusAdmitted,
		Reason:  admitReason,
		Summary: admitSummary,
		Metrics: cloneLoopMetrics(baseMetrics),
	})); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return nil, 1, false
	}
	return childArgv, 0, true
}

func prepareLoopChildEnv(ledger, loopID, runID string) (childEnv []string, performanceRSIOutput string, performanceRSIPrepErr error) {
	performanceRSIInput := strings.TrimSpace(os.Getenv(perfrsiscore.LoopTurnInputEnv))
	if performanceRSIInput != "" {
		return nil, "", nil
	}
	performanceRSIOutput, performanceRSIPrepErr = reserveLoopPerformanceRSIOutput(ledger)
	env := envMap(os.Environ())
	env[loopIDEnv] = loopID
	env[loopRunIDEnv] = runID
	if performanceRSIOutput != "" {
		env[loopPerformanceRSIOutputEnv] = performanceRSIOutput
	} else {
		delete(env, loopPerformanceRSIOutputEnv)
	}
	env[loopSandboxEnvAllow] = appendLoopEnvAllow(env[loopSandboxEnvAllow],
		loopPerformanceRSIOutputEnv, loopIDEnv, loopRunIDEnv)
	return envSliceFromMap(env), performanceRSIOutput, performanceRSIPrepErr
}

func recordLoopRunPerformanceRSI(stderr io.Writer, runID, performanceRSIOutput string, performanceRSIPrepErr error) {
	performanceRSIReceipt := perfrsiscore.ScoreLoopTurnFromEnvironment()
	if strings.TrimSpace(os.Getenv(perfrsiscore.LoopTurnInputEnv)) == "" {
		performanceRSIReceipt = scoreAutomaticLoopPerformanceRSI(performanceRSIOutput, runID, performanceRSIPrepErr)
	}
	if err := perfrsiscore.RecordLoopTurnUsage(performanceRSIReceipt); err != nil {
		fmt.Fprintf(stderr, "fak loop run: record performance-rsi usage: %v\n", err)
	}
	fmt.Fprintf(stderr, "fak loop run: performance-rsi loop-turn %s\n", perfrsiscore.FormatLoopTurnReceipt(performanceRSIReceipt))
}

func defaultLoopLedger() string {
	if v := os.Getenv("FAK_LOOP_LEDGER"); v != "" {
		return v
	}
	return filepath.Join(".fak", "loops.jsonl")
}

func defaultLoopPolicy() string {
	if v := os.Getenv("FAK_LOOP_POLICY"); v != "" {
		return v
	}
	return filepath.Join(".fak", "loop-policy.json")
}

func defaultLoopRegistry() string {
	if v := os.Getenv("FAK_LOOP_REGISTRY"); v != "" {
		return v
	}
	return filepath.Join("tools", "loop-registry.json")
}

func appendLoopRunEvent(ledger string, ev loopmgr.Event) error {
	_, err := loopmgr.Append(ledger, ev)
	return err
}

func cloneLoopMetrics(in map[string]int64) map[string]int64 {
	out := make(map[string]int64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func defaultLoopRunID(loopID string) string {
	replacer := strings.NewReplacer("/", "-", "\\", "-", ":", "-", " ", "-")
	name := strings.Trim(replacer.Replace(loopID), "-")
	if name == "" {
		name = "loop"
	}
	return fmt.Sprintf("%s-%s-%d", name, time.Now().UTC().Format("20060102T150405Z"), os.Getpid())
}

func loopGuardArgv(fakBin string, cmdArgs []string) []string {
	out := []string{fakBin, "guard", "--"}
	out = append(out, cmdArgs...)
	return out
}

func loopContainmentViolations(cmdArgs []string) []repoguard.Violation {
	command := loopRepoguardCommand(cmdArgs)
	if strings.TrimSpace(command) == "" {
		return nil
	}
	cwd, _ := os.Getwd()
	workspaceRoot := repoguard.FindRepoRoot(cwd)
	return repoguard.ClassifyCommand(command, workspaceRoot, repoguard.SafeRootsForWorkspace(workspaceRoot))
}

func loopRepoguardCommand(cmdArgs []string) string {
	if len(cmdArgs) == 0 {
		return ""
	}
	if command, ok := loopShellCCommand(cmdArgs); ok {
		return command
	}
	parts := make([]string, 0, len(cmdArgs))
	for _, arg := range cmdArgs {
		parts = append(parts, loopShellQuote(arg))
	}
	return strings.Join(parts, " ")
}

func loopShellCCommand(cmdArgs []string) (string, bool) {
	if len(cmdArgs) < 3 {
		return "", false
	}
	base := strings.ToLower(strings.TrimSuffix(filepath.Base(cmdArgs[0]), ".exe"))
	switch base {
	case "bash", "sh", "zsh", "dash", "ksh":
	default:
		return "", false
	}
	for i := 1; i < len(cmdArgs)-1; i++ {
		arg := cmdArgs[i]
		if arg == "--" {
			return "", false
		}
		if strings.HasPrefix(arg, "--") {
			continue
		}
		if arg == "-c" || (strings.HasPrefix(arg, "-") && strings.Contains(arg[1:], "c")) {
			return cmdArgs[i+1], true
		}
	}
	return "", false
}

func loopShellQuote(arg string) string {
	if arg == "" {
		return "''"
	}
	if strings.IndexFunc(arg, func(r rune) bool {
		return r <= ' ' || strings.ContainsRune(`'"$`+"\\"+`;|&<>(){}[]*?~!`, r)
	}) < 0 {
		return arg
	}
	return "'" + strings.ReplaceAll(arg, "'", "'\\''") + "'"
}

type loopKVList []string

func (l *loopKVList) String() string {
	if l == nil {
		return ""
	}
	return strings.Join(*l, ",")
}

func (l *loopKVList) Set(v string) error {
	*l = append(*l, v)
	return nil
}

func parseLoopEvidence(items []string) []loopmgr.EvidenceRef {
	out := make([]loopmgr.EvidenceRef, 0, len(items))
	for _, item := range items {
		kind, ref, ok := strings.Cut(item, "=")
		if !ok {
			continue
		}
		kind = strings.TrimSpace(kind)
		ref = strings.TrimSpace(ref)
		if kind == "" || ref == "" {
			continue
		}
		out = append(out, loopmgr.EvidenceRef{Kind: kind, Ref: ref})
	}
	return out
}
