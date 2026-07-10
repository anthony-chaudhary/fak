package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guard_startup.go — cohesive boot helpers peeled off cmdGuard (guard.go) so the god-function
// stays under its ceiling. These are pure code motion from cmdGuard: the capability-floor load,
// the full startup-report render, and the startup-banner emit. No behavior changes.

// guardStartupView carries every local cmdGuard hands the startup-report/banner helpers.
// The fields are exactly the cmdGuard locals the two render/emit helpers read.
type guardStartupView struct {
	providerAutodetected bool
	up                   string
	command              []string
	gwURL                string
	resolvedBase         string
	remoteBase           string
	floorSource          string
	injected             [][2]string
	logLabel             string
	auditLabel           string
	refusalCarryForward  []guardRefusalCarry
	localModel           bool
	ggufPath             string
	preCompactInstall    guardPreCompactInstall
	stopHookInstall      guardStopHookInstall
	handoffCfg           guardTaskHandoffConfig
	codexInstall         guardCodexInstall
	piInstall            guardPiInstall
	mcpInstall           guardMCPInstall
	debugStatsStderr     bool
	debugStats           bool
	quiet                bool
	pinUpstream          bool
	apiKey               string
	apiKeyEnv            string
	oauthSource          string
	mcache               guardManagedCachePosture
	contextBudgetLimit   int
	guardTraceID         string
	resetOnBudget        bool
	restartOnBudget      bool
	maxDurationLimit     time.Duration
	auditJournal         *journal.Journal
	bannerMode           string
}

// renderGuardStartupReport builds the FULL startup report — always, even under --quiet or the
// compact banner — so the session serves it back on demand for its whole life (`fak info
// --startup` / startup_report on /debug/vars). What reaches the terminal RIGHT NOW is the
// bannerMode emit in emitGuardStartupBanner.
func renderGuardStartupReport(v guardStartupView) string {
	var startupReport strings.Builder
	{
		if v.providerAutodetected {
			fmt.Fprintf(&startupReport, "fak guard: detected agent %q -> --provider %s (pass --provider to override)\n", strings.ToLower(filepath.Base(v.command[0])), v.up)
		}
		localLabel := ""
		if v.localModel {
			localLabel = filepath.Base(v.ggufPath)
		}
		printGuardBanner(&startupReport, guardBannerVersion(), guardBannerBuildStamp(), v.gwURL, v.up, v.resolvedBase, v.floorSource, formatGuardInjectedEnvForBanner(v.injected), v.logLabel, v.auditLabel, v.refusalCarryForward, v.remoteBase != "", v.localModel, localLabel, v.command)
		// A stamped-but-BEHIND (Skewed) or OFF-trunk (Diverged) guard is a distinct footgun from the
		// unstamped case printGuardBanner already warns about on its `build` row: this binary CAN attest
		// its commit — that commit is just provably behind/off origin/main — and the default guard path
		// re-execs THIS same file. Classify once by git ancestry and say so right under the banner.
		if warn := guardSkewBuildWarning(guardBuildSkewAssessment()); warn != "" {
			fmt.Fprint(&startupReport, warn)
		}
		if v.preCompactInstall.Applied {
			fmt.Fprintf(&startupReport, "fak guard: Claude PreCompact hook: %s (settings %s)\n", v.preCompactInstall.Mode, v.preCompactInstall.SettingsPath)
		}
		if v.stopHookInstall.Applied {
			fmt.Fprintf(&startupReport, "fak guard: Claude Stop hook (deny-all auto-continue): %s — graduated nudge→warn(%d)→final(%d)→give-up(>%d consecutive); a floor-refused-everything turn is reported as end_turn and this resumes the agent past it with escalating guidance, the give-up logged (--deny-all-continue off to disable)\n", v.stopHookInstall.Mode, v.stopHookInstall.WarnAt, v.stopHookInstall.FinalAt, v.stopHookInstall.Max)
		}
		if len(guardTaskHandoffEnv(v.handoffCfg)) > 0 {
			live := "validate-only"
			if v.handoffCfg.Live {
				live = "live-issue-sync"
			}
			fmt.Fprintf(&startupReport, "fak guard: task handoff Stop gate: %s (%s) — clean stops require %s; child sees $%s\n", v.handoffCfg.Mode, live, v.handoffCfg.File, guardTaskHandoffFileEnv)
		}
		printGuardCodexNote(&startupReport, v.codexInstall)
		printGuardPiNote(&startupReport, v.piInstall)
		printGuardMCPNote(&startupReport, v.mcpInstall)
		printGuardCapabilitiesNote(&startupReport, v.mcpInstall)
		switch {
		case v.debugStatsStderr:
			fmt.Fprintln(&startupReport, "  debug      : observable layer ON — one cache/token-value line per turn to stderr (request_tokens/cache_read/cache_creation/cache_hit/cache_rebate_tokens/compact/health); --debug-stats=false or --quiet to silence")
		case v.debugStats && !v.quiet:
			fmt.Fprintln(&startupReport, "  debug      : observable layer ON — the per-turn cache/token-value economy is kept OUT of the agent's full-screen UI to avoid corrupting it; read it live in the `fak info` pane and in the exit summary. Pass --debug-stats to also stream it here, --debug-stats=false to disable")
		}
		// A LOCAL in-kernel model has no upstream credential to report; the proxy-path auth
		// note (subscription OAuth vs passthrough) only applies when fak proxies an API.
		if !v.localModel {
			switch {
			case v.pinUpstream && v.up == "anthropic":
				fmt.Fprintf(&startupReport, "fak guard: upstream auth — Claude Pro/Max subscription (provider-reported identity; OAuth token from %s, sent as a bearer token)\n", v.oauthSource)
			case v.up == "anthropic" && v.apiKey != "":
				fmt.Fprintf(&startupReport, "fak guard: upstream auth — API key (from --api-key-env %s; provider-side API billing, not a fak claim)\n", v.apiKeyEnv)
			case v.up == "anthropic":
				fmt.Fprintln(&startupReport, "fak guard: upstream auth — passthrough (Claude Code forwards its own credential through the gateway)")
			}
		}
		// The session's prompt-cache posture, made explicit at boot: whether fak actively
		// manages the outbound cache_control (1h TTL upgrade) or stays passive and why.
		// Printed for the local branch too — "no provider prompt-cache wire" is itself the
		// answer an operator scanning for cache posture needs.
		fmt.Fprintln(&startupReport, v.mcache.bannerLine())
		if v.contextBudgetLimit > 0 {
			fmt.Fprintf(&startupReport, "fak guard: session budget — trace_id=%s context_tokens=%d\n", v.guardTraceID, v.contextBudgetLimit)
			if v.resetOnBudget {
				fmt.Fprintln(&startupReport, "fak guard: session reset — transparent carryover enabled")
			}
			if v.restartOnBudget {
				fmt.Fprintln(&startupReport, "fak guard: session restart — child relaunch on budget exhaustion enabled")
			}
		}
		if v.maxDurationLimit > 0 {
			fmt.Fprintf(&startupReport, "fak guard: session time budget — trace_id=%s max_duration=%s\n", v.guardTraceID, v.maxDurationLimit.String())
		}
	}
	auditThreadPath := v.auditLabel
	if v.auditJournal != nil {
		auditThreadPath = v.auditJournal.Path()
	}
	if row, err := enqueueGuardSessionThread(v.guardTraceID, v.up, v.command, auditThreadPath, time.Now()); err == nil {
		fmt.Fprintf(&startupReport, "fak guard: slack thread — queued root in %s (nonce=%s)\n", row.Channel, row.Nonce)
		startGuardSessionThreadDrain()
	} else {
		fmt.Fprintf(&startupReport, "fak guard: slack thread — unavailable: %v\n", err)
	}
	return startupReport.String()
}

// emitGuardStartupBanner spills to stderr exactly what bannerMode asks for, using the already
// rendered full report for the full-banner case. The full report is on the gateway either way.
func emitGuardStartupBanner(v guardStartupView, report string) {
	switch v.bannerMode {
	case guardBannerFull:
		fmt.Fprint(os.Stderr, report)
	case guardBannerCompact:
		printGuardCompactBanner(os.Stderr, guardBannerVersion(), guardShortBuildID(), v.gwURL, v.command, v.refusalCarryForward)
	case guardBannerAnimate:
		// Attended interactive: play the in-place icon animation instead of flashing text, but
		// only into a real color TTY that has not opted out — otherwise fall back to the static
		// compact banner so a piped stderr / NO_COLOR / FAK_GUARD_LAUNCH_ANIM=off run stays
		// byte-clean and motion-free. The full report is on the gateway either way.
		if guardLaunchAnimEnabled(v.bannerMode, guardFdIsTerminal(int(os.Stderr.Fd())), os.Getenv("NO_COLOR") != "", os.Getenv(guardLaunchAnimEnv)) {
			printGuardLaunchAnimation(os.Stderr, guardBannerVersion(), guardShortBuildID(), v.gwURL, v.command, v.refusalCarryForward, guardLaunchStderrWidth())
		} else {
			printGuardCompactBanner(os.Stderr, guardBannerVersion(), guardShortBuildID(), v.gwURL, v.command, v.refusalCarryForward)
		}
	}
}

// loadGuardCapabilityFloor installs the capability floor cmdGuard enforces. An explicit --policy
// file wins; otherwise the embedded guard floor. With NO floor the kernel default-denies every
// tool and the wrapped agent can do nothing — so guard ALWAYS loads one, fail-loud. It unions the
// operator allow overlay, digests the floor, and applies the runtime before returning the runtime,
// the floor-source label (banner), the policy digest (spawn metadata), and the load duration.
func loadGuardCapabilityFloor(policyPath string) (rt policy.Runtime, floorSource, policyDigest string, dur time.Duration) {
	var (
		err         error
		policyBytes []byte
	)
	tPolicy := time.Now()
	if policyPath != "" {
		policyBytes, err = os.ReadFile(policyPath)
		if err == nil {
			rt, err = policy.ParseRuntime(policyBytes)
			if err != nil {
				err = fmt.Errorf("policy %s: %w", policyPath, err)
			}
		} else {
			err = fmt.Errorf("policy: %w", err)
		}
		floorSource = policyPath
	} else {
		policyBytes = guardDefaultPolicyJSON
		rt, err = policy.ParseRuntime(guardDefaultPolicyJSON)
		floorSource = "built-in guard floor (--dump-policy to see it)"
	}
	must(err)
	// Union the OPERATOR allow overlay (`fak guard allow`) on top of whichever floor
	// loaded. It only widens Allow / AllowPrefix — the danger arg-rules and explicit
	// denies below stay intact — so an operator can re-admit a DEFAULT_DENY'd tool
	// out-of-band from the agent without ever loosening the genuine-danger floor. A
	// missing overlay is the common no-op; a malformed one fails loud (see guard_allow.go).
	overlayPath := guardAllowOverlayPath()
	allowOverlay, overlayErr := loadGuardAllowOverlay(overlayPath)
	if overlayErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", overlayErr)
		os.Exit(2)
	}
	if n := guardApplyAllowOverlay(&rt, allowOverlay); n > 0 {
		floorSource += fmt.Sprintf(" + operator allow overlay (%d extra tool(s); fak guard allow --list)", n)
	}
	policyDigest = guardPolicyDigest(policyBytes)
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	dur = time.Since(tPolicy)
	return rt, floorSource, policyDigest, dur
}
