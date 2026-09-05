package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/cloudroute"
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
	policyDigest         string
	injected             [][2]string
	responseProfile      string
	workProfile          string
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
	keychainAPIKey       bool
	oauthSource          string
	mcache               guardManagedCachePosture
	contextBudgetLimit   int
	guardTraceID         string
	resetOnBudget        bool
	restartOnBudget      bool
	maxDurationLimit     time.Duration
	auditJournal         *journal.Journal
	bannerMode           string
	// upstreamTrustNote names the trust store this session validates with, empty on a
	// host with no corporate CA bundle declared (#8172).
	upstreamTrustNote string
	// cloudRouteWaived records that the session runs against a request-signed cloud
	// model route with the UPSTREAM_UNSUPPORTED refusal waived (#8172). The report has
	// to say so: every other line in it describes adjudication fak is NOT performing on
	// this posture, and a banner that stays silent about that is the report lying.
	cloudRouteWaived bool
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
		responseProfile, workProfile := v.responseProfile, v.workProfile
		if responseProfile == "" {
			responseProfile = "full"
		}
		if workProfile == "" {
			workProfile = "standard"
		}
		fmt.Fprintf(&startupReport, "  response profile : %s\n", responseProfile)
		fmt.Fprintf(&startupReport, "  work profile     : %s\n", workProfile)
		if v.policyDigest != "" {
			fmt.Fprintf(&startupReport, "fak guard: active config digest %s\n", v.policyDigest)
		}
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
			case v.up == "anthropic" && v.apiKey != "" && v.keychainAPIKey:
				fmt.Fprintln(&startupReport, "fak guard: upstream auth — API key (Claude Code's saved key from the macOS Keychain; provider-side API billing, not a fak claim)")
			case v.up == "anthropic" && v.apiKey != "":
				fmt.Fprintf(&startupReport, "fak guard: upstream auth — API key (from --api-key-env %s; provider-side API billing, not a fak claim)\n", v.apiKeyEnv)
			case v.up == "anthropic":
				fmt.Fprintln(&startupReport, "fak guard: upstream auth — passthrough (Claude Code forwards its own credential through the gateway)")
			}
		}
		// #8172: the enterprise posture, on the durable report rather than a stderr line a
		// compact launch would lose. Both are empty/false on a host with no corporate CA
		// bundle and no cloud route selector, which is every non-managed host.
		if v.upstreamTrustNote != "" {
			fmt.Fprint(&startupReport, v.upstreamTrustNote)
		}
		if v.cloudRouteWaived {
			fmt.Fprintf(&startupReport, "fak guard: upstream adjudication — NONE for model traffic: a request-signed cloud route (%s=1) was waived, so the child signs its own requests and never reads the injected base URL. The hook floor, tool brokering, transcript, and sandbox below still apply; the model traffic above them does not pass through fak. `fak serve --stdio --policy FILE` is the adjudicated route on this posture.\n", cloudroute.WaiverKey)
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
	// Clear the process-wide handle before attempting this session's root. A guard process
	// owns exactly one session in production; the reset also keeps repeated in-process test
	// runs from inheriting a prior card when the next enqueue fails.
	guardSessionCardHandle = nil
	startedAt := time.Now()
	if row, err := enqueueGuardSessionThread(v.guardTraceID, v.up, v.command, auditThreadPath, startedAt); err == nil {
		fmt.Fprintf(&startupReport, "fak guard: slack thread — queued root in %s (nonce=%s)\n", row.Channel, row.Nonce)
		guardSessionCardHandle = newGuardSessionCard(row.Channel, row.Nonce, startedAt)

		// The root alone is only a header. Queue the launch banner + operator context under
		// it BEFORE starting the drainer, using ParentNonce so one pass can post the root and
		// resolve every reply onto that exact thread. This is the production wiring for the
		// already-tested control-point rows; without it the live channel carried 100/100
		// start-only roots even though the renderer existed.
		cwd, _ := os.Getwd()
		n, replyErr := enqueueGuardSessionControlPoints(row.Nonce, row.Channel, startupReport.String(), guardSessionControlContext{
			Command:    v.command,
			Cwd:        cwd,
			Audit:      auditThreadPath,
			Trace:      v.guardTraceID,
			Provider:   v.up,
			GatewayURL: v.gwURL,
		})
		if replyErr != nil {
			fmt.Fprintf(&startupReport, "fak guard: slack thread — control replies unavailable: %v\n", replyErr)
		} else {
			fmt.Fprintf(&startupReport, "fak guard: slack thread — queued %d control %s\n", n, pluralWord(n, "reply", "replies"))
		}
		startGuardSessionThreadDrain()
	} else {
		fmt.Fprintf(&startupReport, "fak guard: slack thread — unavailable: %v\n", err)
	}
	return startupReport.String()
}

// emitGuardStartupBanner spills to stderr exactly what bannerMode asks for, using the already
// rendered full report for the full-banner case. The full report is on the gateway either way.
func emitGuardStartupBanner(v guardStartupView, report string) {
	writeGuardStartupBanner(os.Stderr, v, report, guardFdIsTerminal(int(os.Stderr.Fd())), os.Getenv("NO_COLOR") != "", os.Getenv(guardLaunchAnimEnv), guardLaunchStderrWidth())
}

// writeGuardStartupBanner is the byte-level render seam. In particular, the healthy default
// progress mode writes nothing here; its delayed surface is emitted after setup begins.
func writeGuardStartupBanner(w io.Writer, v guardStartupView, report string, stderrTTY, noColor bool, animEnv string, width int) {
	switch v.bannerMode {
	case guardBannerFull:
		fmt.Fprint(w, report)
	case guardBannerCompact:
		printGuardCompactBanner(w, guardBannerVersion(), guardShortBuildID(), v.gwURL, v.command, v.refusalCarryForward)
	case guardBannerAnimate:
		// Attended interactive: play the in-place icon animation instead of flashing text, but
		// only into a real color TTY that has not opted out — otherwise fall back to the static
		// compact banner so a piped stderr / NO_COLOR / FAK_GUARD_LAUNCH_ANIM=off run stays
		// byte-clean and motion-free. The full report is on the gateway either way.
		if guardLaunchAnimEnabled(v.bannerMode, stderrTTY, noColor, animEnv) {
			printGuardLaunchAnimation(w, guardBannerVersion(), guardShortBuildID(), v.gwURL, v.command, v.refusalCarryForward, width)
		} else {
			printGuardCompactBanner(w, guardBannerVersion(), guardShortBuildID(), v.gwURL, v.command, v.refusalCarryForward)
		}
	case guardBannerProgress:
		// Progress is rendered only after the startup delay. Off remains silent via no case.
	}
}

// loadGuardCapabilityFloor installs the capability floor cmdGuard enforces. An explicit --policy
// file wins; otherwise the embedded guard floor. With NO floor the kernel default-denies every
// tool and the wrapped agent can do nothing — so guard ALWAYS loads one, fail-loud. It unions the
// operator allow overlay, digests the floor, and applies the runtime before returning the runtime,
// the floor-source label (banner), the policy digest (spawn metadata), and the load duration.
func loadGuardCapabilityFloor(policyPath string, overrides ...string) (rt policy.Runtime, floorSource, policyDigest string, dur time.Duration) {
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

	var profileStr string
	if len(overrides) > 1 && strings.TrimSpace(overrides[1]) != "" {
		profileStr = strings.TrimSpace(overrides[1])
	} else if env := strings.TrimSpace(os.Getenv("FAK_PROFILE")); env != "" {
		profileStr = env
	} else if rt.PolicyContext.Profile != "" {
		profileStr = rt.PolicyContext.Profile
	}
	if profileStr != "" {
		prof, pErr := policy.ParseProfile(profileStr)
		must(pErr)
		prof.Apply(&rt)
		floorSource += "; profile=" + string(prof)
	}

	effectivePosture := rt.Adjudicator.Posture
	var postureStr string
	if len(overrides) > 0 && overrides[0] != "" {
		p, pErr := policy.ParsePosture(overrides[0])
		must(pErr)
		effectivePosture = p
		postureStr = overrides[0]
	} else if env := strings.TrimSpace(os.Getenv("FAK_GUARD_POSTURE")); env != "" {
		p, pErr := policy.ParsePosture(env)
		must(pErr)
		effectivePosture = p
		postureStr = env
	} else if profileStr != "" {
		effectivePosture = rt.Adjudicator.Posture
		postureStr = effectivePosture.String()
	} else if policyPath == "" {
		effectivePosture = adjudicator.PostureDefaultOpen
		postureStr = "default_open"
	}
	if postureStr != "" {
		floorSource += "; posture=" + postureStr
	}
	rt.Adjudicator.Posture = effectivePosture
	rt.PolicyContext.Posture = effectivePosture
	// Union the OPERATOR allow overlay (`fak guard allow`) on top of whichever floor
	// loaded. It only widens Allow / AllowPrefix — the danger arg-rules and explicit
	// denies below stay intact — so an operator can re-admit a DEFAULT_DENY'd tool
	// out-of-band from the agent without ever loosening the genuine-danger floor. A
	// missing overlay is the common no-op; a malformed one fails loud (see guard_allow.go).
	allowOverlay, _, overlayErr := loadGuardAllowOverlayLayers()
	if overlayErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", overlayErr)
		os.Exit(2)
	}
	if n := guardApplyAllowOverlay(&rt, allowOverlay); n > 0 {
		floorSource += fmt.Sprintf(" + operator allow overlay (%d extra tool(s); fak guard allow --list)", n)
	}
	if attached := guardAllowShellAttachments(allowOverlay.Allow); len(attached) > 0 {
		floorSource += fmt.Sprintf("; inherited shell danger rules attached: %s", strings.Join(attached, ", "))
	}
	launchGrant, _ := applyLaunchToolGrant(&rt)
	if n := len(launchGrant.Allow); n > 0 {
		floorSource += fmt.Sprintf(" + launch-scoped allow grant (%d exact tool(s); --allow-tool, expires with this process)", n)
	}
	if attached := guardAllowShellAttachments(launchGrant.Allow); len(attached) > 0 {
		floorSource += fmt.Sprintf("; inherited shell danger rules attached: %s", strings.Join(attached, ", "))
	}
	denyPath := guardDenyOverlayPath()
	denyOverlay, denyErr := loadGuardDenyOverlay(denyPath)
	if denyErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", denyErr)
		os.Exit(2)
	}
	if n := guardApplyDenyOverlay(&rt, denyOverlay); n > 0 {
		floorSource += fmt.Sprintf(" + repo-local deny overlay (%d tool(s); fak guard deny --list)", n)
	}
	// Union the SELF-authored tighten overlay (#5181, epic #5170 Track F) — the one
	// amendment channel the WRAPPED AGENT itself may write. Unlike the two operator
	// overlays above it is not trusted on arrival: it is admitted only through the
	// amendment gate (guard_self_tighten.go), which classifies base -> base∪overlay
	// via policy.DiffAmendment and admits a no-op or tighten-only delta only. It runs
	// AFTER both operator overlays so the agent tightens the floor an operator actually
	// configured, and BEFORE protectGuardPolicyConfig so its own file is deliberately
	// left OUT of SelfModifyGlobs — the agent must be able to write it, which is why it
	// does not live under the self-modify-protected .fak/guard/ (see the overlay file).
	//
	// A malformed overlay REFUSES the overlay wholesale and leaves the base floor
	// standing (byte-identical to the pre-#5181 launch) rather than os.Exit(2)ing like
	// the operator overlays do: this file is agent-writable, so a fatal path would hand
	// the wrapped agent a launch-denial lever over its own operator. Either way the
	// refusal is loud on stderr and recorded in the floor-source provenance.
	selfTightenPath := guardSelfTightenOverlayPath()
	selfTighten, selfTightenErr := loadGuardSelfTightenOverlay(selfTightenPath)
	if selfTightenErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: %v (self-tighten overlay REFUSED; base floor stands)\n", selfTightenErr)
		floorSource += "; agent self-tighten overlay REFUSED (unreadable)"
	} else if admit, class, reason, n := guardApplySelfTightenOverlay(&rt, selfTighten); !admit {
		fmt.Fprintf(os.Stderr, "fak guard: self-tighten overlay %s refused (%s): %s\n", selfTightenPath, class, reason)
		floorSource += guardSelfTightenFloorNote(selfTightenPath, admit, class, reason, n)
	} else {
		floorSource += guardSelfTightenFloorNote(selfTightenPath, admit, class, reason, n)
	}
	// The adjudicator runs in this parent process. Declare the narrow Claude
	// scratch tree here so structural write/delete gates can prove containment;
	// never widen this default to the whole OS temp directory. Whatever is
	// declared — this default or an operator's override — is then expanded to
	// carry BOTH host spellings of each root, because those gates prove
	// containment by string comparison and Git Bash spells the drive-letter
	// directory `/c/…` (see guardScratchpadRootsValue: an alias renames a root,
	// it never adds one, so the default stays exactly this narrow).
	declaredScratch := strings.TrimSpace(os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS"))
	if declaredScratch == "" {
		declaredScratch = filepath.Join(os.TempDir(), "claude")
	}
	_ = os.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", guardScratchpadRootsValue(declaredScratch))
	// The core-lock-all posture (#5423) is NOT applied to this assembly — the launch
	// floor is what the posture clamps, not an amendment to it, so gating here would
	// refuse the session its own first floor. It is announced in the provenance so the
	// banner states the posture the operator is running under, and every LIVE amendment
	// site below (guardReloadDefaultFloor, applyPolicyRuntimeLocked) enforces it.
	if guardCoreLockAllActive() {
		floorSource += "; --core-lock-all ACTIVE (session is ratchet-tighten-only: no channel may widen this floor)"
	}
	effectiveAllow := mergeLaunchAllowOverlays(allowOverlay, launchGrant)
	policyDigest = guardEffectivePolicyDigest(policyBytes, effectiveAllow, denyOverlay)
	rt = protectGuardPolicyConfig(rt, append(guardAllowOverlayLayerPaths(), denyPath, policyPath)...)
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	dur = time.Since(tPolicy)
	return rt, floorSource, policyDigest, dur
}

// guardReloadDefaultFloor re-derives and re-applies the built-in guard capability floor
// (the embedded guardDefaultPolicyJSON) unioned with the operator allow overlay — the
// empty-path (no --policy) branch of guardResolvePolicy, expressed as a hot reload. It
// exists so POST /v1/fak/policy/reload works on the MOST COMMON guard launch (#3957):
// the default-floor guard used to wire a nil reloader, so the route 404'd and `fak guard
// allow`'s "or POST .../policy/reload on a running gateway" advice was false — an overlay
// edit demanded a relaunch. A malformed overlay is TOLERATED here (not fatal), matching
// reloadPolicy's on-reload overlay handling: the loud failure already fired at launch,
// and a live gateway must not wedge over a bad out-of-band edit.
func guardReloadDefaultFloor() (policy.Runtime, string, error) {
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		return policy.Runtime{}, "", err
	}
	denyPath := guardDenyOverlayPath()
	overlayWarning := ""
	if ov, _, ovErr := loadGuardAllowOverlayLayers(); ovErr == nil {
		guardApplyAllowOverlay(&rt, ov)
	} else {
		overlayWarning = "overlay_error: " + ovErr.Error()
	}
	applyLaunchToolGrant(&rt)
	if ov, ovErr := loadGuardDenyOverlay(denyPath); ovErr == nil {
		guardApplyDenyOverlay(&rt, ov)
	} else if overlayWarning == "" {
		overlayWarning = "deny_overlay_error: " + ovErr.Error()
	} else {
		overlayWarning += "\ndeny_overlay_error: " + ovErr.Error()
	}
	rt = protectGuardPolicyConfig(rt, append(guardAllowOverlayLayerPaths(), denyPath)...)
	// CORE-LOCK-ALL (#5423). This is the reload path an ordinary `fak guard -- claude`
	// actually takes — the allow watcher and POST /v1/fak/policy/reload both land here —
	// and it had NO widening gate of its own: an edit to the operator allow overlay was
	// installed live, unconditionally. Under --core-lock-all that is exactly the move the
	// posture exists to refuse, so the re-derived floor is classified against the LIVE one
	// and a widening is refused BEFORE SetPolicy, leaving the last-good floor standing.
	// The refusal is an error (not a warning) so the watcher reports "rejected; keeping
	// last-good floor" and the reload route answers non-2xx rather than silently no-oping.
	current := adjudicator.Default.PolicySnapshot()
	rt.Adjudicator.Posture = current.Posture
	if admit, reason := guardCoreLockAllAdmitAmendment(current, rt.Adjudicator); !admit {
		err := fmt.Errorf("guard floor reload refused: %s", reason)
		journal.Active().AppendConfigSwap(journal.ConfigSwapFloor, "built-in guard floor", guardPolicyDigest(guardDefaultPolicyJSON), journal.ConfigSwapRejected, err.Error())
		return policy.Runtime{}, "", err
	}
	widening := diffPolicyWidening(current, rt.Adjudicator)
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	// Audit parity with the --policy reload path (reloadPolicy): the security boundary was
	// just re-applied, so record which bytes are authoritative. The embedded floor digest
	// is stable; the mutable part an operator changes between reloads is the overlay, folded
	// into rt above. journal.Active() no-ops on an unjournaled run, keeping it byte-identical.
	j := journal.Active()
	j.AppendConfigSwap(journal.ConfigSwapFloor, "built-in guard floor", guardPolicyDigest(guardDefaultPolicyJSON), journal.ConfigSwapOK, "")
	recordLiveWidenings(j, current, widening, strings.Join(guardAllowOverlayLayerPaths(), ","), "operator allow overlay reloaded")
	return rt, overlayWarning, nil
}

func protectGuardPolicyConfig(rt policy.Runtime, paths ...string) policy.Runtime {
	seen := map[string]bool{}
	for _, existing := range rt.Adjudicator.SelfModifyGlobs {
		seen[filepath.ToSlash(existing)] = true
	}
	for _, raw := range paths {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			continue
		}
		clean := filepath.ToSlash(filepath.Clean(raw))
		if !seen[clean] {
			rt.Adjudicator.SelfModifyGlobs = append(rt.Adjudicator.SelfModifyGlobs, clean)
			seen[clean] = true
		}
	}
	return rt
}
