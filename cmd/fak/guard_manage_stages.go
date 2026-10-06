package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/logvault"
)

// applyGuardDeadlineEnvOverrides lets FAK_GUARD_SOFT_DEADLINE_LEAD,
// FAK_GUARD_COMMIT_GRACE_PERIOD and FAK_GUARD_CHILD_STOP_GRACE override the parsed
// deadline flags when they hold a valid duration.
func applyGuardDeadlineEnvOverrides(softDeadlineLead, commitGracePeriod, childStopGrace *time.Duration) {
	if env := strings.TrimSpace(os.Getenv("FAK_GUARD_SOFT_DEADLINE_LEAD")); env != "" {
		if d, err := time.ParseDuration(env); err == nil {
			*softDeadlineLead = d
		}
	}
	if env := strings.TrimSpace(os.Getenv("FAK_GUARD_COMMIT_GRACE_PERIOD")); env != "" {
		if d, err := time.ParseDuration(env); err == nil {
			*commitGracePeriod = d
		}
	}
	if env := strings.TrimSpace(os.Getenv("FAK_GUARD_CHILD_STOP_GRACE")); env != "" {
		if d, err := time.ParseDuration(env); err == nil {
			*childStopGrace = d
		}
	}
}

// guardResolveLocalAuto implements --local: auto-detect a running local
// OpenAI-compatible server (Ollama/LM Studio/Qwen3.6 dogfood/llama.cpp) and wire the
// upstream to it. This is a PROXY path (the server
// is external), so on detection we set provider=openai + base-URL=<detected>/v1 exactly
// as if the user had typed those flags, and the standard resolution flow below handles it.
// Precedence:
//   - --gguf wins (it is the no-server in-kernel path); --local is then a no-op.
//   - --base-url / --remote-serve conflict (the detected server IS the upstream).
//   - nothing detected + no --gguf -> fail loud with how to start a server.
//
// It returns the detection duration for the boot timeline (zero when no probe ran).
func guardResolveLocalAuto(localAuto, localModel, quiet bool, remoteBase string, provider, baseURL, model *string) time.Duration {
	var localDetectDur time.Duration
	if localAuto && !localModel {
		if strings.TrimSpace(*baseURL) != "" || remoteBase != "" {
			fmt.Fprintln(os.Stderr, "fak guard: --local auto-detects the upstream server, so it is mutually exclusive with --base-url / --remote-serve — pass only one")
			os.Exit(2)
		}
		tLocal := time.Now()
		detBase, detModel, detLabel, found := guardDetectLocalBackend()
		localDetectDur = time.Since(tLocal)
		if !found {
			fmt.Fprintln(os.Stderr, guardLocalNothingDetectedMessage())
			os.Exit(2)
		}
		*provider, *baseURL = "openai", detBase
		if strings.TrimSpace(*model) == "" {
			*model = detModel
		}
		extraApplied, extraAlreadySet, _, extraErr := guardApplyLocalProviderExtraBody(detLabel, *model, os.Getenv, os.Setenv)
		if extraErr != nil {
			fmt.Fprintf(os.Stderr, "fak guard: --local could not apply Qwen3.6 provider tuning: %v\n", extraErr)
			os.Exit(2)
		}
		if !quiet {
			fmt.Fprintln(os.Stderr, guardLocalDetectedBanner(detLabel, detBase, detModel))
			switch {
			case extraApplied:
				fmt.Fprintln(os.Stderr, "-> local tuning: Qwen3.6 provider extra body enabled (top_k=20, preserve_thinking=true)")
			case extraAlreadySet:
				fmt.Fprintln(os.Stderr, "-> local tuning: using existing FAK_PROVIDER_EXTRA_BODY_JSON")
			}
		}
	} else if localAuto && localModel && !quiet {
		fmt.Fprintln(os.Stderr, "fak guard: --gguf is set, so --local is ignored (the in-kernel model is the upstream)")
	}
	return localDetectDur
}

// guardCheckRemoteServe validates --remote-serve against --base-url/--provider and
// preflights the remote box, failing loud on a conflict or an unreachable serve. It
// returns the preflight duration for the boot timeline (zero when --remote-serve is unset).
func guardCheckRemoteServe(remoteBase, baseURL, provider string) time.Duration {
	if remoteBase == "" {
		return 0
	}
	if strings.TrimSpace(baseURL) != "" && strings.TrimSpace(baseURL) != remoteBase {
		fmt.Fprintf(os.Stderr, "fak guard: --remote-serve and --base-url disagree (%s vs %s) — pass only one\n", remoteBase, strings.TrimSpace(baseURL))
		os.Exit(2)
	}
	if p := strings.ToLower(strings.TrimSpace(provider)); p == "anthropic" {
		fmt.Fprintln(os.Stderr, "fak guard: --remote-serve uses the OpenAI-compatible wire fak serve exposes; drop --provider anthropic")
		os.Exit(2)
	}
	// Preflight: a remote serve that is not answering is the most common failure here
	// (box not started, wrong port). Fail loud with the next step, mirroring the
	// exec.LookPath check above, rather than binding a gateway that 502s on first call.
	tRemote := time.Now()
	preflightErr := guardPreflightRemoteServe(remoteBase)
	remotePreflightDur := time.Since(tRemote)
	if preflightErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: --remote-serve %s is not reachable: %v\n  start it on the box with `fak serve --gguf <weights> --backend cuda --addr 0.0.0.0:8080`, or check the host/port.\n", remoteBase, preflightErr)
		os.Exit(2)
	}
	return remotePreflightDur
}

// validateGuardBudgetFlags fails loud (exit 2) on a negative or inconsistent budget,
// restart, admission, or wall-clock flag before anything binds.
func validateGuardBudgetFlags(contextBudgetTokens, contextBudgetLimit int, resetOnBudget, restartOnBudget bool, restartLimit, nativeAdmissionTokenBudget int, maxDurationLimit time.Duration) {
	if contextBudgetTokens < 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --context-budget-tokens must be non-negative")
		os.Exit(2)
	}
	if resetOnBudget && contextBudgetLimit <= 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --reset-on-budget requires --context-budget-tokens N")
		os.Exit(2)
	}
	if restartOnBudget && contextBudgetLimit <= 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --restart-on-budget requires --context-budget-tokens N")
		os.Exit(2)
	}
	if restartLimit < 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --restart-limit must be non-negative")
		os.Exit(2)
	}
	// Same wording serve refuses a non-positive --native-admission-token-budget with:
	// a typo must fail loud at launch, never silently boot a seat whose scheduler
	// budget was not the one the operator declared.
	if nativeAdmissionTokenBudget <= 0 {
		fmt.Fprintf(os.Stderr, "fak guard: --native-admission-token-budget must be positive (got %d)\n", nativeAdmissionTokenBudget)
		os.Exit(2)
	}
	if maxDurationLimit < 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --max-duration must be non-negative")
		os.Exit(2)
	}
}

// guardBootStartupPhases assembles THIS guard process's boot timeline: flag-parse and
// policy-load always fire; optional phases are omitted when their duration is zero.
func guardBootStartupPhases(parseDur, policyDur, localDetectDur, remotePreflightDur, upstreamResolveDur, pathLookupDur time.Duration, loadPhase gateway.StartupPhase, tokenizerLoadDur, listenDur time.Duration) []gateway.StartupPhase {
	startupPhases := []gateway.StartupPhase{
		{Name: "flag-parse", Dur: parseDur},
		{Name: "policy-load", Dur: policyDur},
	}
	if localDetectDur > 0 {
		startupPhases = append(startupPhases, gateway.StartupPhase{Name: "local-detect", Dur: localDetectDur})
	}
	if remotePreflightDur > 0 {
		startupPhases = append(startupPhases, gateway.StartupPhase{Name: "remote-serve-preflight", Dur: remotePreflightDur})
	}
	startupPhases = append(startupPhases, gateway.StartupPhase{Name: "upstream-resolve", Dur: upstreamResolveDur})
	startupPhases = append(startupPhases, gateway.StartupPhase{Name: "path-lookup", Dur: pathLookupDur})
	if loadPhase.Name != "" {
		startupPhases = append(startupPhases, loadPhase)
	}
	if tokenizerLoadDur > 0 {
		startupPhases = append(startupPhases, gateway.StartupPhase{Name: "tokenizer-load", Dur: tokenizerLoadDur})
	}
	startupPhases = append(startupPhases, gateway.StartupPhase{Name: "listener-bind", Dur: listenDur})
	return startupPhases
}

// startGuardResourceSampler starts the kernel-half harness resource sampler when
// --resource-stats is on and exposes it on the gateway; nil when disabled.
func startGuardResourceSampler(resourceStats bool, netCounter *harnessres.CountingListener, chatBackend compute.Backend, srv *gateway.Server) *harnessres.Sampler {
	var resSampler *harnessres.Sampler
	if resourceStats {
		resSampler = harnessres.New()
		// Feed the kernel half's network axis from the listener counter installed at bind
		// time (#2049). Set BEFORE Start so the first sample already carries it.
		if netCounter != nil {
			resSampler.SetNetworkProvider(func() (rx, tx uint64, ok bool) {
				rx, tx = netCounter.Bytes()
				return rx, tx, true
			})
		}
		// Feed the GPU/accelerator VRAM axis when a model runs IN-KERNEL (--gguf/--backend):
		// the harness's hardware footprint then includes the device. The default proxy path
		// has no local GPU, so the provider reports ok=false and the axis stays honestly n/a
		// (#2052). VRAM PREFERS the same compute HAL the serve capacity checks use (the
		// in-kernel backend's own device handle); it falls back to nvidia-smi only on a host
		// where the handle cannot report — the fail-soft fallback the issue names.
		if chatBackend != nil {
			resSampler.SetGPUProvider(func() (used, total uint64, ok bool) {
				t, free, known := compute.DeviceMemoryInfo(chatBackend)
				var smi []compute.GPUStat
				if !known || t <= 0 {
					smi, _ = compute.SystemGPUStats() // fail-soft; nil → axis stays n/a
				}
				return compute.HarnessGPUVRAM(t, free, known, smi)
			})
			// Feed the GPU utilization axis. The in-kernel device-handle seam
			// (DeviceMemoryInfo) reports memory only — there is no utilization on it — so
			// this is the accelerator fallback the issue names: per-device VRAM+util folded
			// to the busiest device's percent. Fail-soft (no probe / timeout /
			// unparseable → ok=false), so the util axis stays honestly n/a rather than a
			// fabricated 0 on a host that lacks the tool (#2052, #11319).
			resSampler.SetGPUUtilProvider(func() (pct float64, ok bool) {
				stats, present := compute.SystemGPUStats()
				if !present {
					return 0, false
				}
				_, _, util, aok := compute.AggregateGPUStats(stats)
				return util, aok
			})
		}
		resSampler.Start(guardResourceSampleInterval)
		// Expose the live harness resource snapshot on the gateway's /metrics as the
		// fak_harness_* family, so a running session's CPU/mem/IO is scrapeable — not
		// only printed at exit (epic #2044 / #2047). Pull-only: rendered per scrape.
		srv.SetHarnessMetricsProvider(func() string { return resSampler.Snapshot().PrometheusText() })
		// Structured twin of the /metrics harness family, on /debug/vars, so the live `fak
		// info` pane can show the kernel CPU/RSS/IO the exit summary prints instead of only
		// scraping Prometheus text. Same pull sampler, converted to the gateway's shape.
		srv.SetSessionHarnessProvider(func() gateway.SessionHarness { return guardHarnessToSession(resSampler.Snapshot()) })
	}
	return resSampler
}

// installGuardLogvaultMetrics exposes the fak_logvault_* gauges on the gateway when a
// capture vault manifest exists on this box.
func installGuardLogvaultMetrics(srv *gateway.Server) {
	if vaultDir := resolveLogvaultDir(repoRoot()); vaultDir != "" {
		if _, statErr := os.Stat(filepath.Join(vaultDir, logvault.ManifestName)); statErr == nil {
			lv := &logvault.Vault{Dir: vaultDir}
			srv.SetLogvaultMetricsProvider(func() string {
				text, err := lv.MetricsText(logvaultMetricsVerifySample, time.Now().UnixNano())
				if err != nil {
					return "" // unreadable manifest: emit nothing rather than a broken family
				}
				return text
			})
		}
	}
}
