package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// guardFloor bundles the installed capability floor and its provenance/timing, so
// cmdGuard can unpack them to the same local names it used before the installGuardFloor
// extraction.
type guardFloor struct {
	rt           policy.Runtime
	floorSource  string
	policyDigest string
	dur          time.Duration
}

// installGuardFloor installs the capability floor (an explicit --policy file, else the
// embedded guard floor), unions the operator allow overlay on top, defaults the
// scratchpad carve-out, and applies the runtime. Pure code motion out of cmdGuard;
// behavior (including the fail-loud os.Exit paths) is unchanged.
func installGuardFloor(policyPath string) guardFloor {
	// An explicit --policy file wins; otherwise the embedded guard floor. With NO floor the
	// kernel default-denies every tool and the wrapped agent can do nothing — so guard
	// ALWAYS loads one, fail-loud.
	var (
		rt           policy.Runtime
		err          error
		floorSource  string
		policyBytes  []byte
		policyDigest string
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
	// Default the out-of-tree-write floor's scratchpad carve-out to the Claude Code
	// harness scratchpad tree, so a sanctioned build/write INTO the scratchpad — which
	// lives OUTSIDE the repo and is thus reachable only via `..` — is not false-POLICY_BLOCK'd
	// by the `-o ..` / redirect / cp-family rules (internal/adjudicator/outoftree.go reads this
	// env). The adjudicator runs in THIS parent process, so the signal must live in the PARENT
	// env, not the child's. An operator/harness override is respected; the default is the narrow
	// `<temp>/claude` subtree, never the whole temp dir — the pinned `../../tmp/exfil` exfil DENY
	// targets /tmp, which stays outside this root, so the scratchpad unblocks without weakening it.
	if strings.TrimSpace(os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS")) == "" {
		_ = os.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", os.TempDir()+"/claude")
	}
	policyDigest = guardPolicyDigest(policyBytes)
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	return guardFloor{rt: rt, floorSource: floorSource, policyDigest: policyDigest, dur: time.Since(tPolicy)}
}

// resolveGuardLocalBackend handles the --local auto-detect path: probe for a running local
// OpenAI-compatible server and, on detection, rewrite provider/base-URL/model (via the
// passed flag pointers) and apply any provider tuning. Returns the detect duration for the
// boot timeline. Pure code motion out of cmdGuard; behavior (including the fail-loud
// os.Exit paths) is unchanged.
func resolveGuardLocalBackend(localAuto, localModel bool, provider, baseURL, model *string, remoteBase string, quiet bool) time.Duration {
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

// guardInKernelLoad bundles the loaded in-kernel model surface, so cmdGuard can unpack it
// to the same local names it used before the loadGuardInKernelModel extraction.
type guardInKernelLoad struct {
	model        *fakmodel.Model
	tok          *tokenizer.Tokenizer
	q4k          bool
	backend      compute.Backend
	profile      *gateway.ModelLoadProfile
	phase        gateway.StartupPhase
	tokenizerDur time.Duration
}

// loadGuardInKernelModel resolves the --gguf alias/URI (downloading on demand), picks the
// decode backend, and loads the weights + tokenizer through the same serve loaders
// `fak serve --gguf` uses. Returns the zero struct in the proxy path (localModel false).
// Pure code motion out of cmdGuard; behavior (including the fail-loud os.Exit paths) is
// unchanged.
func loadGuardInKernelModel(localModel, localAlongside, quiet bool, ggufPath *string, gpuBackend, tokPath string, contextBudgetLimit int, localAlias string) guardInKernelLoad {
	var r guardInKernelLoad
	if !localModel {
		return r
	}
	// Alias (`qwen2.5:7b`) → target ref, then an hf:// URI → a locally cached file.
	if resolved, expanded := modelreg.Resolve(*ggufPath); expanded {
		fmt.Fprintf(os.Stderr, "fak guard: --gguf %s → %s\n", *ggufPath, resolved)
		*ggufPath = resolved
	}
	if hfhub.IsURI(*ggufPath) {
		fctx, fstop := signal.NotifyContext(context.Background(), os.Interrupt)
		resolved, ferr := hfhub.FetchURI(fctx, *ggufPath, os.Stderr)
		fstop()
		if ferr != nil {
			fmt.Fprintf(os.Stderr, "fak guard: --gguf %v\n", ferr)
			os.Exit(1)
		}
		*ggufPath = resolved
	}
	var berr error
	r.backend, berr = resolveServeChatBackend(gpuBackend)
	if berr != nil {
		fmt.Fprintln(os.Stderr, "fak guard:", berr)
		os.Exit(2)
	}
	if r.backend != nil {
		fmt.Fprintf(os.Stderr, "fak guard: in-kernel decode → device backend %q\n", r.backend.Name())
	}
	r.model, r.q4k, r.profile, r.phase = loadServeInKernelModel(*ggufPath, r.backend, false, contextBudgetLimit, nil, 1)
	if r.model == nil {
		fmt.Fprintf(os.Stderr, "fak guard: failed to load %q into the in-kernel engine\n", *ggufPath)
		os.Exit(1)
	}
	tTok := time.Now()
	var tokOK bool
	r.tok, tokOK = resolveServeTokenizer(tokPath, *ggufPath)
	r.tokenizerDur = time.Since(tTok)
	if !tokOK || r.tok == nil {
		fmt.Fprintf(os.Stderr, "fak guard: %q has no usable tokenizer; pass --tokenizer or use a GGUF with an embedded tokenizer\n", *ggufPath)
		os.Exit(1)
	}
	if localAlongside && !quiet {
		fmt.Fprintf(os.Stderr, "fak guard: ALONGSIDE mode — model id %q (or \"local\") decodes in-kernel on this box; every other model id proxies to the API upstream as usual\n", localAlias)
	}
	return r
}
