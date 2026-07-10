package main

// bench_labtarget.go — resolveBenchBaseURL: the one seam that lets a benchmark's
// --base-url address lab GPU hardware BY NAME (@lab/<model>) instead of a hand-copied
// host:port, reusing the SAME readiness+latency-gated resolver that
// `fak guard --remote-serve` uses (resolveGuardRemoteServe -> resolveLabTarget). This is
// how `fak deepseekbench` / `fak swebench` / `fak webbench` are meant to "use the lab
// hardware properly": a live run cannot start against a box that is not linkstate.Clear /
// AdmitDispatch, nor against a latency-degraded route, and never needs a private box
// coordinate pasted onto the command line.
//
// Wire mapping (why the /v1 re-append): the lab resolver normalizes a target's base_url
// with any trailing /v1 STRIPPED (http://host:8000/v1 -> http://host:8000), whereas the
// benchmark clients build "<base>/chat/completions" (see
// internal/deepseekbench.MeasureStreamed) — so a self-hosted vLLM/SGLang box needs the
// /v1 root restored. guardOpenAIV1Base (guard_provider.go) does exactly that, idempotently.

import "strings"

// resolveBenchBaseURL maps a benchmark --base-url onto a concrete OpenAI-compatible root.
// A plain URL — the hosted-provider default (e.g. https://api.deepseek.com) or an explicit
// http://host:8000/v1 — is returned UNCHANGED (passthrough), so existing usage is
// untouched. An @lab/<model> alias is resolved through the readiness-gated lab-target seam
// and normalized to the /v1 root the benchmark clients append "/chat/completions" to.
//
// Fails CLOSED: an unready / absent / not-found / latency-degraded lab target surfaces the
// resolver's LAB_* structured refusal (empty string, non-nil error) rather than silently
// falling back to a public endpoint — a benchmark never runs against a box the readiness
// gate has not cleared.
func resolveBenchBaseURL(raw string) (string, error) {
	if !isLabTargetAlias(strings.TrimSpace(raw)) {
		return raw, nil
	}
	base, err := resolveGuardRemoteServe(raw)
	if err != nil {
		return "", err
	}
	return guardOpenAIV1Base(base), nil
}
