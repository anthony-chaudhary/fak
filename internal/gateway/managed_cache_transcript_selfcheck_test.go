package gateway

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Managed-cache self-check on the ACTUAL transcript trajectory (epic #1844 C6, guards the
// #1850 beta-union regression recorded in memory managed-cache-1h-ttl-400-missing-beta).
//
// The existing managed-cache pipeline tests (messages_compact_test.go) pin the contract on a
// MINIMAL 1-block synthetic body. The real Claude Code wire is not that shape: `system` is a
// MULTI-block array whose LAST block carries the cache_control breakpoint, `tools` is a full
// array, and `messages` is a multi-turn conversation with tool_use / tool_result. This file
// drives the REAL served pipeline (prepareServedAnthropicRequest — the same function the live
// gateway runs, not a reimplementation) over that realistic trajectory and proves the whole
// managed-cache contract end to end:
//
//	(1) the 1h upgrade lands on the correct stable head (the LAST system block), not a tools
//	    entry and not an earlier system block;
//	(2) every byte BEFORE the upgraded breakpoint is identical — the cached prefix the provider
//	    keys its hit on is preserved verbatim on the real multi-block shape;
//	(3) the extended-cache-ttl beta is unioned into the forwarded set while the client's inbound
//	    betas survive (the load-bearing wiring whose absence 400'd a forced --managed-cache on
//	    session upstream — the bug the fleet default now rides on);
//	(4) the decoded conversation body is untouched (the kernel's trust boundary);
//	(5) the forwarded body still decodes as a valid Anthropic request; and
//	(6) the attempt is witnessed on the /metrics outcome counter.
//
// realisticTranscriptBody is a faithful Claude Code POST /v1/messages body written as raw wire
// bytes (not a marshaled map, so key order and block boundaries match the actual trajectory).
// The system head is byte-STABLE: the date block is date-only (no HH:MM) and no block carries a
// per-request UUID, so the stable-head guard admits it and the upgrade fires.
const realisticTranscriptBody = `{"model":"claude-sonnet-4-6","max_tokens":8192,"stream":true,` +
	`"system":[` +
	`{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude."},` +
	`{"type":"text","text":"Today's date is 2026-07-10. The user's OS is darwin. The working dir is a local project checkout."},` +
	`{"type":"text","text":"# Project context\nThis repo builds the fak gateway. Follow the house style.","cache_control":{"type":"ephemeral"}}` +
	`],` +
	`"tools":[` +
	`{"name":"Bash","description":"Run a shell command.","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}},` +
	`{"name":"Read","description":"Read a file.","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}` +
	`],` +
	`"messages":[` +
	`{"role":"user","content":[{"type":"text","text":"List the go files in cmd."}]},` +
	`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_stableid01","name":"Bash","input":{"command":"ls cmd"}}]},` +
	`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_stableid01","content":"main.go\nvcache.go"}]},` +
	`{"role":"user","content":[{"type":"text","text":"Now open main.go."}]}` +
	`]}`

// rawClientTranscriptBody is a realistic body from a caller that sends NO cache_control anywhere
// (a minimal SDK, not the Claude Code CLI). Under managed cache this must still reach the 1h tier
// via the composed place-then-upgrade path (#2175): the stable system head gets a breakpoint
// placed AND upgraded as one transform.
const rawClientTranscriptBody = `{"model":"claude-sonnet-4-6","max_tokens":4096,"stream":true,` +
	`"system":[` +
	`{"type":"text","text":"You are a helpful assistant embedded in a build tool."},` +
	`{"type":"text","text":"# Repo policy\nPrefer small, reviewable diffs. Never invent APIs."}` +
	`],` +
	`"messages":[{"role":"user","content":[{"type":"text","text":"Summarize the repo."}]}]}`

// inboundClientBetas are the betas the wrapped claude CLI negotiates — deliberately NOT the
// extended-cache-ttl beta the 1h upgrade requires, so the union (not replace) is observable.
const inboundClientBetas = "claude-code-20250219,fine-grained-tool-streaming-2025-05-14"

func inboundTranscriptRequest() *http.Request {
	r := httptest.NewRequest("POST", "/v1/messages", nil)
	r.Header.Set("anthropic-beta", inboundClientBetas)
	return r
}

// TestManagedCacheSelfCheckRealisticTranscript_UpgradesLastSystemBlock proves the flagship
// managed-cache contract on a realistic multi-block Claude Code trajectory: the upgrade lands on
// the last system block, the cached prefix is byte-identical, the beta is unioned, the
// conversation body is untouched, the body re-decodes, and the attempt is witnessed.
func TestManagedCacheSelfCheckRealisticTranscript_UpgradesLastSystemBlock(t *testing.T) {
	raw := []byte(realisticTranscriptBody)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("fixture must decode as a valid Anthropic request: %v", err)
	}

	s := anthropicPassthroughServer(0) // compaction OFF: isolate the managed-cache transform
	s.cacheTTL1H.Store(true)           // --managed-cache ACTIVE
	s.metrics = newGatewayMetrics(time.Now())

	prep := s.prepareServedAnthropicRequest(context.Background(), inboundTranscriptRequest(), req, "", servedSessionTurn{})

	// (1) the 1h upgrade landed on the LAST system block's breakpoint.
	if !bytes.Contains(req.Raw, []byte(`"cache_control":{"type":"ephemeral","ttl":"1h"}`)) {
		t.Fatalf("managed cache did not upgrade the stable-head breakpoint to 1h on a realistic body:\n%s", req.Raw)
	}
	// The fixture has exactly one cache_control (on the last system block). Prove the upgrade
	// went there and nowhere else — no second breakpoint sprouted on a tools entry.
	if n := bytes.Count(req.Raw, []byte(`"cache_control"`)); n != 1 {
		t.Fatalf("expected exactly one cache_control after upgrade, got %d:\n%s", n, req.Raw)
	}

	// (2) every byte before the breakpoint is preserved verbatim — the cached prefix the
	// provider keys its hit on (all earlier system blocks + tools) is byte-identical.
	cc := bytes.Index(raw, []byte(`"cache_control"`))
	if cc < 0 {
		t.Fatal("fixture sanity: input is missing the stable-head cache_control")
	}
	if !bytes.Equal(raw[:cc], req.Raw[:cc]) {
		t.Fatalf("bytes before the upgraded breakpoint changed — cached prefix not preserved:\nin =%s\nout=%s", raw[:cc], req.Raw[:cc])
	}

	// (3) the extended-cache-ttl beta is unioned in AND the inbound betas survive (union, not
	// replace) — the wiring whose absence 400'd a forced --managed-cache session upstream.
	if !strings.Contains(prep.upstreamBeta, extendedCacheTTLBeta) {
		t.Fatalf("upstreamBeta = %q must union the extended-cache-ttl beta (ttl:1h without it is 400'd upstream)", prep.upstreamBeta)
	}
	for _, want := range []string{"claude-code-20250219", "fine-grained-tool-streaming-2025-05-14"} {
		if !strings.Contains(prep.upstreamBeta, want) {
			t.Errorf("upstreamBeta = %q dropped inbound beta %q — must UNION, not replace", prep.upstreamBeta, want)
		}
	}

	// (4) the conversation body is untouched — the trust boundary the kernel adjudicates. The
	// tool_use/tool_result turns and the trailing user turn survive byte-for-byte.
	for _, want := range []string{`"tool_use_id":"toolu_stableid01"`, `"text":"Now open main.go."`} {
		if !bytes.Contains(req.Raw, []byte(want)) {
			t.Errorf("forwarded body dropped conversation content %q — managed cache must not touch messages", want)
		}
	}

	// (5) the forwarded body still decodes as a valid Anthropic request.
	if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
		t.Fatalf("forwarded body no longer decodes after the upgrade: %v", err)
	}

	// (6) the attempt is witnessed on the outcome counter as a plain upgrade (an existing
	// breakpoint was extended, not placed).
	if n := s.metrics.ttlUpgrades["upgraded"]; n != 1 {
		t.Fatalf("expected the outcome counter to witness exactly one 'upgraded', got %d", n)
	}
	if n := s.metrics.ttlUpgrades[agent.TTLUpgradeReasonNoStableBreakpoint]; n != 0 {
		t.Fatalf("no_stable_breakpoint must stay flat on a body that shipped a breakpoint, got %d", n)
	}
}

// TestManagedCacheSelfCheckRealisticTranscript_OffIsByteIdentity proves the negative half of the
// self-check: with managed cache OFF the same realistic trajectory forwards byte-for-byte
// unchanged and carries no gratuitous extended-cache-ttl beta.
func TestManagedCacheSelfCheckRealisticTranscript_OffIsByteIdentity(t *testing.T) {
	raw := []byte(realisticTranscriptBody)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	orig := append([]byte(nil), req.Raw...)

	s := anthropicPassthroughServer(0) // cacheTTL1H stays false → lever OFF
	s.metrics = newGatewayMetrics(time.Now())

	prep := s.prepareServedAnthropicRequest(context.Background(), inboundTranscriptRequest(), req, "", servedSessionTurn{})

	if !bytes.Equal(req.Raw, orig) {
		t.Fatalf("managed cache OFF must forward the body byte-for-byte unchanged:\nin =%s\nout=%s", orig, req.Raw)
	}
	if strings.Contains(prep.upstreamBeta, extendedCacheTTLBeta) {
		t.Fatalf("managed cache OFF must not add the extended-cache-ttl beta, got upstreamBeta = %q", prep.upstreamBeta)
	}
}

// TestManagedCacheSelfCheckRawClient_PlacesThenUpgrades proves a caller that sends NO
// cache_control still reaches the 1h tier on a realistic body: the composed place-then-upgrade
// path (#2175) places a breakpoint on the stable system head and upgrades it as one transform,
// witnessed distinctly as placed_and_upgraded.
func TestManagedCacheSelfCheckRawClient_PlacesThenUpgrades(t *testing.T) {
	raw := []byte(rawClientTranscriptBody)
	req, err := agent.DecodeAnthropicMessagesRequest(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if bytes.Contains(raw, []byte(`"cache_control"`)) {
		t.Fatal("fixture sanity: the raw-client body must carry NO cache_control")
	}

	s := anthropicPassthroughServer(0) // compaction OFF and vcacheAnchor OFF: the composed
	s.cacheTTL1H.Store(true)           // place-then-upgrade path must stand on its own.
	s.metrics = newGatewayMetrics(time.Now())

	prep := s.prepareServedAnthropicRequest(context.Background(), inboundTranscriptRequest(), req, "", servedSessionTurn{})

	// A breakpoint was placed AND upgraded to 1h on the stable head.
	if !bytes.Contains(req.Raw, []byte(`"cache_control":{"type":"ephemeral","ttl":"1h"}`)) {
		t.Fatalf("raw-client body did not get a 1h breakpoint placed+upgraded on its stable head:\n%s", req.Raw)
	}
	// The beta is unioned so the placed ttl:1h is accepted upstream.
	if !strings.Contains(prep.upstreamBeta, extendedCacheTTLBeta) {
		t.Fatalf("upstreamBeta = %q must union the extended-cache-ttl beta after a place-then-upgrade", prep.upstreamBeta)
	}
	// The forwarded body still decodes.
	if _, err := agent.DecodeAnthropicMessagesRequest(req.Raw); err != nil {
		t.Fatalf("forwarded body no longer decodes after place-then-upgrade: %v", err)
	}
	// The composed path is witnessed as its own outcome, not folded into upgrade-only or the
	// no_stable_breakpoint bail.
	if n := s.metrics.ttlUpgrades[cacheTTLUpgradePlacedAndUpgraded]; n != 1 {
		t.Fatalf("expected exactly one placed_and_upgraded outcome, got %d", n)
	}
	if n := s.metrics.ttlUpgrades[agent.TTLUpgradeReasonNoStableBreakpoint]; n != 0 {
		t.Fatalf("no_stable_breakpoint must stay flat on the composed path, got %d", n)
	}
}
