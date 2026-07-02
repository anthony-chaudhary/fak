package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/guardtrace"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

// guardTraceFixturePath is the shared end-to-end fixture, authored in the gateway package's
// testdata and reused by the CLI replay so the operator watches the SAME trace the gateway
// test asserts on. The module root is the repo root, so the relative path resolves from
// cmd/fak.
const guardTraceFixturePath = "../../internal/gateway/testdata/guard-trace-e2e.json"

// TestGuardReplayShippedFloorDeniesEveryFixtureDanger is the anti-drift witness: the REAL
// shipped guard floor (guardDefaultPolicyJSON, the one --replay-trace installs by default)
// must DENY every call the fixture marks "deny", with the reason the fixture declares. If a
// future edit to the shipped floor stops firing on a danger class the fixture records, this
// fails — so the replay can never quietly demo a floor weaker than production.
func TestGuardReplayShippedFloorDeniesEveryFixtureDanger(t *testing.T) {
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatalf("parse shipped guard floor: %v", err)
	}
	f, err := guardtrace.LoadFixture(guardTraceFixturePath)
	if err != nil {
		t.Fatalf("load fixture: %v", err)
	}

	// Decide each fixture call directly against the shipped floor (no gateway needed for the
	// drift check — this isolates the floor itself). A FRESH adjudicator over the live ABI
	// resolver, NOT adjudicator.Default, so the check never mutates the process floor other
	// cmd/fak tests rely on (the same discipline TestGuardDefaultPolicyDeniesDangerAllowsBenign
	// uses).
	adj := adjudicator.New(rt.Adjudicator)
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatal("no Ref resolver registered (internal/registrations blank import missing)")
	}

	for _, turn := range f.Turns {
		for _, c := range turn.Calls {
			ref, err := res.Put(context.Background(), []byte(c.ArgString()))
			if err != nil {
				t.Fatalf("put args for %s: %v", c.ID, err)
			}
			v := adj.Adjudicate(context.Background(), &abi.ToolCall{Tool: c.Tool, Args: ref})
			if c.ExpectAllow() {
				if v.Kind != abi.VerdictAllow {
					t.Errorf("call %s (%s %s): shipped floor gave %v, want ALLOW", c.ID, c.Tool, c.ArgPreview(), v.Kind)
				}
				continue
			}
			if v.Kind != abi.VerdictDeny {
				t.Errorf("call %s (%s %s): shipped floor gave %v, want DENY — the production floor stopped firing on a fixture danger class", c.ID, c.Tool, c.ArgPreview(), v.Kind)
			}
			if c.Reason != "" {
				if got := abi.ReasonName(v.Reason); got != c.Reason {
					t.Errorf("call %s: shipped floor deny reason = %q, want %q", c.ID, got, c.Reason)
				}
			}
		}
	}
}

// TestGuardReplayRunsCleanOnBothWires drives the full runGuardReplay end to end over the
// shared fixture on both wires and asserts it reports success (exit 0) and prints the
// per-call verdicts, the exit summary, and the verified journal — the observable
// operator-facing path working, not just the units under it.
func TestGuardReplayRunsCleanOnBothWires(t *testing.T) {
	// runGuardReplay programmatically enables the process-global decision journal; reset it
	// after so it does not leak into a sibling test that assumes a clean boot (e.g.
	// TestGuardEnableAuditEnablesVerifiableTrail).
	t.Cleanup(journal.ResetActiveForTest)
	for _, wire := range []string{"anthropic", "openai"} {
		t.Run(wire, func(t *testing.T) {
			t.Cleanup(journal.ResetActiveForTest)
			var sb strings.Builder
			code := runGuardReplay(guardTraceFixturePath, wire, "", &sb)
			out := sb.String()
			if code != 0 {
				t.Fatalf("runGuardReplay(%s) exit = %d, want 0\n%s", wire, code, out)
			}
			for _, want := range []string{
				"fak guard --replay-trace",
				"DENY[POLICY_BLOCK]",
				"kernel decision(s)",
				"journal chain verified",
				"every call landed on its expected disposition",
			} {
				if !strings.Contains(out, want) {
					t.Errorf("%s replay output missing %q:\n%s", wire, want, out)
				}
			}
			// The glanceable report must not leak the dangerous command text into a banner
			// line beyond the bounded arg preview (the preview is fine; a full unbounded dump
			// is not). The fixture's rm path is short enough to preview, so assert the long
			// secret-ish content never appears.
			if strings.Contains(out, "ssh-rsa AAAA attacker") {
				t.Errorf("%s replay leaked full write content into the report:\n%s", wire, out)
			}
		})
	}
}

func TestGuardReplayWritesExplicitContextSnapshot(t *testing.T) {
	t.Cleanup(journal.ResetActiveForTest)
	dir := t.TempDir()
	fixturePath := filepath.Join(dir, "ctxview-replay.json")
	snapPath := filepath.Join(dir, "vcache-turns.jsonl")
	t.Setenv(vcachesnapshot.EnvPath, snapPath)

	f := guardtrace.Fixture{
		SliceID: "ctxview-replay",
		Turns: []guardtrace.Turn{{
			Messages: []guardtrace.RequestMessage{
				{Role: "system", Content: "You are a coding agent under fak guard."},
				{Role: "user", Content: "rotate the auth token and then check the refund policy"},
				{Role: "assistant", Content: strings.Repeat("weather sunny unrelated padding ", 5000)},
				{Role: "user", Content: "what is the auth token rotation and refund window"},
			},
			Usage: guardtrace.Usage{InputTokens: 240, OutputTokens: 12, CacheReadInputTokens: 1800},
			Calls: []guardtrace.Call{{
				ID:    "call_read_policy",
				Tool:  "Read",
				Args:  json.RawMessage(`{"file_path":"README.md"}`),
				Class: "allow",
			}},
		}},
	}
	raw, err := json.Marshal(f)
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	if err := os.WriteFile(fixturePath, raw, 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	var sb strings.Builder
	if code := runGuardReplay(fixturePath, "openai", "", &sb); code != 0 {
		t.Fatalf("runGuardReplay exit = %d, want 0\n%s", code, sb.String())
	}
	if !strings.Contains(sb.String(), "wrote vcache snapshot") {
		t.Fatalf("replay output did not point at the explicit vcache snapshot:\n%s", sb.String())
	}

	turns, ok, err := vcachesnapshot.Read(snapPath)
	if err != nil {
		t.Fatalf("read replay snapshot: %v", err)
	}
	if !ok || len(turns) == 0 {
		t.Fatalf("replay snapshot missing turns: ok=%v turns=%+v", ok, turns)
	}
	if turns[0].ContextEvents != 1 || turns[0].ContextDroppedTurns != 1 || turns[0].ContextShedTokens <= 0 {
		t.Fatalf("replay context evidence = events:%d dropped:%d shed:%d, want 1/1/>0",
			turns[0].ContextEvents, turns[0].ContextDroppedTurns, turns[0].ContextShedTokens)
	}

	var out, errb bytes.Buffer
	if code := runVCache(&out, &errb, []string{"score", "--json"}); code != 0 && code != 1 {
		t.Fatalf("vcache score exit=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("vcache score emitted invalid json: %v\n%s", err, out.String())
	}
	if !rep.Planes.ContextWitnessed.Available || rep.AgenticActivation.ContextEvents != 1 {
		t.Fatalf("score context plane=%+v activation=%+v, want replay snapshot context witness",
			rep.Planes.ContextWitnessed, rep.AgenticActivation)
	}
}

// TestGuardReplayUnknownWireRejected proves a bad --replay-wire fails loud (exit 2) rather
// than silently defaulting.
func TestGuardReplayUnknownWireRejected(t *testing.T) {
	var sb strings.Builder
	if code := runGuardReplay(guardTraceFixturePath, "gemini", "", &sb); code != 2 {
		t.Fatalf("unknown wire exit = %d, want 2\n%s", code, sb.String())
	}
}
