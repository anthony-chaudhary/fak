package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/gateway"
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
const guardContextTraceFixturePath = "testdata/guard-trace-context-e2e.json"

// installEmptyReplayLeaseAuthority supplies the host-owned half of the gateway's workspace
// write admission for replay tests whose subject is the floor + journal + report, not lease
// ownership. The provider cmd/fak installs at init (serveWorkspaceLeaseAdmission) shells out
// to the `dos` CLI and reads the process cwd's live lease refs, so without this the fixture's
// benign writes land on DEFAULT_DENY ("workspace lease authority read failed") on any host
// without `dos` on PATH (CI), or on LEASE_HELD where a peer holds a lease over the tree. The
// lease boundary itself is witnessed by TestFakAdjudicateUsesRealDOSWorkspaceLease and the
// internal/gateway admission tests; the gateway twin of this replay
// (internal/gateway guard_trace_e2e_test.go) pins the same empty authority.
func installEmptyReplayLeaseAuthority(t *testing.T) {
	t.Helper()
	root := t.TempDir()
	// Admission canonicalizes a new leaf only through an existing parent, so give the
	// workspace the .fak/ state dir a fak-managed checkout has; the fixture's benign writes
	// (.fak/guard-trace-*.txt, .fak/authorized_keys) land there.
	if err := os.Mkdir(filepath.Join(root, ".fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	gateway.SetLeasePlaneProviders(serveLeasePlaneLeases, func(context.Context) (gateway.LeasePresenceView, error) {
		return gateway.LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	gateway.SetWorkspaceLeaseAdmissionProvider(func(context.Context) (gateway.WorkspaceLeaseAdmissionView, error) {
		return gateway.WorkspaceLeaseAdmissionView{WorkspaceRoot: root}, nil
	})
	// Restore exactly what leaseplane_endpoint.go's init installed for the rest of the package.
	t.Cleanup(func() {
		gateway.SetLeasePlaneProviders(serveLeasePlaneLeases, serveLeasePlanePresence)
		gateway.SetWorkspaceLeaseAdmissionProvider(serveWorkspaceLeaseAdmission)
	})
}

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
				if ok, why := floorAdmits(c.Tool, v); !ok {
					t.Errorf("call %s (%s %s): shipped floor did not admit it (%s), want ALLOW", c.ID, c.Tool, c.ArgPreview(), why)
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

// floorNativeReroutes is the sanctioned in-syscall re-route table the shipped floor applies to
// native read-family tools: since #11150 (commit 306d37f125) a Read is transparently rewritten
// to fak_read, and since #11499 (commit 7fa61ef91c) grep/glob-family calls to fak_grep/fak_glob
// (internal/adjudicator/decide.go). Keys are lower-cased tool names.
var floorNativeReroutes = map[string]string{
	"read": "fak_read",
	"grep": "fak_grep", "rg": "fak_grep", "ripgrep": "fak_grep", "search": "fak_grep",
	"glob": "fak_glob", "find": "fak_glob",
}

// floorAdmits reports whether a floor verdict ADMITS tool, in the same sense the gateway
// counts a call as admitted (internal/gateway/adjudicate_proposed.go treats ALLOW and TRANSFORM
// alike): a bare ALLOW, or a TRANSFORM that re-routes a native read-family tool to its
// sanctioned fak_* twin. A TRANSFORM to any other tool, or one with no substitute, does NOT
// count, so a rewrite can never pass as "allowed" by accident. The second result says why not.
func floorAdmits(tool string, v abi.Verdict) (bool, string) {
	switch v.Kind {
	case abi.VerdictAllow:
		return true, ""
	case abi.VerdictTransform:
		want, ok := floorNativeReroutes[strings.ToLower(tool)]
		if !ok {
			return false, fmt.Sprintf("TRANSFORM of %q, which has no sanctioned fak_* re-route", tool)
		}
		tp, ok := v.Payload.(abi.TransformPayload)
		if !ok {
			return false, fmt.Sprintf("TRANSFORM with payload %T, want abi.TransformPayload", v.Payload)
		}
		if tp.NewTool != want {
			return false, fmt.Sprintf("TRANSFORM to %q, want the sanctioned re-route %q", tp.NewTool, want)
		}
		return true, ""
	default:
		return false, fmt.Sprintf("verdict %v (%s)", v.Kind, abi.ReasonName(v.Reason))
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
	// ...and start from one: journal.Enable is idempotent, so a journal some earlier test left
	// active (an in-process guard run with --audit <its TempDir>) would be reused, the report
	// would name that foreign path instead of the fak-guard-replay- default asserted below, and
	// verification would fail once that test's TempDir is gone.
	journal.ResetActiveForTest()
	installEmptyReplayLeaseAuthority(t)
	for _, wire := range []string{"anthropic", "openai"} {
		t.Run(wire, func(t *testing.T) {
			t.Cleanup(journal.ResetActiveForTest)
			var sb strings.Builder
			code := runGuardReplay(guardTraceFixturePath, wire, "", "", false, &sb)
			out := sb.String()
			if code != 0 {
				t.Fatalf("runGuardReplay(%s) exit = %d, want 0\n%s", wire, code, out)
			}
			for _, want := range []string{
				"fak guard --replay-trace",
				"DENY[POLICY_BLOCK]",
				"kernel decision(s)",
				"journal chain verified",
				"fak-guard-replay-",
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
	snapPath := filepath.Join(dir, "vcache-turns.jsonl")
	t.Setenv(vcachesnapshot.EnvPath, snapPath)

	var sb strings.Builder
	if code := runGuardReplay(guardContextTraceFixturePath, "openai", "", "", false, &sb); code != 0 {
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

func TestGuardReplayHonorsExplicitAuditPath(t *testing.T) {
	dir := t.TempDir()
	t.Cleanup(journal.ResetActiveForTest)
	installEmptyReplayLeaseAuthority(t)
	auditPath := filepath.Join(dir, "replay-audit.jsonl")

	var sb strings.Builder
	code := runGuardReplay(guardTraceFixturePath, "openai", "", auditPath, false, &sb)
	out := sb.String()
	if code != 0 {
		t.Fatalf("runGuardReplay exit = %d, want 0\n%s", code, out)
	}
	if _, err := os.Stat(auditPath); err != nil {
		t.Fatalf("explicit audit path was not created: %v\n%s", err, out)
	}
	if rows, err := journal.Verify(auditPath); err != nil || rows == 0 {
		t.Fatalf("explicit audit path did not verify: rows=%d err=%v\n%s", rows, err, out)
	}
	if !strings.Contains(out, auditPath) {
		t.Fatalf("replay output did not name explicit audit path %q:\n%s", auditPath, out)
	}
}

// TestGuardReplayUnknownWireRejected proves a bad --replay-wire fails loud (exit 2) rather
// than silently defaulting.
func TestGuardReplayUnknownWireRejected(t *testing.T) {
	var sb strings.Builder
	if code := runGuardReplay(guardTraceFixturePath, "gemini", "", "", false, &sb); code != 2 {
		t.Fatalf("unknown wire exit = %d, want 2\n%s", code, sb.String())
	}
}
