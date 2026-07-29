package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// commit_msg_advisory_test.go — the durable half of the COMMIT_MSG subject advisory.
//
// The advisory exists to move the commit_debt card by making a gradeable subject the default
// going forward. The DANGER it carries is the opposite of its purpose: this repo is a shared
// working tree with ~27 concurrent committing sessions, so an advisory that ever became a
// refusal would wedge every lane at once. These tests pin NON-BLOCKING as a property, not as a
// promise in a comment:
//
//   - the signature carries no value to branch on (compile-time pin, below);
//   - every ungradeable subject still reaches commitFn and still exits 0;
//   - even FLEET_MSG_GUARD=block — the escalation the HOOK layer honours — only warns here;
//   - the advisory never touches stdout, so `--json` stays machine-parseable;
//   - the advisory never edits the message that lands.
//
// If a later change turns this into a gate, at least one of these fails.

// The compile-time pin. renderCommitMsgAdvisory must stay a pure side-effect procedure: giving
// it an int/error/bool return — the first move anyone makes when converting an advisory into a
// refusal — stops this assignment compiling, and a package that does not compile is a red test.
var _ func(io.Writer, string, []string, string) = renderCommitMsgAdvisory

// advisoryEnvDefault pins the guard env to its out-of-the-box state so an operator's exported
// FLEET_MSG_GUARD cannot make these tests pass or fail for the wrong reason.
func advisoryEnvDefault(t *testing.T) {
	t.Helper()
	t.Setenv("FLEET_MSG_GUARD", "")
	t.Setenv("FLEET_ALLOW_MSG", "")
}

// ungradeableSubjects are subjects that survive deriveCommitMessageStamp's auto-heal — the
// residue the advisory exists for. Each leads with a noun or an unrecognized verb, or carries a
// type outside the conventional set, and none has a deterministic mechanical rewrite.
var ungradeableSubjects = []string{
	"gateway reclaim improvements",                           // no `type(scope):` shape at all
	"feat(gateway): reclaim path improvements (fak gateway)", // unrecognized leading verb
	"chore: cleanup of the reclaim path",                     // noun-led description
	"wip(gateway): tinkering with the reclaim path",          // type outside the conventional set
}

// TestCommitMsgAdvisory_warnsAndStillCommits is the headline witness: for every ungradeable
// subject the advisory PRINTS, the commit still runs (commitFn is reached), and the exit code
// is 0. The commitFn assertion is the load-bearing one — an exit-code check alone would still
// pass if a future gate short-circuited the commit and returned 0.
func TestCommitMsgAdvisory_warnsAndStillCommits(t *testing.T) {
	for _, subject := range ungradeableSubjects {
		t.Run(subject, func(t *testing.T) {
			advisoryEnvDefault(t)
			called := false
			withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
				called = true
				return safecommit.Result{Committed: true, Verified: true, SHA: "abc123def456", Paths: o.Paths}, nil
			})
			var out, errb bytes.Buffer
			code := runCommit(&out, &errb, []string{
				"--dir", t.TempDir(), "--no-build-check",
				"--path", "internal/gateway/server.go",
				"-m", subject,
			})
			if code != 0 {
				t.Fatalf("ADVISORY MUST NOT BLOCK: exit %d for %q (stdout=%q stderr=%q)", code, subject, out.String(), errb.String())
			}
			if !called {
				t.Fatalf("ADVISORY MUST NOT BLOCK: commitFn was never reached for %q (stderr=%q)", subject, errb.String())
			}
			if !strings.Contains(errb.String(), commitMsgAdvisoryHeadline) {
				t.Errorf("want the advisory headline %q on stderr for %q; got %q", commitMsgAdvisoryHeadline, subject, errb.String())
			}
			if !strings.Contains(errb.String(), "type(scope): <verb> <what>") {
				t.Errorf("advisory must name the fixed shape; got %q", errb.String())
			}
			// The operator-facing bytes are the product here, so a -v run shows them verbatim:
			// the wording is what a committing session actually has to act on, and reviewing it
			// from a passing run is cheaper than reconstructing it from a failure message.
			t.Logf("exit=%d commitFn reached=%v; stderr:\n%s", code, called, errb.String())
		})
	}
}

// TestCommitMsgAdvisory_blockModeStillOnlyWarns is the fleet-safety pin. The hook layer lets
// FLEET_MSG_GUARD=block escalate COMMIT_MSG into a refusal; this advisory deliberately has NO
// block mode, because a refusal on the mutating commit path would wedge every concurrent
// session at once. Setting block here must still warn, still commit, still exit 0.
func TestCommitMsgAdvisory_blockModeStillOnlyWarns(t *testing.T) {
	advisoryEnvDefault(t)
	t.Setenv("FLEET_MSG_GUARD", "block")
	called := false
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		called = true
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123def456", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--dir", t.TempDir(), "--no-build-check",
		"--path", "internal/gateway/server.go",
		"-m", "chore: cleanup of the reclaim path",
	})
	if code != 0 || !called {
		t.Fatalf("FLEET_MSG_GUARD=block must NOT block the mutating commit: exit=%d called=%v stderr=%q", code, called, errb.String())
	}
	if !strings.Contains(errb.String(), commitMsgAdvisoryHeadline) {
		t.Errorf("block mode should still warn; stderr=%q", errb.String())
	}
	if out.Len() != 0 && strings.Contains(out.String(), commitMsgAdvisoryHeadline) {
		t.Errorf("advisory must not reach stdout; stdout=%q", out.String())
	}
}

// TestCommitMsgAdvisory_silentOnGradeableSubject is the negative fixture: a subject that IS
// witness-gradeable and already stamped earns no advisory noise at all.
func TestCommitMsgAdvisory_silentOnGradeableSubject(t *testing.T) {
	advisoryEnvDefault(t)
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123def456", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--dir", t.TempDir(), "--no-build-check",
		"--path", "internal/gateway/server.go",
		"-m", "feat(gateway): add the reclaim path (fak gateway)",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%q)", code, errb.String())
	}
	if strings.Contains(errb.String(), commitMsgAdvisoryHeadline) {
		t.Errorf("a gradeable subject must earn no advisory; stderr=%q", errb.String())
	}
	t.Logf("exit=%d; stderr (expected advisory-free):\n%s", code, errb.String())
}

// TestCommitMsgAdvisory_silentAfterAutoHeal pins the ORDERING that makes this advisory quiet
// enough to live on the mandated commit path: it runs AFTER deriveCommitMessageStamp, so the
// two deterministic defects that stamp derivation already repairs — a near-miss type
// (`feature:` -> `feat:`) and an inflected leading verb (`added` -> `add`) — never reach the
// operator as a warning. Only the residue nobody can mechanically fix is worth their attention;
// an advisory that also nagged about defects fak had silently fixed would train sessions to
// ignore it, which is the failure mode that makes an advisory worthless.
func TestCommitMsgAdvisory_silentAfterAutoHeal(t *testing.T) {
	for _, subject := range []string{
		"feature(gateway): add the reclaim path (fak gateway)", // near-miss type, healed to feat:
		"feat(gateway): added the reclaim path (fak gateway)",  // inflected verb, healed to add
		"feat(gateway): wiring the reclaim path (fak gateway)", // gerund, healed to wire
	} {
		t.Run(subject, func(t *testing.T) {
			advisoryEnvDefault(t)
			var got safecommit.Options
			withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
				got = o
				return safecommit.Result{Committed: true, Verified: true, SHA: "abc123def456", Paths: o.Paths}, nil
			})
			var out, errb bytes.Buffer
			code := runCommit(&out, &errb, []string{
				"--dir", t.TempDir(), "--no-build-check",
				"--path", "internal/gateway/server.go",
				"-m", subject,
			})
			if code != 0 {
				t.Fatalf("want exit 0, got %d (stderr=%q)", code, errb.String())
			}
			if strings.Contains(errb.String(), commitMsgAdvisoryHeadline) {
				t.Errorf("auto-heal already fixed %q — the advisory must stay silent; stderr=%q", subject, errb.String())
			}
			// Guard against the vacuous pass: silence must come from the heal having WORKED,
			// not from the advisory failing to run. The landed subject must differ from the
			// input and must itself be gradeable.
			landed := firstCommitLine(got.Message)
			if landed == subject {
				t.Fatalf("expected stamp derivation to heal %q, but it landed unchanged", subject)
			}
			if ok, why := hooks.CommitMsgVerdict(landed); !ok {
				t.Fatalf("healed subject %q is still ungradeable (%s) — silence would be a miss, not a fix", landed, why)
			}
			t.Logf("healed %q -> %q; advisory silent", subject, landed)
		})
	}
}

// TestCommitMsgAdvisory_silencedByGuardOff confirms the only knob the advisory honours can
// silence it and nothing more (FLEET_MSG_GUARD=off, FLEET_ALLOW_MSG=1). Both still exit 0.
func TestCommitMsgAdvisory_silencedByGuardOff(t *testing.T) {
	for _, env := range []struct{ key, value string }{
		{"FLEET_MSG_GUARD", "off"},
		{"FLEET_ALLOW_MSG", "1"},
	} {
		t.Run(env.key+"="+env.value, func(t *testing.T) {
			advisoryEnvDefault(t)
			t.Setenv(env.key, env.value)
			withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
				return safecommit.Result{Committed: true, Verified: true, SHA: "abc123def456", Paths: o.Paths}, nil
			})
			var out, errb bytes.Buffer
			code := runCommit(&out, &errb, []string{
				"--dir", t.TempDir(), "--no-build-check",
				"--path", "internal/gateway/server.go",
				"-m", "chore: cleanup of the reclaim path",
			})
			if code != 0 {
				t.Fatalf("want exit 0, got %d (stderr=%q)", code, errb.String())
			}
			if strings.Contains(errb.String(), commitMsgAdvisoryHeadline) {
				t.Errorf("%s=%s must silence the advisory; stderr=%q", env.key, env.value, errb.String())
			}
		})
	}
}

// TestCommitMsgAdvisory_doesNotAlterTheCommittedMessage pins that the advisory OBSERVES the
// message and never edits it: the subject handed to commitFn is byte-for-byte the one the
// existing stamp derivation produced.
func TestCommitMsgAdvisory_doesNotAlterTheCommittedMessage(t *testing.T) {
	advisoryEnvDefault(t)
	const subject = "feat(gateway): reclaim path improvements (fak gateway)"
	var got safecommit.Options
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		got = o
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123def456", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	if code := runCommit(&out, &errb, []string{
		"--dir", t.TempDir(), "--no-build-check",
		"--path", "internal/gateway/server.go",
		"-m", subject + "\n\nBody stays here.",
	}); code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%q)", code, errb.String())
	}
	if want := subject + "\n\nBody stays here."; got.Message != want {
		t.Fatalf("advisory must not rewrite the message\n got %q\nwant %q", got.Message, want)
	}
}

// TestCommitMsgAdvisory_keepsJSONStdoutParseable pins the stream discipline: the advisory goes
// to stderr, so a `--json` consumer still gets a clean object on stdout even for the worst
// possible subject.
func TestCommitMsgAdvisory_keepsJSONStdoutParseable(t *testing.T) {
	advisoryEnvDefault(t)
	withCommitFn(t, func(_ context.Context, o safecommit.Options) (safecommit.Result, error) {
		return safecommit.Result{Committed: true, Verified: true, SHA: "abc123def456", Paths: o.Paths}, nil
	})
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--json", "--dir", t.TempDir(), "--no-build-check",
		"--path", "internal/gateway/server.go",
		"-m", "gateway reclaim improvements",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (stderr=%q)", code, errb.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("advisory leaked into --json stdout: %v\nstdout=%q", err, out.String())
	}
	if !strings.Contains(errb.String(), commitMsgAdvisoryHeadline) {
		t.Errorf("the advisory should still have been emitted on stderr; got %q", errb.String())
	}
}

// TestRenderCommitMsgAdvisory_exemptSubjectsAreQuiet confirms the advisory reuses the lint's
// exempt classes rather than the bare verb gate: a release anchor, a merge and a revert are
// intentionally not `type(scope): <verb>` and must not be nagged.
func TestRenderCommitMsgAdvisory_exemptSubjectsAreQuiet(t *testing.T) {
	advisoryEnvDefault(t)
	for _, subject := range []string{
		"v1.4.0: cut the release",
		"Merge branch 'main' into main",
		"Revert \"feat(gateway): add the reclaim path\"",
	} {
		var b bytes.Buffer
		renderCommitMsgAdvisory(&b, subject, []string{"internal/gateway/server.go"}, t.TempDir())
		if b.Len() != 0 {
			t.Errorf("exempt subject %q must earn no advisory; got %q", subject, b.String())
		}
	}
}
