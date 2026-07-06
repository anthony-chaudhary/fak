package adjudicator_test

// Flag-bypass / headless-allow capability-floor conformance (issue #2921, child of
// the "mine Hermes for what a kernel does better" epic #2908 track C).
//
// Hermes evidence (tools/approval.py, a 2,921-line ~90-regex approval layer): a
// `force=True` on the terminal tool SKIPS ALL guards, and a non-interactive /
// non-gateway session AUTO-APPROVES with a warning ("trust-by-config"). So the
// effective posture there is default-ALLOW the moment you are headless, and any
// caller that can set the flag has a full bypass. A floor with a `force` escape
// hatch and a headless default-allow is not a floor.
//
// fak's capability floor has NEITHER. Adjudicator.Adjudicate(ctx, *abi.ToolCall)
// is a PURE function of (installed Policy, ToolCall): there is no `force`
// parameter, no interactive/headless/TTY/session-mode input, and no env-based
// auto-approve. abi.ToolCall.Meta is an OPEN map whose "unknown keys MUST be
// ignored" (types.go) and which decide.go documents as model-controlled and
// unable to widen authority. The ONLY way to admit a denied capability is to
// change the reviewed, in-git Policy — never a runtime flag, env var, or headless
// mode. These tests are the executable regression witnesses for that structural
// guarantee: the same structural deny applies interactive OR headless, flag-set
// OR unset.
//
// Sibling conformance witnesses for other bypass axes:
//   - internal/policy/isolation_capfloor_conformance_test.go — a stronger
//     process-isolation tier never bypasses the #2018 adjudication floor.
//   - internal/adjudicator/dogfood_manifest_test.go — the shipped floor's
//     verdict matrix; a manifest edit that silently widens it fails there.

import (
	"context"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// bypassFloor is a real, small deployable floor exercising every hard deny rung
// plus an affirmative allow: an explicit name-deny, a self-modify glob, an
// arg-VALUE deny (the exact class Hermes' `force` skips), the hardwired egress
// floor, fail-closed default-deny, and one allowed read. If ANY flag/env/headless
// signal could escalate a capability, one of these deny cells would flip.
func bypassFloor() adjudicator.Policy {
	return adjudicator.Policy{
		Posture:         adjudicator.PostureFailClosed,
		Allow:           map[string]bool{"Read": true},
		AllowPrefix:     []string{"read_"},
		Deny:            map[string]abi.ReasonCode{"git_push": abi.ReasonPolicyBlock},
		SelfModifyGlobs: []string{"internal/kernel/", ".git/"},
		ArgPredicates: []adjudicator.ArgPredicate{{
			Tool:   "Bash",
			Arg:    "command",
			Kind:   adjudicator.ArgDenyRegex,
			Re:     regexp.MustCompile(`\brm\s+-[A-Za-z]*[rRfF]`), // mirrors the shipped dogfood rm rule
			Reason: abi.ReasonPolicyBlock,
		}},
	}
}

// bypassCase is one call the floor must decide IDENTICALLY in every matrix cell.
type bypassCase struct {
	name   string
	tool   string
	args   string
	kind   abi.VerdictKind
	reason abi.ReasonCode
}

var bypassCases = []bypassCase{
	// The one affirmative allow: legitimate work stays allowed identically.
	{"allowed read", "Read", `{"file_path":"README.md"}`, abi.VerdictAllow, abi.ReasonNone},
	// Every hard deny rung — the capabilities a `force`/headless bypass would escalate.
	{"name deny", "git_push", `{}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
	{"self-modify", "Edit", `{"file_path":"internal/kernel/x.go"}`, abi.VerdictDeny, abi.ReasonSelfModify},
	{"arg-value deny", "Bash", `{"command":"rm -rf /tmp/x"}`, abi.VerdictDeny, abi.ReasonPolicyBlock},
	{"egress floor", "WebFetch", `{"url":"http://169.254.169.254/latest/meta-data/"}`, abi.VerdictDeny, egressfloor.ReasonEgressBlock},
	{"default deny", "weirdTool", `{}`, abi.VerdictDeny, abi.ReasonDefaultDeny},
}

// flagVariants are the "force flag" axis: nil (unset) plus hostile Meta payloads a
// caller might set to try to escalate. Meta is the ONLY escalation-shaped surface
// on the envelope (an open string map), so if a flag could bypass the floor it
// would ride one of these keys. It cannot: Meta is ignored by adjudication.
var flagVariants = []struct {
	name string
	meta map[string]string
}{
	{"flag-unset", nil},
	{"force", map[string]string{"force": "true"}},
	{"force+approve", map[string]string{"force": "true", "approve": "true", "auto_approve": "true"}},
	{"skip-permissions", map[string]string{"dangerously_skip_permissions": "true", "skip_permissions": "true", "yolo": "true"}},
	{"trust-by-config", map[string]string{"trust": "config", "trust_by_config": "true", "interactive": "false"}},
	{"bypass-guards", map[string]string{"bypass": "true", "bypass_guards": "true", "skip_guards": "true", "headless": "true", "force": "true"}},
}

// TestFloorHoldsAcrossFlagAndHeadlessMatrix is the #2921 acceptance matrix:
// (interactive | headless) x (flag-unset | flag-set) x every deny/allow rung. The
// floor's verdict is byte-identical in every cell.
//
//   - The FLAG axis is real: each cell re-adjudicates with a hostile force/approve/
//     bypass Meta payload merged onto the call. None flips a Deny to an Allow.
//   - The HEADLESS axis is a STRUCTURAL no-op by construction — Adjudicate takes no
//     session-mode argument — and this test proves the floor also ignores every
//     ambient "I am headless / non-interactive / CI" ENV signal a naive
//     trust-by-config auto-approver would key on. Set them all; the verdict does
//     not move.
func TestFloorHoldsAcrossFlagAndHeadlessMatrix(t *testing.T) {
	a := adjudicator.New(bypassFloor())

	sessions := []struct {
		name string
		env  map[string]string // headless/non-interactive ambient signals
	}{
		{"interactive", map[string]string{"CI": "", "FAK_HEADLESS": "", "FAK_NONINTERACTIVE": "", "TERM": "xterm"}},
		{"headless", map[string]string{"CI": "true", "FAK_HEADLESS": "1", "FAK_NONINTERACTIVE": "1", "TERM": ""}},
	}

	for _, sess := range sessions {
		t.Run(sess.name, func(t *testing.T) {
			// Every headless/non-interactive signal a bypass might read is set here;
			// the floor consults none of them (t.Setenv restores after the subtest).
			for k, v := range sess.env {
				t.Setenv(k, v)
			}
			for _, fv := range flagVariants {
				for _, c := range bypassCases {
					call := &abi.ToolCall{
						Tool: c.tool,
						Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(c.args)},
						Meta: mergedMeta(fv.meta),
					}
					v := a.Adjudicate(context.Background(), call)
					if v.Kind != c.kind || v.Reason != c.reason {
						t.Errorf("session=%s flags=%s case=%q: got Kind=%v Reason=%s, want Kind=%v Reason=%s — a flag/env/headless signal must NEVER move the floor",
							sess.name, fv.name, c.name, v.Kind, abi.ReasonName(v.Reason), c.kind, abi.ReasonName(c.reason))
					}
				}
			}
		})
	}
}

// mergedMeta layers a flag payload over the baseline per-call Meta so a hostile key
// cannot collide the readOnlyHint the harness sends. A nil flag payload yields the
// baseline Meta (the flag-unset cell).
func mergedMeta(flag map[string]string) map[string]string {
	m := map[string]string{"readOnlyHint": "true"}
	for k, v := range flag {
		m[k] = v
	}
	return m
}

// TestToolCallEnvelopeHasNoBypassSeam is the structural tripwire behind the matrix:
// the frozen request envelope abi.ToolCall exposes NO typed field a force/headless
// bypass could ride. Meta (the open string map) is the only escalation-shaped
// surface, and the matrix above proves Meta cannot widen authority. If a future
// edit adds a `Force bool` / `Interactive bool` / `SkipPermissions bool` typed seam
// to the envelope, this fails loud so the change must reckon with the floor rather
// than quietly open a bypass-by-flag class.
func TestToolCallEnvelopeHasNoBypassSeam(t *testing.T) {
	banned := []string{
		"force", "approve", "headless", "interactive", "bypass",
		"skip", "danger", "trust", "yolo", "override", "escalat", "allowall",
	}
	rt := reflect.TypeOf(abi.ToolCall{})
	for i := 0; i < rt.NumField(); i++ {
		lname := strings.ToLower(rt.Field(i).Name)
		for _, b := range banned {
			if strings.Contains(lname, b) {
				t.Fatalf("abi.ToolCall has field %q matching bypass-seam token %q: a typed force/headless seam would let a flag widen authority past the capability floor. Keep escalation impossible by construction — carry advisory hints in Meta (which adjudication ignores), and gate any real capability change through the reviewed Policy, not the call envelope.",
					rt.Field(i).Name, b)
			}
		}
	}
}
