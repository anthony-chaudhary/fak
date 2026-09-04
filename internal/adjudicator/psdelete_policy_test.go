package adjudicator

import (
	"context"
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestPosixDeleteRuleIsStructuralOnEveryShippedSurface pins the surface half of the
// same defect. The POSIX delete regex ships byte-identically on Bash, shell_command
// and functions.shell_command, but only the Bash copy used to be decided
// structurally — so on the other two the rule kept the raw-regex path and lost BOTH
// carve-outs: the scratchpad containment route and #4983's force-only
// single-literal-target route. The same command got two different verdicts based on
// nothing but which tool NAME the harness used.
func TestPosixDeleteRuleIsStructuralOnEveryShippedSurface(t *testing.T) {
	remove := "r" + "m"
	for _, tool := range []string{"Bash", "shell_command", "functions.shell_command"} {
		t.Run(tool, func(t *testing.T) {
			t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", "/tmp/claude")
			a := New(Policy{
				Allow: map[string]bool{tool: true},
				ArgPredicates: []ArgPredicate{{
					Tool: tool, Arg: "command", Kind: ArgDenyRegex,
					Re: regexp.MustCompile(defaultRmRfDenyRegex), Reason: abi.ReasonPolicyBlock,
				}},
			})
			routine := []string{
				remove + " -rf /tmp/claude/session-123/clone", // confined to a declared scratch root
				remove + " -f notes.txt",                      // #4983: force-only, one literal path
			}
			for _, cmd := range routine {
				v := a.Adjudicate(context.Background(), inlineCall(tool, jsonCmd(cmd)))
				if v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonPolicyBlock {
					t.Errorf("%q stayed a terminal POLICY_BLOCK on %s but is admitted on Bash — the floor's strictness must not depend on the harness's tool name", cmd, tool)
				}
			}
			fatal := []string{
				remove + " -rf /work/fak/internal",        // recursive delete of real work
				remove + " -f *",                          // force-only but unbounded
				remove + " -rf /tmp/claude-evil/x",        // sibling-prefix escape
				remove + " -rf /tmp/claude/a /work/fak/x", // one contained target does not license the other
			}
			for _, cmd := range fatal {
				v := a.Adjudicate(context.Background(), inlineCall(tool, jsonCmd(cmd)))
				if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
					t.Errorf("%q on %s = %v/%s, want Deny/POLICY_BLOCK", cmd, tool, v.Kind, abi.ReasonName(v.Reason))
				}
			}
		})
	}
	// A differently-spelled policy still keeps raw-regex semantics on those surfaces.
	pr := &ArgPredicate{Tool: "shell_command", Arg: "command", Re: regexp.MustCompile(`custom-delete-pattern`)}
	if isRmRfArgRule(pr) {
		t.Fatal("a custom shell_command delete regex must keep the raw-regex path")
	}
}

// TestDeleteRefusalNamesTheAdmittedRoute binds the shipped refusal TEXT to what the
// shipped rule actually DOES.
//
// Both delete rules already carry a scratchpad carve-out — a recursive delete whose
// targets all sit strictly below a declared harness scratchpad root is admitted — but
// neither refusal said so. An agent that hit the block read "recursive/forced deletes
// are operator-only", concluded the route was closed, and either stopped or escalated
// to a human for work the floor had already decided was safe. A capability floor that
// hides its own sanctioned route is indistinguishable, from the agent's side, from one
// that has none.
//
// This pins both halves at once, which is the invariant the self-refuting-remedy class
// keeps violating (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md): the fix text must
// name the admitted route, AND that route must actually be admitted by the very regex
// the fix text is attached to.
func TestDeleteRefusalNamesTheAdmittedRoute(t *testing.T) {
	b, err := os.ReadFile("../../cmd/fak/guard-default-policy.json")
	if err != nil {
		t.Fatalf("read shipped policy: %v", err)
	}
	var manifest struct {
		ArgRules []struct {
			Tool      string `json:"tool"`
			Arg       string `json:"arg"`
			DenyRegex string `json:"deny_regex"`
			Fix       string `json:"fix"`
		} `json:"arg_rules"`
	}
	if err := json.Unmarshal(b, &manifest); err != nil {
		t.Fatalf("parse shipped policy: %v", err)
	}

	remove := "r" + "m"
	families := map[string]struct {
		regex    string
		root     string
		admitted string
		escape   string
		surfaces []string
	}{
		"posix": {
			regex:    defaultRmRfDenyRegex,
			root:     "/tmp/claude",
			admitted: remove + " -rf /tmp/claude/session-123/clone",
			escape:   remove + " -rf /tmp/claude-evil/clone",
			surfaces: []string{"bash", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command"},
		},
		"powershell": {
			regex:    defaultPSDeleteDenyRegex,
			root:     winScratchRoot(),
			admitted: psDeleteCommand(` ` + winScratchRoot() + `\session-123\clone -Recurse -Force`),
			escape:   psDeleteCommand(` ` + winScratchRoot() + `-evil\clone -Recurse -Force`),
			surfaces: []string{"powershell", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command"},
		},
	}

	seen := map[string]map[string]bool{}
	fixes := map[string]map[string][]string{}
	for name := range families {
		seen[name] = map[string]bool{}
		fixes[name] = map[string][]string{}
	}

	for _, r := range manifest.ArgRules {
		for name, fam := range families {
			if r.DenyRegex != fam.regex {
				continue
			}
			pr := &ArgPredicate{Tool: r.Tool, Arg: r.Arg, Kind: ArgDenyRegex, Re: regexp.MustCompile(r.DenyRegex)}
			if !isRmRfArgRule(pr) {
				t.Errorf("%s family: tool %q ships the delete deny_regex but the structural recogniser rejects it — the rule would fall back to the raw-regex path and lose the carve-out entirely",
					name, r.Tool)
				continue
			}
			seen[name][strings.ToLower(r.Tool)] = true
			fixes[name][r.Fix] = append(fixes[name][r.Fix], r.Tool)

			if !strings.Contains(r.Fix, "FAK_GUARD_SCRATCHPAD_ROOTS") {
				t.Errorf("%s family: tool %q refusal never names the scratchpad route it already admits: %q", name, r.Tool, r.Fix)
			}

			t.Run(name+"/"+r.Tool, func(t *testing.T) {
				t.Setenv("FAK_GUARD_SCRATCHPAD_ROOTS", fam.root)
				a := New(Policy{
					Allow: map[string]bool{r.Tool: true},
					ArgPredicates: []ArgPredicate{{
						Tool: r.Tool, Arg: r.Arg, Kind: ArgDenyRegex,
						Re: regexp.MustCompile(r.DenyRegex), Reason: abi.ReasonPolicyBlock,
					}},
				})
				v := a.Adjudicate(context.Background(), inlineCall(r.Tool, shellCommandArgs(r.Arg, fam.admitted)))
				if v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonPolicyBlock {
					t.Errorf("the refusal advertises the scratchpad route but %q is still a terminal POLICY_BLOCK", fam.admitted)
				}
				esc := a.Adjudicate(context.Background(), inlineCall(r.Tool, shellCommandArgs(r.Arg, fam.escape)))
				if esc.Kind != abi.VerdictDeny || esc.Reason != abi.ReasonPolicyBlock {
					t.Errorf("sibling-prefix escape %q = %v/%s, want Deny/POLICY_BLOCK", fam.escape, esc.Kind, abi.ReasonName(esc.Reason))
				}
			})
		}
	}

	for name, fam := range families {
		for _, tool := range fam.surfaces {
			if !seen[name][tool] {
				t.Errorf("%s family: shipped policy has no delete rule for tool %q — the recogniser's surface list is stale", name, tool)
			}
		}
		// The rule is copied across surfaces by hand, and the LAST array element has
		// no trailing comma — so a search-and-replace over the others silently leaves
		// it behind and one surface then advertises a different boundary than the
		// rest. An agent refused on shell_command must read the same guidance it
		// would get on Bash.
		if len(fixes[name]) > 1 {
			for fix, tools := range fixes[name] {
				t.Errorf("%s family: fix text differs across surfaces %v: %q", name, tools, fix)
			}
		}
	}
}
