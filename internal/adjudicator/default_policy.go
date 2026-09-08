package adjudicator

import "github.com/anthony-chaudhary/fak/internal/abi"

// interpreterEvalSpec pairs a general-purpose interpreter with the inline-program flags
// whose presence as a TOKEN means it runs code from an opaque string argument able to
// write a file directly.
type InlineEvalSpec struct {
	Interp string
	Flags  []string
}

type interpreterEvalSpec struct {
	interp string   // the interpreter named as a command word (`ruby`, `node`, …)
	flags  []string // its inline-eval flags (`-e`, `--eval`, `-c`, `-p`/`--print`)
}

// interpreterEvalFlags is the adjudicator's inline-eval write floor (#172 Hole 1
// residual): the general-purpose interpreters most likely on a coding agent's PATH —
// python, node, and ruby — each paired with the inline-program flags that run an opaque
// program string. It is the interpreter analogue of shellWriteVerbs; commandWrites ranges
// over it and treats such a command as write-shaped, routing it through the SAME
// commandSelfModify guard the shell floor uses.
//
// ruby joins python/node to close an asymmetry the rulesynth RSI loop (internal/rsiloop,
// run with -harness rulesynth) mined from the near-miss corpus and the keep-bit KEPT:
// `ruby -i` (an in-place edit) is already caught by shellWriteVerbs, but `ruby -e
// 'File.write("internal/adjudicator/decide.go", …)'` — the EVAL flag doing the same
// self-edit — slipped every rung. `perl -e`, `php -r`, and `lua -e` are the same shape and
// remain the residual queue the loop's corpus drives next, so they are deliberately not
// listed here yet.
//
// Detection is by TOKEN, not a fixed `<interp> <flag> ` prefix (see interpreterEvalMatch):
// the interpreter need only appear as a word and the flag as its own argument, so the
// idiomatic no-space, quoted, and `=`-joined spellings (`ruby -e'…'`, `node --eval=…`) and
// intervening flags (`ruby -rjson -e …`) are all caught — closing the porous-prefix gap a
// fixed-spelling table leaves open. The identifier name `interpreterEvalFlags` is pinned by
// architest (TestInlineEvalFloorWiredInCommandWrites); rename only with that gate's constant.
var interpreterEvalFlags = []interpreterEvalSpec{
	{"python3", []string{"-c"}},
	{"python", []string{"-c"}},
	{"node", []string{"-e", "--eval", "-p", "--print"}},
	{"ruby", []string{"-e", "--eval"}},
}

// DefaultPolicy is the v0.1 baseline: allow the read-only tool family + the
// frozen tau2 trace tools, deny a self-modify glob set, redact common secret arg
// fields. Tuned to be permissive enough to drive the bench yet fail-closed on
// unknown + self-modifying calls.
func DefaultPolicy() Policy {
	return Policy{
		Allow: map[string]bool{
			"search_flights": true, "get_reservation_details": true,
			"get_user_details": true, "list_all_airports": true,
			"calculate": true, "search_direct_flight": true,
			"transfer_to_human_agents": true, "send_certificate": true,
			"book_reservation": true, "update_reservation_flights": true,
			"fak_grep": true, "fak_glob": true,
		},
		AllowPrefix: []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"},
		Deny: map[string]abi.ReasonCode{
			"shell_rm_rf": abi.ReasonPolicyBlock,
			"exfiltrate":  abi.ReasonSecretExfil,
		},
		// Witness machinery (#172 Hole 2): internal/architest + internal/shipgate
		// are covered by the fak/internal/ catch-all, but dos.toml (the lane
		// taxonomy + stamp grammar the referee binds to) is not — name it so a
		// self-improving loop cannot quietly rewrite the gate that witnesses it.
		SelfModifyGlobs: []string{"internal/abi/", "internal/kernel/", "internal/adjudicator/", "internal/architest/", "internal/shipgate/", "dos.toml", ".dos/", "fak/internal/"},
		RedactFields:    []string{"password", "secret", "api_key", "token", "authorization"},
	}
}

// DevAgentPolicy is the deployable floor for a CODING agent that runs against this
// repo (the dogfood case — fak passing its own CICD test). It differs from the
// permissive bench DefaultPolicy in three load-bearing ways:
//
//   - it DENIES the shared-history git mutations (push / merge / tag): a dev agent
//     adapts code, it never moves the branch on its own say-so;
//   - it bounds writes AWAY from the spine — a write whose target touches the ABI,
//     the kernel, the adjudicator, the policy loader, the defconfig, or the .git
//     dir is a SELF_MODIFY (ESCALATE), not a silent edit;
//   - it ALLOWS a single high-level `ship_release` action that is NOT free: the
//     shipgate adjudicator holds it behind the require-witness rung, so it proceeds
//     only when a claimed ship is corroborated by git evidence the agent did not
//     author (an unwitnessed ship is refused).
//
// Mirrored on disk as examples/dev-agent-policy.json (round-trips through the
// policy manifest loader), so an adopter selects it with `--policy` — no fork.
func DevAgentPolicy() Policy {
	return Policy{
		Posture: PostureDefaultOpen,
		Allow: map[string]bool{
			// safe inspect / build / test tools a coding agent drives
			"Read":       true,
			"fak_read":   true,
			"git_status": true, "git_diff": true, "git_log": true,
			"go_build": true, "go_test": true, "run_tests": true,
			// the high-level ship action: allowed at the floor, but witness-gated by
			// the shipgate adjudicator (a claimed ship must be git-corroborated).
			"ship_release": true,
		},
		AllowPrefix: []string{"read_", "get_", "search_", "list_", "lookup_", "find_", "calc"},
		Deny: map[string]abi.ReasonCode{
			"git_push":   abi.ReasonPolicyBlock,
			"git_merge":  abi.ReasonPolicyBlock,
			"git_tag":    abi.ReasonPolicyBlock,
			"exfiltrate": abi.ReasonSecretExfil,
		},
		SelfModifyGlobs: []string{
			"internal/abi/", "internal/kernel/", "internal/adjudicator/",
			"internal/policy/", "internal/registrations/",
			// Witness machinery (#172 Hole 2): the trees that, if self-edited,
			// would let an RSI loop grade its own homework — the architest gates,
			// the shipgate require-witness rung, and the workspace's dos.toml
			// (lane taxonomy + stamp grammar the referee binds to).
			"internal/architest/", "internal/shipgate/", "dos.toml",
			".git/", ".dos/", "policy.json", "VERSION", "id_rsa",
		},
		RedactFields: []string{"password", "secret", "api_key", "token", "authorization"},
	}
}

// Default is the registered instance.
var Default = New(DefaultPolicy())
