package agent

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// SPAWN PLACEMENT AT THE DISPATCH SEAM (#5420, epic #5416 track E).
//
// A sub-agent or background turn inherits the parent session's model today: the loop
// routes each tool call under the manifest, and a spawn call is just another tool call,
// so the child lands wherever the parent happened to be. modelroute already answers what
// SHOULD happen instead — Roster.PlaceSpawn gives a delegated turn its own walk down the
// zone ladder, and Roster.SpawnClassFor reads the work class an operator DECLARED for
// that agent type — but nothing on a dispatch path called either. This file is that
// caller.
//
// INHERITANCE IS A FLOOR BYPASS, NOT ONLY A BILL. That is the load-bearing half and the
// reason this is worth wiring even where the delegated token share is small. TierPolicy
// fixes the floor by the WORK: a security/release child requires T1 however small the
// request looks. A child that inherits is never handed to Place, and the parent's floor
// was computed for the PARENT's class — so a routine turn legitimately placed on a small
// T2 model can hand destructive work to that same T2 model with every gate in modelroute
// satisfied, because the child never went through one. Placing the child closes the
// safety hole and the cost hole with the same call.
//
// ROUTE-BEFORE-ADJUDICATE IS PRESERVED. The placement is resolved at exactly the point
// resolveToolEngine already ran — before k.Syscall, before the ToolCall exists — so the
// residency PDP still adjudicates the REAL destination. A placement decided at dispatch
// time would fail open past the floor, which is why this hooks the route seam rather
// than the dispatch one.
//
// PRECEDENCE, AND WHY IT GOES THIS WAY. When spawn placement is armed AND the operator
// declared this spawn's type, the placement wins over the manifest's route for that call.
// The manifest route for a spawn call is precisely the parent-shaped decision this issue
// exists to stop inheriting; the placement is a decision about the CHILD's own work. Both
// are operator inputs, and the more specific, explicitly-armed one takes the call.
//
// WHAT IS A DECLARATION AND WHAT IS A GUESS. An UNDECLARED spawn type is left alone: the
// call falls through to today's route. That is not a silent fallback around a
// misconfiguration, it is the contract spawnclass.go states — "the cost of an omission is
// that a spawn keeps whatever placement it has today; the cost of guessing would be a
// sub-agent on a laptop doing work nobody said was laptop work." Nothing gets cheaper by
// accident. Every OTHER failure here is loud: an armed policy with no roster, a candidate
// the roster cannot resolve, a ladder that can serve nobody, a principal not admitted to
// the placed account. A placement policy must never turn a misconfiguration into a silent
// fallback (#5420), so the only quiet path is the one where no operator said anything at
// all.

// SpawnPlacementPolicy is an operator's arming of spawn placement for one run: the pool a
// delegated turn may be placed into, and where the SPAWNING turn landed.
//
// Parent is recorded, never obeyed. PlaceSpawn does not pass it to the placement call at
// all, which is what keeps a vendor parent from pinning its whole subtree to the vendor
// rung — the zero Placement is the honest value for a root turn, and a half-filled one (a
// zone with no model, or a model in an unknown zone) is refused rather than guessed at.
//
// Serving is the liveness snapshot the child is placed against. It defaults to the zero
// report, which is silence everywhere and reaches the identical placement, so a fleet
// with no prober takes the same code path. It is carried because the child is exactly the
// traffic this epic wants on the company's own hardware, and therefore also the traffic a
// dead GPU host hits first.
type SpawnPlacementPolicy struct {
	Parent     modelroute.Placement
	Candidates []modelroute.Candidate
	Serving    modelroute.ServingReport
}

// WithSpawnPlacement arms per-spawn placement for the in-process agent loop: a tool call
// that CREATES delegated work gets its own rung from the roster's ladder instead of
// inheriting the engine the parent turn was routed to.
//
// It composes with WithRouteAccounts rather than replacing it — the roster wired there is
// the same roster consulted here, for the spawn_classes declaration and for resolving the
// placed model to a residency-honest Target.EngineRoute(). Arming this without a roster is
// a wiring error and fails loud on the first spawn call rather than degrading to the
// inherit-the-parent behaviour it was wired to stop. Not arming it at all leaves the loop
// byte-for-byte unchanged, so a caller may pass the option unconditionally with a zero
// policy only if they mean "no candidates" — which is itself a loud refusal, not a quiet
// one.
func WithSpawnPlacement(p SpawnPlacementPolicy) RunOption {
	return func(c *runConfig) {
		policy := p
		c.spawnPlace = &policy
	}
}

// spawnToolNames is the closed set of tool calls that CREATE delegated work in this
// harness, mirroring internal/sessionaudit's set so the placer and the auditor cannot
// disagree about what counts as a spawn.
//
// Matched EXACTLY. This harness also ships TaskCreate, TaskUpdate, TaskList, TaskStop and
// TaskOutput, which are todo-list bookkeeping and spawn nothing; a prefix match on "Task"
// would route them through a placement decision they have no business in. Being
// INCOMPLETE here is safe in the only direction that matters — a spawner missing from
// this set keeps whatever placement it has today — while being over-broad would place
// ordinary calls against a class declared for delegated work.
var spawnToolNames = map[string]bool{
	"Task":     true, // the historical sub-agent spawn tool
	"Agent":    true, // its current name
	"Workflow": true, // a background workflow, which spawns sub-agents of its own
}

// spawnTypeArgKeys are the argument keys a spawn call carries its agent TYPE under, in
// precedence order. The type is the caller's own structured choice of which kind of agent
// to run — unlike the prompt, which is prose, and reading a work class out of prose is
// the guess this epic refuses everywhere (see modelroute.ClassOf).
var spawnTypeArgKeys = []string{"subagent_type", "agent_type", "agentType"}

// spawnTypeFor names the token to look up in the roster's spawn_classes for one call.
//
// The arguments win when they carry a type at all, and then they are the ONLY thing
// consulted — a present-but-undeclared type must NOT fall back to a broader declaration
// on the tool name, because a catch-all "Agent" entry would then answer for the
// "code-reviewer" nobody classified, which is the exact shape of the floor bypass this
// seam exists to close.
//
// The TOOL NAME is the fallback only when the arguments carry no type token at all —
// the untyped spawn, and the background workflow, which is half of what #5420 names. It
// is a legitimate declaration key for the same reason the type token is: it is a
// structured caller choice rather than prose, and an operator who does not write an entry
// for it gets today's behaviour rather than a cheap guess.
func spawnTypeFor(tool, rawArgs string) string {
	if t := spawnTypeFromArgs(rawArgs); t != "" {
		return t
	}
	return tool
}

// spawnTypeFromArgs reads the declared agent type out of a spawn call's arguments,
// returning "" when the arguments are absent, unparseable, or carry no type key. Every
// one of those is "nothing was declared" rather than an error: malformed arguments are
// the kernel's business to refuse, and this seam must not turn an argument-shape problem
// into a routing refusal.
func spawnTypeFromArgs(rawArgs string) string {
	if strings.TrimSpace(rawArgs) == "" {
		return ""
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return ""
	}
	for _, key := range spawnTypeArgKeys {
		if s, ok := args[key].(string); ok && strings.TrimSpace(s) != "" {
			return strings.TrimSpace(s)
		}
	}
	return ""
}

// resolveSpawnEngine returns the engine route a delegated turn should dispatch to, and
// whether spawn placement answered this call at all.
//
// placed=false with a nil error means "not a spawn, not armed, or the operator declared
// nothing for this type" — the caller falls through to the ordinary per-tool-call route,
// which is today's behaviour exactly. Any error is fail-loud and the call never reaches
// the kernel.
func (c runConfig) resolveSpawnEngine(tool, rawArgs string) (string, bool, error) {
	sp, placed, err := c.placeSpawn(tool, rawArgs)
	if err != nil || !placed {
		return "", false, err
	}
	target := sp.Placement.Target
	if !target.Admits(c.principal) {
		// Same fail-closed verdict, and the same shape, as resolveToolEngine's principal
		// refusal: name the principal and the account, never the credential. A spawn must
		// not be the one path that reaches an account this caller is not provisioned for
		// — a cheaper rung is not a weaker tenancy boundary.
		who := c.principal
		if strings.TrimSpace(who) == "" {
			who = "<unattributed>"
		}
		return "", false, fmt.Errorf("spawn placement: principal %s is not admitted to account %q (placed model %q, zone %s): that account's principals allowlist scopes it to another tenant (#5332) — add this principal to the account, or declare a spawn class whose ladder reaches one it is provisioned for",
			who, target.Account, sp.Placement.Model, sp.Placement.Zone)
	}
	return target.EngineRoute(), true, nil
}

// placeSpawn runs the roster's spawn ladder for one call: is this a spawn, did the
// operator declare what that agent type does, and where does the child's OWN class land
// it. It is separated from resolveSpawnEngine so the full SpawnPlacement — the ladder
// walk, the parent relationship, and the inheritance counterfactual — is available to a
// preflight witness without dispatching anything.
func (c runConfig) placeSpawn(tool, rawArgs string) (modelroute.SpawnPlacement, bool, error) {
	if c.spawnPlace == nil || !spawnToolNames[tool] {
		return modelroute.SpawnPlacement{}, false, nil
	}
	if c.roster == nil {
		return modelroute.SpawnPlacement{}, false, fmt.Errorf("spawn placement is armed but no account roster is wired: the roster is where spawn_classes declares what each agent type does and where a placed model resolves to an account — wire WithRouteAccounts alongside WithSpawnPlacement, or drop WithSpawnPlacement; a spawn placer with nothing to read must not silently fall back to inheriting the parent's engine (#5420)")
	}
	class, declared := c.roster.SpawnClassFor(spawnTypeFor(tool, rawArgs))
	if !declared {
		// The one quiet path, and it is a contract rather than a fallback: an undeclared
		// type keeps whatever placement it has today. See the file header.
		return modelroute.SpawnPlacement{}, false, nil
	}
	sp, err := c.roster.PlaceSpawnWithServing(c.spawnPlace.Parent, class, c.spawnPlace.Candidates, c.spawnPlace.Serving)
	if err != nil {
		return modelroute.SpawnPlacement{}, false, fmt.Errorf("spawn placement for %s(type=%q, class=%s): %w — no silent fallback to the parent's engine", tool, spawnTypeFor(tool, rawArgs), class, err)
	}
	return sp, true, nil
}

// resolveCallEngine is the loop's single pre-Syscall route decision for one tool call:
// the child's own placement when this call creates delegated work and an operator
// declared its class, and otherwise the ordinary per-tool-call route.
//
// Both arms resolve to a residency-honest engine id BEFORE the ToolCall is built, which
// is the ordering the residency floor depends on.
func (c runConfig) resolveCallEngine(tool, rawArgs string, callMeta ...map[string]string) (string, error) {
	engine, placed, err := c.resolveSpawnEngine(tool, rawArgs)
	if err != nil {
		return "", err
	}
	if placed {
		return engine, nil
	}
	return c.resolveToolEngine(tool, callMeta...)
}

// ResolveSpawnPlacement applies RunOptions to one spawn call and reports the placement
// the owned loop would bind before kernel submit, without dispatching anything.
//
// It is the delegated sibling of ResolveToolRoute, and it returns the whole
// SpawnPlacement rather than just the engine id on purpose: the ladder walk and the
// inheritance counterfactual are what tell an operator whether re-placing this child is a
// cost change or a floor-bypass fix, and re-deriving that from an engine string is not
// possible. placed=false means no placement was made — not a spawn, not armed, or the
// type is undeclared.
func ResolveSpawnPlacement(tool, rawArgs string, opts ...RunOption) (modelroute.SpawnPlacement, bool, error) {
	return resolveRunConfig(opts).placeSpawn(tool, rawArgs)
}

// ResolveSpawnRoute applies RunOptions to one spawn call and returns the exact engine
// route the owned loop will bind before kernel submit, mirroring ResolveToolRoute for
// delegated work. placed=false means the call falls through to the ordinary tool route.
func ResolveSpawnRoute(tool, rawArgs string, opts ...RunOption) (string, bool, error) {
	return resolveRunConfig(opts).resolveSpawnEngine(tool, rawArgs)
}
