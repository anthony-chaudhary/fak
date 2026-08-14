# Human control intent index — spine inventory (2026-08-14)

## Value frame

- **For:** a human supervising one or many running agents.
- **Problem:** free-form feedback mixes the desired outcome, its strength, its target, its rationale, and delivery timing; systems currently reduce that to raw prose or a tiny lifecycle enum.
- **Today:** fak has typed session lifecycle decisions (`continue`, `end-turn`, `pause`, `stop`) and a typed intervention model, but no reusable index of what the human is actually trying to accomplish.
- **Better because:** `humanctl` makes semantic intent queryable and composable while preserving text it cannot honestly classify.
- **Witness:** `go test ./internal/humanctl` resolves representative phrases, retains target/reason/text, applies explicit strengths, and rejects impossible ordered programs.

Problem centrality is **Core**: this is the human side of the agent-kernel control seam. P1 managed context improves because repeated prose can become compact typed state; P2 net-true efficiency remains `not yet` pending a live extractor benchmark; P3 bounded adaptation improves because the closed index and validation constrain action; P4 integrated operations remains a follow-on until a session-control surface consumes the type.

## Model

A control instruction is not just a verb:

```
semantic verb × strength × target × reason × residual text
```

Delivery is a separate dimension:

```
delivery = immediate | next_safe_point | next_turn | queued | draft
```

The first spine implements the semantic side. It deliberately does not equate **steer**, **queue**, **inject**, or **send now** with human intent: those are delivery choices that can carry any semantic control. This distinction is repeatedly visible in Codex and Claude Code issue histories, where users ask for explicit queue-versus-steer behavior without changing message meaning.

Ordered composition matters. “Avoid X, prioritize Y, then verify Z” is a small control program. A pause or stop cannot precede executable controls without lying about what will run, so the spine rejects trailing instructions after non-composable lifecycle verbs.

## Initial inventory

| Family | Verbs | What the human is trying to do |
|---|---|---|
| Evaluation | `flag_concern`, `reject` | express uncertainty or a known negative judgment without forcing a fabricated diagnosis |
| Direction | `reinforce`, `redirect`, `avoid` | preserve, bend, replace, or exclude a trajectory |
| Allocation | `prioritize`, `deprioritize` | change attention, ordering, or resource share |
| Scope | `narrow`, `broaden` | change the admitted problem or solution space |
| Assurance | `investigate`, `verify` | gather evidence or demand a checkable witness |
| Recovery | `retry`, `undo` | repeat or return toward a prior state |
| Lifecycle | `continue`, `pause`, `resume`, `stop` | control whether execution proceeds |

`low`, `medium`, `high`, and `absolute` are ordinal strengths, not separate verbs. Unspecified strength remains unspecified in the submitted instruction; an executor may ask for the declared default explicitly with `EffectiveStrength`.

## Evidence sampled

### This repository and workstation

- Existing design: `docs/notes/CONCEPT-INTERVENTION-AS-STATE-OPERATOR-2026-07-17.md` models interventions as state operators, while `internal/policy/control_vocab.go` closes the lifecycle vocabulary. The new index is additive and orthogonal.
- Existing backlog: issue #2753 asks for a closed out-of-band control vocabulary; #2439 asks for principal-tagged control routing; #6365 binds reviewed workflow concepts to bounded steering actions. None provides a broad semantic intent index.
- A literal, case-insensitive scan of 4,301 local Codex session JSONL files found examples of the rare phrases named by the operator (`double down` 28, `seems off` 22, `seems wrong` 18, `wrong because` 33) alongside much larger lifecycle/assurance vocabulary. These counts include repeated embedded context and are **discovery evidence, not usage-frequency measurements**.
- Git history contains many steering, stop, verify, priority, and recovery terms, confirming that the vocabulary spans runtime and development trajectories rather than one UI.

### External field evidence (retrieved 2026-08-14)

- OpenAI Codex issue #33303 names an explicit “Send / Queue / Steer / Draft” surface; #37883 asks for Queue and Steer controls; #23615 adds Stash. These support separating semantic intent from delivery and draft state.
- Anthropic Claude Code issues #64624 and #78401 ask for mid-generation steering/injection, while #85210 and #50246 ask for next-turn queueing. The same prose can require different delivery semantics.
- Model Context Protocol 2025-06-18 specifies cancellation and progress as distinct protocol utilities, supporting lifecycle/progress controls as typed protocol concerns rather than arbitrary message strings:
  - https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/cancellation
  - https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/progress
- Amershi et al., *Guidelines for Human-AI Interaction* (CHI 2019), includes support for efficient invocation/dismissal, correction, and scoped controls; these are interaction goals rather than a wire vocabulary:
  - https://www.microsoft.com/en-us/research/wp-content/uploads/2019/01/Guidelines-for-Human-AI-Interaction-camera-ready.pdf

## Honest boundary and next expansion axes

This is an index and algebra, not an NLP classifier. Exact aliases demonstrate the contract, but arbitrary text extraction needs confidence, ambiguity, negation, scope, and provenance. Important unimplemented axes include delivery timing, addressee/cardinality, temporal duration, reversibility, urgency, confidence, authority, condition/trigger, resource budget, and acknowledgement/effect status. Those should extend the typed instruction rather than multiplying every verb into a Cartesian-product enum.
