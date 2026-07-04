# Generation Public Narrative

Market-facing rules for talking about fak generations — **shipped**, **next**,
and **research** — honestly and without hype. This is the external half of the
[Generation Contract](generation.md): that doc governs how the fleet labels and
promotes work internally; this doc governs how we describe those horizons to the
market, so a public statement never overclaims what a stream actually is.

This artifact backs [#1668](https://github.com/anthony-chaudhary/fak/issues/1668)
under epic [#1625](https://github.com/anthony-chaudhary/fak/issues/1625),
milestone `Generation G3 - Future` (`gen/future`). It is narrative/standards
work: a decision model for public claims, not a product commitment.

A future agent can act on this doc without re-reading the generation epic: read
the mapping table, apply the four narrative rules, and stamp every public claim
with one honest fence tag.

## Public vocabulary (three words, not four)

Externally we speak in three horizons, not the internal four. The market does
not need our `second-next` distinction; collapsing it keeps the story honest and
legible.

| Public word | Internal stream(s) | What we may say publicly | What we may NOT say |
|---|---|---|---|
| **Shipped** | `gen/now` | "You can run this today: `<command>`." Cite a verifiable witness (test, captured output, commit). | "Production-grade / battle-tested" unless the witness supports it. |
| **Next** | `gen/next` | "Landing soon, dogfooded behind a gate." Name the gate and the missing default-exposure proof. | "Available now" or "just flip a flag" if it is still gated or default-off. |
| **Research** | `gen/second-next`, `gen/future` | "We are studying this / it is an option we carry." Name the open assumption. | "Coming in the next release" or any dated roadmap promise. |

The rule for the fold: `second-next` architecture bets and `gen/future`
long-horizon options both surface publicly as **Research**, because neither has a
default-exposure witness a user could hold us to. Promotion across the internal
streams can change the public word; the reverse is never true — a marketing
deadline must not promote an internal stream.

## The four narrative rules

1. **One honest fence per claim.** Every public statement carries exactly one
   tag: `[SHIPPED]` (a witness a reader can reproduce today), `[NEXT]` (real code,
   gated or default-off, name the gate), or `[RESEARCH]` (an option or memo, name
   the open assumption). This mirrors the `[SHIPPED]`/`[VERIFIED]`/`[PROJECTED]`
   fences already used in [`docs/launch/positioning-brief.md`](launch/positioning-brief.md).
2. **Witness before adjective.** A capability claim leads with the reproducible
   witness (`go run ...`, a test name, `max|Δ|=0`), then the adjective. No
   superlative may appear without the witness that earns it in the same sentence.
3. **Name the boundary of the claim.** State what the claim is *not*: the
   specific mechanism, the scope, the thing a competitor still wins. A narrowed
   true claim beats a broad claim that invites the strawman fight.
4. **No horizon laundering in public.** Do not describe research as "next," or
   "next" as "shipped," to make the roadmap look closer. If a reader could pay us,
   deploy us, or file a bug against a claim, that claim must be `[SHIPPED]`.

## Orthogonality (unchanged from the internal contract)

The public narrative does not relax any of the three orthogonality guarantees:

- **Priority.** The public word is a horizon, not a value judgment. `[RESEARCH]`
  is not "unimportant" and `[SHIPPED]` is not "top priority" — priority still
  comes from issue labels, milestones, and operator decision.
- **Shared trunk.** No public generation story authorizes a branch, a per-release
  fork, or a "generation" version line. Every horizon still lands through `main`
  by explicit path with the same witness and DCO rules. Marketing a generation
  never changes the branch model.
- **Runtime feature gates.** The public word describes the product horizon, not
  runtime exposure. `[NEXT]` code can be live-but-default-off behind a gate;
  `[SHIPPED]` docs can carry no runtime gate at all. A gate decides reachability;
  the narrative decides how we describe the horizon.

## Evidence

**Promotion evidence** (a public word may move closer to Shipped when):

- The internal stream promoted with its own witness (see
  [Generation Contract → Evidence](generation.md#evidence)), *and*
- A user-reproducible proof now exists: a command that runs with no key/GPU, a
  captured output, or a default-exposure witness. Only then may `[NEXT]` become
  `[SHIPPED]` in public copy.

**Demotion / retirement evidence** (a public claim must weaken or be pulled when):

- The witness behind a `[SHIPPED]` claim regressed, went stale, or was found to
  be self-reported where it read as third-party verified → downgrade the fence or
  remove the claim.
- A `[RESEARCH]` option's carry cost exceeded its expected value, or its open
  assumption failed against live evidence → retire the public mention, do not let
  it linger as implied roadmap.
- A public claim was found to overclaim its horizon (laundering) → correct it in
  place; a hidden downgrade (quietly editing copy without recording why) is itself
  an anti-pattern.

## Invalidating assumptions

- **Three words stay legible.** This doc assumes the market reads `shipped / next
  / research` as honest horizon signals rather than as marketing tiers. If public
  audiences start hearing "research" as "vaporware" or "next" as "buy now," the
  vocabulary is doing harm and should be demoted or renamed against that evidence,
  not defended.
- **Witnesses stay cheap to cite.** The rules assume a reproducible witness
  (command, test, captured output) is available for every `[SHIPPED]` claim. If a
  claim's only witness becomes expensive or private, it must drop to `[NEXT]`
  publicly until a cheap public witness exists — the honest fence is only as good
  as the witness a reader can actually reach.

## Non-goals

- **No branch per generation.** This narrative is copy and claim discipline, not a
  release-line or branch strategy. Shared trunk is unchanged.
- **`research`/`gen/future` is not "lower priority."** The public word is a
  horizon label. A Research item can be the most valuable bet we carry; the fence
  says "assumption still open," never "not worth your attention."
- **Not a launch calendar.** This doc defines how to phrase a horizon, not when to
  ship it. Dated promises belong to release readiness, never to a generation word.

## See also

- [Generation Contract](generation.md) — the internal now/next/second-next/future
  taxonomy, promotion verbs, and intake rules this narrative maps to.
- [`docs/launch/positioning-brief.md`](launch/positioning-brief.md) — the honest-fence
  discipline applied to specific launch claims.
