# What fell under adversarial review

*Companion to [the core note](expire-by-default.md). This is the single home for every
concession — the core is written in already-corrected voice so the thesis lands cleanly;
the retreats live here, in one place, on purpose.*

The thesis was run through two rounds of adversarial review: a panel that tried to
*disprove* it, then a panel that attacked the *rebuttal* to catch any overclaim merely
relocated one level down. Then four independent readers — a memory researcher, a staff
engineer, a security reviewer, an editor — reacted to it cold. Several claims fell. The
discipline this project preaches (claim → independently-authored witness → recorded
retreat) should apply to its own writing, so here is the ledger.

## What fell

**"Forced, not tuned."** The original claimed the recoverability asymmetry — false
positive silently self-uncorrectable, false negative self-announcing — holds in *every*
regime. It does not. Where the break-even sits is a cost-sensitive threshold,
θ\* = L(FP)/(L(FP)+L(FN)) weighted by base-rate odds; a high-durable-base-rate agent with
an expensive recovery channel can have a *persist*-by-default optimum. And in the
headless, stated-once, one-shot regime the *sign* itself inverts: a byte-exact-evicted
false negative leaves no auditable artifact, while a stale false positive leaves a value a
later read can catch. The forced/tuned line does not run between sign and magnitude — both
depend on the same regime variables and cross together. It runs between the **attended**
pipeline and the **headless** one. The core now scopes the claim to the attended regime
and names the flip.

**The poisoning motivation (OWASP T1).** An earlier draft reached for OWASP Memory
Poisoning to motivate the *durability* gate. Wrong axis: poisoning is a *trust* failure,
defended by the provenance/`ifc` gate that already ships. A competent attacker injects a
durable-*if-true* payload ("always deploy through host Y") that a *correct* truth-duration
classifier routes straight to `durable` — so the durability gate is neither necessary nor
sufficient against T1. The two axes are severed; the core motivates enforcement on the
weaker, harder-to-attack "components have bugs" floor instead.

**"The kernel is the one layer with both."** False on the first counterexample: any sole
writer to the durable store (an app framework owning the only write path) is equally
fail-closed against a text-emitting model. The property doing the work is *sole-path
topology*, not being the kernel. And only *capture* is timing-forced — the durability
*verdict* can be computed lazily at read.

**The memory-*semantics* framing of the headline.** "Context and memory are separated by
truth-duration" reifies an endpoint unobservable at write time, and on the happy path the
write-time durability call is the same kind of judgment Mem0 and Zep already make. The
surviving contribution is an **enforcement-topology** claim — one more fail-closed tag on
a gate that already ships for trust — which is what the core now leads with. Smaller, and
the one the running code actually backs.

## Residual live wounds — still open, stated plainly

1. **The sign-flip is real, not eliminated — only scoped.** Outside the attended regime
   the asymmetry can invert. The core bounds where it holds; it does not make it universal.
2. **The escalate arm is unbuilt and architecturally homeless.** "This span gates an
   irreversible unattended action" is read-time *action* context, not a write-time span
   property — so it can't live at the sole writer to the durable store; it sits at the
   action site, *outside* the enforcement boundary the rest of the thesis leans on. By the
   thesis's own "advisory is not a defense" criterion, the high-stakes arm is therefore
   advisory-by-construction at the privileged layer. Its detector is strictly harder than
   the ~55%-on-STALE classifier, its own false negatives are a new silent-failure surface,
   and "escalate" assumes an attended channel — when none exists, *refuse* trades a safety
   failure for a liveness failure. This is the weakest part of the position.
3. **Eviction is the dual of admission, and it is the under-guarded direction.** The whole
   thesis defends the *write-in* direction (don't let bad bytes become durable). Expire-by-
   default makes *eviction* a security-relevant event — a byte-exact, audit-erasing forget
   is a privileged write-out an adversary will target to induce a silent, unrecoverable
   false negative (mis-route a standing safety rule into `session`, or skew the clock to
   force early expiry). The kernel is not naked here — `recall` re-screens on page-in and
   honors trust-epoch revocation, and `RequestContextChange` tombstones are durable ledger
   rows — but the durability *expiry* path does not yet get the same default-deny treatment
   admission does. **Eviction must be equally fail-closed, and its expiry state must
   survive resume, or "expire-by-default" becomes "forget-on-command" for whoever controls
   the classifier, the freshness oracle, or the clock.** This is the sharpest finding the
   internal rounds missed; it was surfaced by an external security read.
4. **The freshness oracle/clock is undefended.** Arm 2 ("persist-with-freshness") is only
   as fail-closed as the *oracle and clock* that evaluate the stamp, not just the stamp's
   value. An attacker who owns or transiently spoofs the re-verification source, or skews
   the wall-clock a TTL depends on, defeats the arm the core calls safe. The stamp was
   hardened; its adjudicator was not.
5. **The representation gap.** A four-way class can't carry the calibrated posterior the
   θ\* decision rule needs (see the core's "what's actually hard"). Both internal rounds
   accepted the class table as given and argued about what to *do* with the class; an
   external researcher caught that the output *type* can't express the rule.
6. **Unmeasured and relocated.** "Modal agent has a low durable base rate" is a conjecture
   about the deployment population, not a number (STALE's 55% is classifier error, not base
   rate). And the single-points-of-failure are relocated, not removed: arm-routing now
   leans on a classifier whose mis-sort reintroduces the silent failure arm 3 exists to
   prevent, and both the trust severance and the escalation argument ultimately depend on
   the host's source registration being correct.

## The honest size of the contribution

One more fail-closed tag on a gate that already ships for trust — defensible as an
enforcement-topology claim, honestly marked proposed as a memory-semantics one. That it
shrank under attack *and stated where*, rather than dissolving, is the strongest thing
that can be said for it. The reader verdicts converged: sound but small, over-framed in
its first form, worth building the stamp-and-filter half of, worth measuring before
believing.
