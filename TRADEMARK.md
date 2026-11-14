# Netra Fused Agent Kernel (`fak`) — the "fak" wordmark and "fak-certified" certification mark

> **The one control lever a permissive license cannot erode.** Apache-2.0 §6 explicitly
> does **not** grant trademark rights. So while anyone may use, fork, and commercialize the
> Apache-2.0 kernel, the **marks** remain a separate, steward-held lever — the structure
> that ties a "certified" claim to actually passing a conformance suite. This file is the
> trademark policy; it is best-effort and **not legal advice**.

## The marks

| Mark | Type | What it identifies |
|---|---|---|
| **fak** | Wordmark | The project and its official binary/distribution. A short coined term — real clearance/distinctiveness risk, so a clearance search precedes any filing. |
| **fak-certified** | **Certification mark** | That a third-party implementation **passed the fak conformance suite at a named ABI version** — nothing more, nothing less. |

A certification mark is structurally different from an ordinary trademark: its owner does
**not** use it to brand its own goods, and it must be licensed to **anyone** who meets the
published, objective standard, without discrimination. Those duties are features here — they
are exactly what makes "certified" a *verifiable* claim rather than a marketing word.

## The certified claim is defined **behaviorally**

> **`fak-certified at ABI vX.Y` means:** the implementation passes the published
> `fak-conformance` suite at ABI version `vX.Y` — i.e. it produces the **exact verdict
> matrix** the suite checks (adjudication semantics, plancfi intent-CFI, KV-ownership
> behavior) and matches the **golden ABI contract** pinned by the `abi-vX.Y` git tag.

The certified claim is **never** broader than what the verdict matrix and the golden
contract check. It does not assert performance, security beyond the tested floor, fitness
for a purpose, or steward endorsement of the implementer. If the suite does not check it,
the mark does not claim it. This keeps the mark honest and keeps the steward's
non-discrimination duty tractable: the standard is a test run, not a judgment call.

### How to earn and use the mark

1. Run the standalone `fak-conformance` suite at a named ABI tag (`abi-vX.Y`).
2. The suite is green → you may state **"fak-certified at ABI vX.Y"** for that build,
   linking the conformance output. The claim is scoped to the build and the ABI version you
   tested.
3. The mark is **SLA'd to the ABI cadence**: a certification against a superseded ABI
   version must say so (e.g. "certified at ABI v0.1"); it may not be presented as current.

## Permitted use of the "fak" wordmark (nominative / fair use)

You may, without permission, use "fak" to **truthfully refer to** the project: "works with
fak", "a driver for fak", "built on fak", "compatible with the fak ABI". You may keep the
name in unmodified redistributions. This is ordinary nominative fair use and the policy does
not try to restrict it.

## Hosted offerings (providers who host fak for customers)

A provider that runs fak — typically the serving surface — and offers it as a hosted
cloud service names the offering so customers can tell *who* operates it:

- **Permitted forms:** "<Provider> fak" ("AcmeCloud fak"), "fak hosted by <Provider>",
  "powered by fak", "built on fak". These are nominative uses: they truthfully refer to
  the project and identify the provider as the operator.
- **Required hygiene:** the offering states that it is operated by the provider, not by
  the steward; any "fak-certified" claim follows the certification rules above (a green
  conformance run at a named ABI tag).
- **Not allowed:** naming the offering "fak" alone, or "fak <product>" with the mark
  leading, where a reasonable customer would assume an official distribution or steward
  endorsement (for example, "fak Cloud" as a standalone product name), or presenting the
  provider's service as the official fak service.

Hosting the serving surface creates no trademark rights or duties beyond these. For the
licensing layer the offering runs on — including the relicensing notice period — see
[`LICENSING.md`](LICENSING.md).

## Uses that need permission / are not allowed

- Using "fak" or a confusingly similar name **as the name of your own product, service, or
  fork** in a way that implies it *is* the official project or is endorsed by the steward.
- Stating or implying **"fak-certified"** without a corresponding green conformance run at a
  named ABI version. This is the misuse the certification structure exists to deter.
- Modifying the project and continuing to present it under the "fak" name as if it were the
  official distribution.

## Honest limits

This policy **deters the careless, not a well-lawyered incumbent.** Trademark is a
best-effort lever: clearance on a short coined term carries real risk, certification-mark law
imposes anti-use-by-owner and non-discrimination duties on the steward, and enforcement is
bounded by resources. The point is not an impregnable moat — it is to make "certified" mean
a specific, checkable thing, and to keep one lever that survives the permissive license.

## Status / to-do (tracking)

- [ ] Trademark clearance + distinctiveness search on "fak" (precedes filing).
- [ ] File the **fak** wordmark.
- [ ] Reserve the **fak-certified** certification mark + file the certification standard
      (the standard = exactly the conformance verdict matrix at a named ABI tag).
- [x] Write this one-page policy defining the certified claim behaviorally.

See [`GOVERNANCE.md`](GOVERNANCE.md) for the ABI/conformance cadence the mark is tied to,
and [`LICENSING.md`](LICENSING.md) for why the mark is the value-capture lever a permissive
license leaves intact.

---

_Not legal advice. "fak" and "fak-certified" are claimed as marks of Netra Systems; this
policy may change as filings progress and as counsel advises._
