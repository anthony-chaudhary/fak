# Contributor License Agreement (CLA)

> **Status: DRAFT pending Netra's legal review.** The infrastructure (this file, the
> sign-off ritual, the references from [`CONTRIBUTING.md`](CONTRIBUTING.md)) is in place;
> the exact legal instrument is counsel's call and may change before it is declared
> binding. Do not treat the wording below as final legal text. See
> [`LICENSING.md`](LICENSING.md) for why this grant exists and
> [`GOVERNANCE.md`](GOVERNANCE.md) for who may change it.

This Contributor License Agreement ("Agreement") is between **Netra Systems** (the
"Steward") and the person or entity ("You") submitting a contribution to the **fak /
fleet** project (the "Project"). It governs Your contributions; it does **not** change the
license under which You *receive* the Project (that is Apache-2.0 for the kernel — see
[`LICENSE`](LICENSE)).

## Why there is a CLA *and* a DCO (the posture)

The Project requires **both**, because they do two different jobs:

| Instrument | What it is | What it certifies | How You give it |
|---|---|---|---|
| **DCO** ([Developer Certificate of Origin](https://developercertificate.org/)) | Lightweight, per-commit provenance | "I wrote this, or have the right to submit it." | `git commit -s` on every commit |
| **CLA** (this file) | One-time, per-contributor grant | The copyright + patent license — **including the sublicense right** — that keeps the Project's layered-licensing optionality open while Netra is the steward. | Statement in Your first PR (below) |

The DCO is cheap provenance and is enforced below the agent layer as a git hook on every
commit. The CLA is the relicense-enabling grant; landing it **before the first external
contribution** is the one irreversible, time-sensitive licensing move. Contributions are
accepted **inbound = outbound**: Your change is licensed to the public under the same
license that governs that part of the tree (today, Apache-2.0 for the kernel), *in
addition to* the grant to the Steward below.

## 1. Definitions

- **"Contribution"** — any original work of authorship (code, documentation, configuration,
  test data) that You intentionally submit to the Project for inclusion, in any form and
  through any channel (pull request, patch, commit, issue attachment).
- **"Submit"** — any act by which a Contribution is sent to the Project, excluding work
  conspicuously marked or otherwise designated in writing as "Not a Contribution."

## 2. Copyright license (and the sublicense/relicense right)

You grant the Steward and recipients of software distributed by the Steward a perpetual,
worldwide, non-exclusive, royalty-free, irrevocable copyright license to reproduce,
prepare derivative works of, publicly display, publicly perform, **sublicense, and
distribute** Your Contributions and such derivative works.

The **sublicense right** is the operative clause: it lets the Steward distribute Your
Contribution under the layered licenses described in [`LICENSING.md`](LICENSING.md) —
Apache-2.0 today for the kernel, with a serving-surface relicensing option (AGPL-3.0 +
commercial) held in reserve. **Invariant:** the Steward will never place the *kernel*
(the importable ABI + adjudication core) under a source-available-only or proprietary-only
license; the kernel stays open under Apache-2.0 (or a successor OSI-approved permissive
license). The sublicense right exists for the *serving surface and value-capture product
layers*, not the kernel.

You retain all right, title, and interest in Your Contributions; this is a license, not an
assignment.

## 3. Patent license

You grant the Steward and recipients a perpetual, worldwide, non-exclusive, royalty-free,
irrevocable (except as below) patent license to make, have made, use, offer to sell, sell,
import, and otherwise transfer Your Contribution, where such license applies only to those
patent claims licensable by You that are necessarily infringed by Your Contribution alone
or by combination of Your Contribution with the Project.

If any entity institutes patent litigation against You or any other entity alleging that
Your Contribution, or the Project to which You contributed, constitutes direct or
contributory patent infringement, then any patent licenses granted to that entity under
this Agreement terminate as of the date such litigation is filed (mirroring Apache-2.0 §3).

## 4. Your representations

You represent that: (a) each Contribution is Your original creation, or You have the right
to submit it under this Agreement; (b) You are legally entitled to grant the above
licenses; and (c) if Your employer has rights to intellectual property You create, You have
received permission to make the Contribution on behalf of that employer, or Your employer
has executed a Corporate CLA with the Steward (see §5).

You are not expected to provide support for Your Contributions, and Contributions are
provided "AS IS" without warranties, to the extent permitted by law (consistent with
Apache-2.0 §7).

## 5. Corporate contributors

If You are contributing on behalf of an entity that owns the IP in Your work, the entity
must accept a **Corporate CLA (CCLA)** so that all of the entity's contributors are covered
by one agreement. The CCLA carries the same §2/§3 grants made by the entity rather than the
individual. **Until the CCLA template is finalized in legal review, corporate contributors
should contact the Steward** (see [`SECURITY.md`](SECURITY.md) / project README for contact)
before their first contribution.

## 6. How You accept this Agreement

Until an automated CLA-assistant is wired into the PR flow, state the following in Your
**first** pull request:

> *"I have read the CLA Document and I hereby sign the CLA."*

Combined with Your DCO sign-off (`git commit -s`) on every commit, this records both the
per-commit provenance and the one-time grant. Acceptance applies to all of Your past and
future Contributions to the Project unless You withdraw in writing before a Contribution is
merged.

---

_This draft is intentionally short and modeled on the widely-used individual-CLA pattern
(Apache ICLA lineage). It is **not** a substitute for legal advice and is subject to
revision in Netra's review. The CLA-vs-DCO decision recorded here — require **both** — is
the Project's deliberate posture, not a default._
