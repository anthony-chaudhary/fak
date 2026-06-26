# Witness attestation signing-key decision

Issue [#832](https://github.com/anthony-chaudhary/fak/issues/832) asks for a
portable witness verdict as an in-toto Statement inside a DSSE envelope. The
implementation must not start until the signing-key story is explicit.

## Decision

The first fak-native signing path should use an **operator-supplied Ed25519 key**
for DSSE envelopes.

Do not wire Sigstore/gitsign as the v0 implementation path. Keep it as a later
identity upgrade once the fleet has an OIDC/root-of-trust policy.

Why:

- fak's module currently has zero external Go dependencies and no `go.sum`; an
  Ed25519 DSSE signer/verifier can be implemented with the standard library.
- The first portable witness token must work offline and in CI without an
  interactive browser or network round trip.
- Sigstore keyless signing is the right long-term identity envelope, but its own
  docs tie that model to OIDC identity, Fulcio certificates, Rekor transparency,
  and TUF-distributed roots. That is an operator policy decision, not a leaf
  package default.
- gitsign is specifically a Git commit/tag signing tool. It can improve commit
  identity, but it is not the in-process DSSE envelope signer for an arbitrary
  fak witness predicate.

## Envelope Shape

Emit an in-toto Statement (`_type: https://in-toto.io/Statement/v1`) as the DSSE
payload:

- `subject[].name`: the git ref or commit label the operator asked about.
- `subject[].digest.gitCommit`: the resolved commit SHA.
- `predicateType`: a fak witness predicate URI, for example
  `https://fak.dev/attestation/witness/v1`.
- `predicate.verifiedLevels` and `predicate.sourceLevels`: SLSA Source VSA-shaped
  fields for consumer familiarity, but with fak-specific values unless the source
  control system actually asserts a SLSA Source level.
- `predicate.verdicts`: the local witness checks that produced the token
  (`ancestor`, `committed_path`, `clean_tree`, and the collective-commit
  disjointness verdict when present).

Use the standard DSSE envelope fields:

- `payloadType: application/vnd.in-toto+json`
- `payload`: base64 JSON Statement
- `signatures[].keyid`: fingerprint of the operator-supplied public key
- `signatures[].sig`: Ed25519 signature over the DSSE pre-authentication encoding

## Non-Claims

A git commit SHA is already content-addressed. Restating it in
`subject[].digest` is not the new guarantee. The additive guarantee is that a
specific signing identity attests that it ran fak's local witness checks over
that SHA and got the recorded verdict.

The token does not replace local `internal/witness` checks, does not make a
commit cross-machine atomic, and does not prove the repo owner endorses the
commit unless the operator's key policy says so.

## Sources

- in-toto Attestation Framework spec:
  <https://github.com/in-toto/attestation/blob/main/spec/README.md>
- in-toto Envelope layer / DSSE requirements:
  <https://github.com/in-toto/attestation/blob/main/spec/v1/envelope.md>
- SLSA Source v1.2 requirements:
  <https://slsa.dev/spec/v1.2/source-requirements>
- Sigstore keyless signing overview:
  <https://docs.sigstore.dev/cosign/signing/overview/>
- gitsign README:
  <https://github.com/sigstore/gitsign>
