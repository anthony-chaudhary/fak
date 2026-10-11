# Proposed corrected contract for #13664

Status: partial contribution using the corrected contract below. The original
ticket's contradictory acceptance criterion remains open; it has not been
edited or closed. The existing profile sweep is
unchanged; this helper has no serving caller or hardware qualification claim.

## Probability law

Use finite normalized autoregressive target rows P and actual proposal rows Q,
after sampling filters. Drafts must really be sampled from Q. On the sampled
path Q_i(X_i)>0 is mandatory; zeros elsewhere are allowed. Acceptance uniforms
and the correction uniform must be independent of each other and drafting.

With w_0=1, recursively set w_i=min(1,w_(i-1)P_i(X_i)/Q_i(X_i)). For i<gamma,
S_i=sum_x max(w_i P_(i+1)(x)-Q_(i+1)(x),0), and
h_i=S_i/(S_i+1-w_i); h_gamma=w_gamma. Define h_i=1 when w_i=1.
Keep the largest i whose uniform is below h_i, defaulting to zero. Draw the
correction from normalized positive residual at that prefix, or the target
bonus row when all retained drafts are accepted.

Primary method: [Sun et al., Algorithm 2 and Theorem 1](https://arxiv.org/html/2403.10444v3).
This is an independent Go implementation of the stated mathematics, with no
upstream engine code copied, vendored, or executed.

## Acceptance-criterion correction

The ticket asks that block and token verification have equal accepted-length
distributions. That is false in general. For gamma=2, P=(1/3,2/3) and
Q=(2/3,1/3), exact enumeration gives:

| tau | joint block | sequential token |
| --- | --- | --- |
| 0 | 3/9 | 3/9 |
| 1 | 1/9 | 2/9 |
| 2 | 5/9 | 4/9 |

Thus their expected accepted lengths differ, 11/9 versus 10/9. The regression
pins both distributions, rather than forcing the incorrect equality. It
compares the completed three-token output law of each algorithm against direct
target sampling, filling any remaining positions from the target after the
correction. Its finite quadrature exactly covers all probability boundaries
in the fixtures, including a prefix-dependent stopping example.

## LiveLength is a verified-row count

This proposal explicitly resolves the ticket's ambiguous `commit <= L`:
LiveLength=L counts target rows INCLUDING correction/bonus. It allows at most
L-1 draft tokens, so total emitted tokens Accepted+1 never exceeds L. L=1 is a
target-only sample; L=0 refuses. Nil uses the whole block. At most 65 target rows
are supported, matching the package's existing maximum 64-draft sweep.

Unused suffix target/proposal/random rows may be absent and suffix draft ids
may be invalid. The helper recomputes the final threshold and bonus at the
actual retained length; it never pretends an unverified target row exists.

A cap may depend on the included draft prefix, never on an excluded token or
verifier randomness. For a stopping tree, write A=sum_x Q_next(x)w_next(x)=w-S.
At terminal nodes the accepted-prefix probability is w. At nonterminal nodes,
backward induction gives A+(1-A)h=A+S=w. Terminal correction uses the target;
nonterminal residual fills the remaining probability. This argument requires
causal rows and independent uniforms. The helper cannot infer those properties
from arrays: they are explicit caller obligations, not runtime assertions.

Float64 execution has rounding error. Probability sums are checked within
1e-12, and undefined selected residuals refuse. Log weights avoid prematurely
losing a tiny prefix weight that a later likelihood ratio can revive. This
does not establish bit-identical model logits or hardware execution.

## Source-study pin and license disposition

The originating study recorded `adapter/block_verify.py:7-11,28,43,56` and
`adapter/verify_cap.py:6-21,60,64,216-220` at
`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks@cad252b7d3000cf21d919764e70a6912b6d3b0b0`.
Those anchors are provenance from the existing study, not independently
re-fetched engine code. That source is AGPL-3.0-or-later: inspire-only,
no expressive copying. Implementation used the ticket's equations and the
primary mathematical paper, plus independently authored deterministic tests.
