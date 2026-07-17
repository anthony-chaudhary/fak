# Negation involution A/B - 2026-07-17

The prototype represents latent negation as a two-dimensional linear map `N`. The strict
reflection and pi-rotation parameterizations satisfy `N^2 = I` by construction. Therefore
`N(N(x)) = x` is a structural identity: double-negation elimination is free and does not
need to be learned as a separate fact. The unconstrained fitted control does not preserve
that law; it needs the explicit `||N^2-I||_F` regularizer.

| variant | `||N^2-I||_F` | double-negation error | synthetic accuracy | conclusion |
|---|---:|---:|---:|---|
| strict involution | 0 | 0 | 1.00 | free by reflection structure |
| pi rotation | 0 | 0 | 1.00 | free by rotation structure |
| unconstrained | 0.416908263290619 | 0.3528190045901722 | 0.50 | regularizer required |

The machine witness is `internal/model/testdata/negation_involution_ab.json`; `TestInvolution`
uses tolerance `1e-12`, while `TestInvolutionAB` recomputes every table value. The feature is
explicitly gated: disabled construction is identity. These are deterministic synthetic
results, not a claim about live model weights.
