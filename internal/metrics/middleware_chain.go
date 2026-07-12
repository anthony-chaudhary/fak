// middleware_chain.go — fail-closed COMPOSITION of the middleware seam (#2904).
//
// middleware_contract.go gave ONE middleware an explicit fail contract via Apply.
// But Hermes' hermes.middleware.v1 is "nested via next_call": middlewares COMPOSE,
// each wrapping the next. Composition is exactly where fail-open-by-accident is
// most dangerous — a broken enforcer buried in a chain must still deny the WHOLE
// call, never let a neighbouring link's admit leak through. A single-link contract
// does not, on its own, prove the guarantee survives nesting.
//
// Chain folds a middleware sequence the fak way. Each link runs under ITS OWN fail
// contract (via Apply, so a throwing fail-closed link already became VerdictDeny and
// a throwing fail-open link already became VerdictDefer), and the per-link verdicts
// fold by abi.FoldRank — the kernel's restrictiveness lattice, where the MOST
// restrictive verdict wins. This mirrors the adjudicator kernel's own rule that
// "the fold takes the most-restrictive verdict regardless of order" (internal/
// adjudicator/decide.go): a lattice max is order-independent, so a fail-closed
// enforcer that errors denies the call no matter WHERE it sits in the chain or what
// a later link would have returned. Deny (FoldRank 100) dominates; a fail-open
// observer's Defer (FoldRank 10) yields to any real verdict and never blocks.
package metrics

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// chainBy is the forensic By tag stamped on a verdict SYNTHESIZED by the fold (the
// empty/all-defer case), distinct from the single-link middlewareBy so an audit can
// tell a composed no-opinion from one middleware's own decision. A verdict that a
// concrete link produced is returned verbatim, carrying that link's own By.
const chainBy = "metrics/middleware-chain"

// Chain folds a middleware sequence into one effective verdict under the
// fail-closed composition contract. Each middleware runs under its own fail
// contract (via Apply); the resulting verdicts fold by abi.FoldRank, most
// restrictive wins, first link winning a rank tie (so the FIRST enforcer to refuse
// is the one cited). The fold is a lattice max, hence order-independent: one
// throwing fail-closed enforcer denies the whole chain regardless of position, and
// no later link's Allow can override an earlier Deny.
//
// An EMPTY chain — or one where every link defers (a fail-open observer that
// errored, or a link with no opinion) — returns VerdictDefer, the no-opinion
// verdict the kernel resolves downstream (default-deny if nothing affirmatively
// allowed the call). Chain never fabricates an Allow on its own: an admit is only
// ever a concrete link's own Allow passing through as the least-restrictive winner.
func Chain(ctx context.Context, ms []Middleware, c *abi.ToolCall) abi.Verdict {
	if len(ms) == 0 {
		return abi.Verdict{Kind: abi.VerdictDefer, By: chainBy}
	}
	winner := Apply(ctx, ms[0], c)
	for _, m := range ms[1:] {
		v := Apply(ctx, m, c)
		if abi.FoldRank(v.Kind) > abi.FoldRank(winner.Kind) {
			winner = v // strictly more restrictive; a rank tie keeps the earlier link
		}
	}
	return winner
}
