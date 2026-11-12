// Package servechain is the multi-tenant billing gateway chain mounted under
// `fak serve --billing` (PLAN-C Phase 3 item 12, ticket bm-p3-12-servechain).
//
// The chain is an HTTP handler that sits in front of the upstream engines a
// billing manifest declares and admits every served request through the same
// closed pipeline, composing only the public billing packages:
//
//	route check   -> served-name closed set, method check (404/405)
//	auth          -> keystore bearer lookup (401 unknown / 403 revoked)
//	scrub         -> Authorization, X-Api-Key, Cookie, X-Fak-Session stripped
//	bounded body  -> Config.MaxBodyBytes (413)
//	shape         -> JSON object with a "model" (400)
//	allowlist     -> model name must be a served ModelRef (404)
//	rate window   -> manifest Limits rpm/tpm (429)
//	policy        -> billing.Policy.Admit (402)
//	reserve       -> ledger.Reserve (402 on insufficient balance)
//	rate commit   -> window commit only after full admission
//	proxy         -> httputil.ReverseProxy to the bound upstream, model
//	                 rewritten to ModelRef.UpstreamModel, usage captured
//	settle        -> asynchronous bounded queue; ledger.Settle with the
//	                 pricebook-estimated units
//
// Valuation is by the SERVED model name (the name the client asked for): a
// request whose usage cannot be established leaves its reservation PENDING —
// the ledger's own convention for unknown outcomes — never zero. An upstream
// failure releases the reservation and returns 502.
//
// The chain is inert by construction: it exists only when the operator passes
// --billing, and `fak serve` without it never constructs one.
package servechain
