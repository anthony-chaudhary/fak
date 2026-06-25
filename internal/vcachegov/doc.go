// Package vcachegov is the vCache Governor — the steady-state policy layer that
// decides, per cacheable prefix, whether to heartbeat-pin it, let it lazy-rebuild,
// ride natural traffic, or evict it; how many prefixes to warm inside rate-limit
// headroom; and how to route chained requests onto a consistent warm shard.
//
// It is milestone M5 of the vCache epic (issue #720). The full design lives in
// docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md; this package implements the
// four acceptance criteria drawn from §5.4 (pin/lazy/evict), §5.5 (rate-limit warm
// budget), §9 + Law D3 (affinity routing), and Law D4 (secret/retention safety).
//
// Like cachemeta's lifecycle.go and placement.go, the Governor is a PURE decision
// layer: the caller injects the calibration M1 produces (arrival rate λ, TTL T,
// read discount r, rate-limit headroom) and the warm set M2/M3 governs, and the
// Governor returns a decision a consumer acts on. It stores no payloads, issues no
// network calls, and never moves bytes — correctness never depends on warmth, so a
// wrong belief can only ever cost money, never corrupt a result (Law A2).
//
// The Governor composes the existing cachemeta primitives rather than duplicating
// them: PrefixStats is projected from a cachemeta.Lifecycle (arrival rate reuses
// Lifecycle.AccessRatePerSec; the TTL clock reuses TierTTL at TierProvider), and a
// pin/evict verdict maps onto the existing Lifecycle.Touch / Lifecycle.Evict ops.
//
// Tier: mechanism (2) — see internal/architest. This package imports only cachemeta
// (tier 1) and the standard library; an upward import fails the architest gate. It
// is deliberately NOT registered into the kernel (internal/registrations): until
// the M1–M3 calibration and warm-set leaves wire it into a live loop, it is an
// off-path decision engine, so it adds zero rungs to the request path.
package vcachegov
