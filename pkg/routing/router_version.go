package routing

// RouterVersion is the product version of the fak router — the unified model
// endpoint the model/serving half of this fabric powers (doc.go, "Mission: one
// unified model endpoint for every harness").
//
// It is deliberately separate from the fak-private monorepo VERSION: the
// monorepo versions the whole commercial factory, while RouterVersion tracks
// the router contract (alias -> physical model, failover, degradation ladder,
// and the /v1/router/* observability surfaces) that harnesses point at and
// that is released on its own cadence.
//
// 0.0.1 is the first cut: the router has stabilized enough to version. It is a
// 0.x version on purpose — the HTTP routing contract is not yet frozen, so a
// breaking change does not require a MAJOR bump under SemVer 0.y.z rules.
const RouterVersion = "0.0.1"

// RouterSchema is the versioned identity of the router's observability
// documents (/v1/router/state, /summary, /trace). It changes only when a
// consumer-visible field of those documents changes shape.
const RouterSchema = "fak-router/0.0.1"
