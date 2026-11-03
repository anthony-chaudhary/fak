// Package guardedremotemaint embeds the fak gateway's guarded remote-maintenance
// syscall as a deliberately small Go API.
//
// A session represents one source-configured, non-human agent. The package mints
// and retains both HTTP and MCP launch credentials, exposes no handler or bearer,
// and sends only the canonical fak_remote_maintenance tool through the gateway's
// authenticated HTTP middleware. HostConfig.Authority remains the sole source of
// maintenance grants; AgentID is attribution, not human approval.
//
// Execute consequently crosses the same kernel decisions, authority validation,
// resource-scope projection, effect boundary, and result-admission gates as the
// MCP surface. It does not provide a policy/config overlay or a side door to the
// authority's Execute method.
//
// Wiring status (audited 2026-11-03, ticket debtwave2-pkg-guardedremotemaint):
// no production code imports this package. The audit ran
// rg -n "github.com/anthony-chaudhary/fak/pkg/guardedremotemaint" --glob '*.go'
// across the repo and found zero non-test importers; the sibling pkg/remotemaint
// is the wired path (internal/adjudicator/remote_maintenance.go and
// cmd/fak/guard_remote_maintenance.go). The minimal production connection is one
// facade entry point exposed through that adjudicator seam (<= 3 files), which
// sits outside this package's lane fence, so it is recorded here as the open
// follow-on rather than forced in-lane. Until a cmd/-reachable importer lands,
// the debt-lanes reader holds this lane at the tested ceiling (maturity 5.0);
// a cross-package loopback witness test under cmd/fak raises it to integrated.
package guardedremotemaint
