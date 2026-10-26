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
package guardedremotemaint
