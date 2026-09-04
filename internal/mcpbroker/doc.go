// Package mcpbroker provides an in-kernel broker and mediator for Model Context Protocol
// (MCP) tool servers. It handles server configuration, tool registration, security policy
// filtering (default-deny and argument inspection), call routing, session tracking, and
// runtime operational telemetry.
//
// Invariant: Security policy filtering is strictly fail-closed; if any filter denies a call, tool execution is blocked and a filtered verdict value is returned without invoking upstream handlers.
//
// Contract:
//   - All exported methods on Broker are safe for concurrent use across goroutines.
//   - Security filtering is fail-closed: if any security filter denies a call, execution is prevented and a typed filtered CallResponse is returned.
//   - Policy rejections (denials) are treated as first-class decision values (deny-as-value), not protocol transport errors.
//   - Upstream server allowlist, denylist, and read-only constraints are validated at both tool registration and call routing time.
//
// Guard: Server-level and tool-level policy checks block mutating operations on read-only targets and reject unlisted or explicitly denied tools.
package mcpbroker
