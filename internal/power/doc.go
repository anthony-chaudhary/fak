// Package power provides cross-platform OS power assertion and wake-lock management
// for background agent execution.
//
// Tier: primitive (1) — see internal/architest. It is stdlib-only and imports
// nothing internal, so it sits at the bottom of the layering DAG and can be
// imported by any higher layer (such as the gateway, background supervisor, or CLI).
//
// On macOS (Darwin), assertions are managed via IOKit IOPMAssertion when cgo is enabled,
// with a fallback to `caffeinate` subprocess invocation when cgo is disabled.
// On Windows, assertions are managed via SetThreadExecutionState and PowerCreateRequest.
// On Linux, assertions are managed via systemd-inhibit (with graceful fallback).
// On other operating systems, a no-op fallback is used.
package power
