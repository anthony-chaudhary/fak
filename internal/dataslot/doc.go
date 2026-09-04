// Package dataslot provides dormant database discovery, descriptor validation,
// and zero-network capability slot management for AI agent environments.
//
// Invariant: data slot detection and validation is fail-closed and deterministic.
// Dormant capability slots never initiate unsolicited network connections or
// background processes, and unverified artifacts are refused by default.
//
// Contract: all detected descriptors must specify a valid database family, non-empty
// identifier, and consistent lifecycle state before admission to runtime contexts.
//
// Guard: inaccessible paths and malformed database headers are skipped fail-closed
// during discovery to ensure scanning cannot panic or stall the host agent.
package dataslot
