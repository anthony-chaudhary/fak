// Package dataslot provides dormant database discovery, descriptor validation,
// and zero-network capability slot management for AI agent environments.
//
// Detected descriptors specify a valid database family, non-empty identifier,
// and consistent lifecycle state before admission to runtime contexts.
// Dormant capability slots never initiate unsolicited network connections or
// background processes, and unverified artifacts are refused by default.
// Inaccessible paths and malformed database headers are skipped during discovery
// to ensure scanning does not panic or stall the host agent.
package dataslot
