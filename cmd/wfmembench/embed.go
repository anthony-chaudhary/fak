package main

import _ "embed"

// fixtureJSONL is the deterministic workflow-memory benchmark transcript. It is a
// finished Claude-Code-shaped session whose eight tool results exercise the six
// workflow-memory hazards issue #434 names: a clean tool result, a stale mutable
// source, a poisoned/sealed result (prompt injection), a tombstoned page, a
// multi-agent (sub-agent) handoff, and a verified vs. unverified effect claim.
// A second sealed page (secret exfil) hardens the poison-leak probe.
//
//go:embed testdata/workflow-memory-session.jsonl
var fixtureJSONL []byte
