// Package loopmgr records and summarizes long-running agent loop events.
//
// It is deliberately small: a stdlib-only, append-only JSONL ledger with a
// SHA-256 hash chain and a read fold. It does not schedule, spawn, notify, or
// authorize anything by itself; those stay in the callers that produce events.
package loopmgr
