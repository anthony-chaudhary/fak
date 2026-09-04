// Package agentsched implements multi-level prioritized agent task scheduling, host envelope telemetry
// monitoring, provider headroom enforcement, and lane clearance governance for concurrent agent workflows.
//
// Invariant: Four-gate admission evaluation is fail-closed; a task is only dequeued and admitted when all four gates pass.
//
// Contract:
//   - Multi-level priority queues maintain FIFO ordering within each priority tier (P0 System through P3 Speculative).
//   - Capacity saturation returns immediate, non-blocking ErrQueueFull without dropping existing higher-priority tasks.
//   - Telemetry-driven dynamic load shedding pauses P3 tasks and downscales worker concurrency during severe host stress.
//   - Gate 4 enforces tree-disjoint lease clearance before any task execution commences.
//
// Guard: Host telemetry breaches (CPU >= 85%, memory exhaustion, power/thermal sag) downscale concurrency and shed speculative load.
package agentsched
