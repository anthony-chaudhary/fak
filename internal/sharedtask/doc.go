// Package sharedtask is the in-memory reference fold for collaborative task
// records. It is deliberately off the hot path: adapters can reuse its patch,
// conflict, scope, event, and journal semantics without each inventing a separate
// task-state contract.
package sharedtask
