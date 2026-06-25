// Package taskmgr is fak's process-local task manager concept.
//
// It tracks the work a running process says it is doing as task and step records,
// then projects a point-in-time snapshot with wall time, process resource samples,
// per-step/concept runtime, progress, and ETA when enough progress data exists.
//
// Tier: foundation (1) - see internal/architest. This package is stdlib-only and
// deliberately off the request path: it is a reference fold a front door can embed,
// not a hosted scheduler, durable store, or fleet coordinator.
package taskmgr
