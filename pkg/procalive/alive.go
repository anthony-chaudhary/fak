// Package procalive is the one process-liveness probe lock reclaimers use to
// decide whether a recorded holder pid is gone.
//
// A false answer lets a peer delete a held lock, so the probe fails closed:
// only proof that the pid does not exist (or has exited) reads as dead, and
// anything inconclusive reads as alive, leaving the lock's TTL to bound a
// wedged holder. It never shells out: a `tasklist` probe took 1.2-2.7s with
// ~1,200 processes on the fleet host (2026-10-09), so its 2s cap timed out,
// read a live publisher as dead, and the publish queue stopped serializing.
package procalive
