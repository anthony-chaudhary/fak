// Package chatopsdetach provides the pure detached-execution decision kernel for
// chatops ACT verbs (dispatch, resume, bench). It folds inbound commands, admission
// verdicts, and prior spool records into deterministic actions (dispatch, re-ack, refuse)
// and evaluates detached run liveness for blocker escalation without direct I/O.
package chatopsdetach
