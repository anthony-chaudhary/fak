package main

import (
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// serveUnkeyedSessionBudget is the budget each gateway-derived per-connection
// session starts with; nil leaves unkeyed sessions at the table default.
var serveUnkeyedSessionBudget atomic.Pointer[session.Budget]

func setServeUnkeyedSessionBudget(b *session.Budget) {
	serveUnkeyedSessionBudget.Store(b)
}

// seedServeUnkeyedSession gives a header-less caller's per-connection session
// its own budget the first time the table sees it.
func seedServeUnkeyedSession(traceID string) {
	b := serveUnkeyedSessionBudget.Load()
	if b == nil || !gateway.IsUnkeyedTrace(traceID) {
		return
	}
	serveSessions.SeedBudget(traceID, *b)
}
