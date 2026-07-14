package main

import (
	"os"
	"strconv"
	"sync"
	"time"
)

const (
	guardWireRetryLimitDefault = 2
	guardWireRetryWindow       = 15 * time.Second
)

type guardWireErrorGauge struct {
	mu   sync.Mutex
	last time.Time
}

func (g *guardWireErrorGauge) Observe(time.Time, error) {
	g.mu.Lock()
	g.last = time.Now()
	g.mu.Unlock()
}
func (g *guardWireErrorGauge) Recent(now time.Time) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return !g.last.IsZero() && now.Sub(g.last) >= 0 && now.Sub(g.last) <= guardWireRetryWindow
}
func (g *guardWireErrorGauge) Consume(now time.Time) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.last.IsZero() || now.Sub(g.last) < 0 || now.Sub(g.last) > guardWireRetryWindow {
		return false
	}
	g.last = time.Time{}
	return true
}

func guardWireRetryLimit() int {
	raw := os.Getenv("FAK_GUARD_WIRE_RETRY_LIMIT")
	if raw == "" {
		return guardWireRetryLimitDefault
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return guardWireRetryLimitDefault
	}
	return n
}

func guardMaybeRetryTransientWireCrash(runErr error, _ *os.ProcessState, command []string, agentName string, transientObserved bool, retries, limit int, _ bool, _ error) ([]string, bool) {
	if runErr == nil || !transientObserved || retries >= limit {
		return nil, false
	}
	flag, known := guardContinueFlagForAgent(agentName)
	if !known {
		return nil, false
	}
	return guardAppendContinueFlag(command, flag), true
}
