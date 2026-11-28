// Structural adaptation of triton-inference-server/perf_analyzer GenAI-Perf
// multi-turn session-concurrency measurement (multi_turn.md:20 @ 24f6b9af4db5, BSD-3-Clause).
package agentbench

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
)

var ErrInvalidSessionSpec = errors.New("agentbench: invalid session concurrency spec")

// TurnMetric records one turn's latency and token volume within a multi-turn session.
type TurnMetric struct {
	SessionID    int   `json:"session_id"`
	TurnIndex    int   `json:"turn_index"`
	LatencyNS    int64 `json:"latency_ns"`
	OutputTokens int   `json:"output_tokens"`
}

// SessionConcurrencyReport aggregates concurrent multi-turn session execution.
type SessionConcurrencyReport struct {
	NumSessions       int          `json:"num_sessions"`
	TurnsPerSession   int          `json:"turns_per_session"`
	MaxActiveSessions int          `json:"max_active_sessions"`
	CompletedTurns    int          `json:"completed_turns"`
	FailedTurns       int          `json:"failed_turns"`
	PerTurnMetrics    []TurnMetric `json:"per_turn_metrics"`
}

// TurnRunner executes one sequential turn of sessionID and returns (outputTokens, latencyNS, err).
type TurnRunner func(ctx context.Context, sessionID, turnIndex int) (outputTokens int, latencyNS int64, err error)

// RunSessionConcurrency runs numSessions concurrently, each executing turnsPerSession
// sequential turns via runner, recording peak concurrent active sessions and per-turn metrics.
func RunSessionConcurrency(ctx context.Context, numSessions, turnsPerSession, maxConcurrent int, runner TurnRunner) (SessionConcurrencyReport, error) {
	if numSessions < 1 || turnsPerSession < 1 || runner == nil {
		return SessionConcurrencyReport{}, ErrInvalidSessionSpec
	}
	if maxConcurrent <= 0 || maxConcurrent > numSessions {
		maxConcurrent = numSessions
	}
	sem := make(chan struct{}, maxConcurrent)
	var active, peak atomic.Int32
	var mu sync.Mutex
	metrics := make([]TurnMetric, 0, numSessions*turnsPerSession)
	failed := 0

	var wg sync.WaitGroup
	for sid := 0; sid < numSessions; sid++ {
		wg.Add(1)
		go func(sessionID int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			cur := active.Add(1)
			for {
				prev := peak.Load()
				if cur <= prev || peak.CompareAndSwap(prev, cur) {
					break
				}
			}
			defer active.Add(-1)

			for turn := 0; turn < turnsPerSession; turn++ {
				toks, latNS, err := runner(ctx, sessionID, turn)
				mu.Lock()
				if err != nil {
					failed++
				} else {
					metrics = append(metrics, TurnMetric{
						SessionID:    sessionID,
						TurnIndex:    turn,
						LatencyNS:    latNS,
						OutputTokens: toks,
					})
				}
				mu.Unlock()
			}
		}(sid)
	}
	wg.Wait()

	return SessionConcurrencyReport{
		NumSessions:       numSessions,
		TurnsPerSession:   turnsPerSession,
		MaxActiveSessions: int(peak.Load()),
		CompletedTurns:    len(metrics),
		FailedTurns:       failed,
		PerTurnMetrics:    metrics,
	}, nil
}
