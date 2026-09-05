package main

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// toGatewaySessionState projects internal/session.State into the gateway's
// session-internals-blind wire DTO. Run becomes its lowercase token; everything
// else is a 1:1 field copy.
func toGatewaySessionState(s session.State) gateway.SessionState {
	return toGatewaySessionStateAt(s, time.Now())
}

// toGatewaySessionStateAt is the clock-injectable core of toGatewaySessionState: the
// only time-dependent projection is the wall-clock TimeBudget (elapsed keeps ticking in
// real time even when no token is spent), so `now` is threaded explicitly here — matching
// the session package's clock-injection discipline — and the process-boundary wrapper
// above supplies time.Now(). A deterministic `now` lets a test assert the projected
// elapsed/remaining without a sleep.
func toGatewaySessionStateAt(s session.State, now time.Time) gateway.SessionState {
	return gateway.SessionState{
		TraceID:    s.TraceID,
		Run:        s.Run.String(),
		TokensUsed: s.Cost.TotalTokens(),
		TokenUsage: s.Cost.TotalTokens(),
		Budget: gateway.SessionBudget{
			TurnsLeft:         s.Budget.TurnsLeft,
			TokensLeft:        s.Budget.TokensLeft,
			ContextTokensLeft: s.Budget.ContextTokensLeft,
			// Context ceiling + last-turn resident window: the two signals the outbound-compaction
			// burst gate needs to reason about a context-budgeted-but-turn-unbounded session's
			// horizon. Both project 0 (omitempty) for an un-budgeted or never-debited session, so
			// the default `fak guard -- claude` wire form is unchanged and no horizon is derived.
			ContextTokensCap:      s.Budget.ContextTokensCap,
			ResidentContextTokens: s.Cost.LatestContextTokens(),
			// The priced spend axis (#2762): carried on the read projection so the envelope
			// route's returned/observed state is honest AND a partial `fak session budget`
			// edit can PRESERVE a live spend ceiling (mergeBudget reads it back) instead of
			// silently clearing it. 0 (omitempty) on a session with no spend budget.
			SpendMicroCentsLeft: s.Budget.SpendMicroCentsLeft,
			SpendMicroCentsCap:  s.Budget.SpendMicroCentsCap,
		},
		Priority:       s.Priority,
		Pace:           gateway.SessionPace{MaxTokensPerTurn: s.Pace.MaxTokensPerTurn, MinTurnGapMs: s.Pace.MinTurnGapMs},
		Reason:         s.Reason,
		ContinuationID: s.ContinuationID,
		ParentTrace:    s.ParentTrace,
		Generation:     s.Generation,
		CacheAffinity: gateway.SessionCacheAffinity{
			Action:      s.CacheAffinity.Action,
			AffinityKey: s.CacheAffinity.AffinityKey,
			FromTraceID: s.CacheAffinity.FromTraceID,
			ToTraceID:   s.CacheAffinity.ToTraceID,
			Reason:      s.CacheAffinity.Reason,
		},
		ResetTransaction: toGatewayResetTransaction(s.ResetTransaction),
		ProviderBoundary: gateway.SessionProviderBoundary{
			Schema:            s.ProviderBoundary.Schema,
			Provider:          s.ProviderBoundary.Provider,
			Source:            s.ProviderBoundary.Source,
			PreviousTrace:     s.ProviderBoundary.PreviousTrace,
			ProviderSessionID: s.ProviderBoundary.ProviderSessionID,
		},
		Assumptions: toGatewaySessionAssumptions(s.Assumptions),
		Time:        toGatewaySessionTime(s.Time, now),
		// Throughput envelope read projection (#2762): the configured expected/min rates
		// plus the measured sustained rate the floor is judged against — the field that
		// makes a `throughput=`/`min_throughput=` ceiling legible in `fak session status`.
		// An unconfigured axis yields the zero SessionThroughput (ObservedTokensPerSec is 0
		// with nothing observed), which omitzero drops so the pre-#2762 wire shape is intact.
		Throughput: gateway.SessionThroughput{
			ExpectedTokensPerSec: s.Throughput.ExpectedTokensPerSec,
			MinTokensPerSec:      s.Throughput.MinTokensPerSec,
			ObservedTokensPerSec: s.Throughput.ObservedTokensPerSec(),
		},
		Rev: s.Rev,
	}
}

// toGatewaySessionTime projects a session's wall-clock TimeBudget into the read-only wire
// form `fak session status` renders — the field that finally makes `--max-duration`
// legible (it was armed and enforced, but never observable). It surfaces the budget
// whenever a wall-clock envelope is configured OR the clock has ticked at all, so an
// UNBOUNDED-but-running guard session ("--max-duration 0 … still tracked for session
// status") still reports its elapsed time. A never-started, unconfigured TimeBudget
// projects to the zero SessionTime, which omitzero drops from the wire entirely.
func toGatewaySessionTime(tb session.TimeBudget, now time.Time) gateway.SessionTime {
	q := tb.Query(now)
	elapsed := tb.Elapsed(now)
	if !q.Bounded && elapsed <= 0 {
		return gateway.SessionTime{}
	}
	return gateway.SessionTime{
		Bounded:          q.Bounded,
		Exceeded:         q.Exceeded,
		ElapsedSeconds:   int64(elapsed / time.Second),
		RemainingSeconds: int64(q.Remaining / time.Second),
		LimitSeconds:     int64(q.Limit / time.Second),
	}
}

func toGatewaySessionAssumptions(in []session.Assumption) []gateway.SessionAssumption {
	if len(in) == 0 {
		return nil
	}
	out := make([]gateway.SessionAssumption, 0, len(in))
	for _, a := range in {
		out = append(out, gateway.SessionAssumption{
			Key:        a.Key,
			Statement:  a.Statement,
			Source:     a.Source,
			Confidence: a.Confidence,
			Expiry:     a.Expiry,
			SourceRef:  a.SourceRef,
		})
	}
	return out
}

func toGatewayResetTransaction(tx session.ResetTransaction) gateway.SessionResetTransaction {
	out := gateway.SessionResetTransaction{
		Schema:       tx.Schema,
		OldTrace:     tx.OldTrace,
		NewTrace:     tx.NewTrace,
		SeedDigest:   tx.SeedDigest,
		Contributors: append([]string(nil), tx.Contributors...),
		BudgetRearm: gateway.SessionResetBudgetRearm{
			TurnsLeft:         tx.BudgetRearm.TurnsLeft,
			TokensLeft:        tx.BudgetRearm.TokensLeft,
			ContextTokensLeft: tx.BudgetRearm.ContextTokensLeft,
			ContextTokensCap:  tx.BudgetRearm.ContextTokensCap,
		},
		WarmPrefixDigest: tx.WarmPrefixDigest,
	}
	if len(tx.OmittedSpans) > 0 {
		out.OmittedSpans = make([]gateway.SessionResetOmittedSpan, 0, len(tx.OmittedSpans))
		for _, span := range tx.OmittedSpans {
			out.OmittedSpans = append(out.OmittedSpans, gateway.SessionResetOmittedSpan{
				Index:  span.Index,
				Role:   span.Role,
				Digest: span.Digest,
				Reason: span.Reason,
			})
		}
	}
	return out
}

func toGatewaySessionVerdict(v session.Verdict) gateway.SessionVerdict {
	return gateway.SessionVerdict{
		Proceed:   v.Proceed,
		MaxTokens: v.MaxTokens,
		MinGapMs:  v.MinGapMs,
		State:     toGatewaySessionState(v.State),
		Stop:      v.Stop,
		Reason:    v.Reason,
	}
}

// taintLevelName renders an abi.TaintLabel as its stable wire name. It mirrors
// ifc's unexported taintName (the enum is not ordered by restrictiveness, so it is
// switched, never formatted).
func taintLevelName(t abi.TaintLabel) string {
	switch t {
	case abi.TaintTrusted:
		return "trusted"
	case abi.TaintTainted:
		return "tainted"
	case abi.TaintQuarantined:
		return "quarantined"
	}
	return "unknown"
}

func applyRuntime(rt policy.Runtime) {
	policy.ApplySources(rt)
	ifc.ConfigureDefaultPolicy(ifcPolicy(rt))
	applyRateLimit(rt.RateLimit)
}

// applyRateLimit pushes the manifest-declared rate_limit into the governor singleton
// (issue #699, Epic 8), mirroring how SafeSinks/Authorize reach ifc. A present block
// installs the cap (authoritative over the FAK_RATELIMIT_* env fallback); an absent
// block resets the limiter to inert  -  so editing the cap out of the file on
// --policy hot-reload removes it. Config and accrued counters are separate
// (SetLimit does not wipe budgets), exactly as a mid-flight env change behaves.
func applyRateLimit(r *policy.RateLimitRule) {
	if r == nil {
		ratelimit.Default.SetLimit(ratelimit.Limit{}, ratelimit.KeyPerTrace) // unlimited/inert
		return
	}
	ratelimit.Default.SetLimit(ratelimit.Limit{
		MaxCalls:   r.MaxCalls,
		MaxCost:    r.MaxCost,
		RetryAfter: time.Duration(r.RetryAfterMS) * time.Millisecond,
	}, rateLimitKeyMode(r.Key))
}

// rateLimitKeyMode maps the manifest key string to the governor's KeyMode. The
// manifest validator already guaranteed trace|tool|global (or empty); "" and "trace"
// both mean per-trace (the governor's default dimension).
func rateLimitKeyMode(key string) ratelimit.KeyMode {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "tool":
		return ratelimit.KeyPerTool
	case "global":
		return ratelimit.KeyGlobal
	default:
		return ratelimit.KeyPerTrace
	}
}

func ifcPolicy(rt policy.Runtime) ifc.Policy {
	p := ifc.Policy{AuthorizedEgressHosts: append([]string(nil), rt.Adjudicator.ResearchEgressAllowHosts...)}
	if rt.StrictGatedSinks || len(rt.SafeSinks) > 0 || len(rt.AuthorizeRules) > 0 || len(rt.Sources) > 0 {
		p.GatedSinks = ifc.StrictGatedSinks()
	}
	if len(rt.SafeSinks) > 0 {
		p.SafeSinks = make(map[string]bool, len(rt.SafeSinks))
		for _, tool := range rt.SafeSinks {
			p.SafeSinks[tool] = true
		}
	}
	type rule struct {
		tool string
		sink ifc.SinkClass
	}
	rules := make([]rule, 0, len(rt.AuthorizeRules))
	for _, r := range rt.AuthorizeRules {
		rules = append(rules, rule{tool: r.Tool, sink: sinkClass(r.Sink)})
	}
	if len(rules) > 0 {
		p.Authorize = func(c *abi.ToolCall, into ifc.SinkClass) bool {
			if c == nil {
				return false
			}
			for _, r := range rules {
				if c.Tool == r.tool && into == r.sink {
					return true
				}
			}
			return false
		}
	}
	return p
}

func sinkClass(name string) ifc.SinkClass {
	switch strings.ToUpper(strings.TrimSpace(name)) {
	case "EGRESS":
		return ifc.SinkEgress
	case "EXEC":
		return ifc.SinkExec
	case "DESTRUCTIVE":
		return ifc.SinkDestructive
	default:
		return ifc.SinkNone
	}
}
