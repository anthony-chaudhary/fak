package microagent

import (
	"context"
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// GatewaySeat binds one independently bounded scheduler to the actual provider
// gateway reached through that seat. Distinct seats may use different accounts,
// endpoints, or provider processes while sharing the microagent control plane.
type GatewaySeat struct {
	ID        string
	Gateway   Gateway
	Scheduler *Scheduler
}

// AffinitySelection is emitted after admission and before the provider call.
type AffinitySelection struct {
	Trace       string
	AffinityKey string
	SeatID      string
}

type AffinityObserver func(AffinitySelection)

type AffinityGatewayConfig struct {
	// DisableAffinity is the explicit ablation control. Capacity remains bounded,
	// but each call rotates its first seat probe instead of preferring a key.
	DisableAffinity bool
	Observe         AffinityObserver
}

// SessionAffinityGateway routes a session to an actual gateway seat. Affinity is
// default-on: an explicit context key wins, then session.Table CacheAffinity,
// then trace identity. Busy preferred seats fall back through SeatPool.
type SessionAffinityGateway struct {
	pool    *SeatPool
	byID    map[string]Gateway
	table   *session.Table
	disable bool
	observe AffinityObserver
}

func NewSessionAffinityGateway(seats []GatewaySeat, table *session.Table, cfg AffinityGatewayConfig) (*SessionAffinityGateway, error) {
	if len(seats) == 0 {
		return nil, errors.New("microagent: affinity gateway requires at least one seat")
	}
	poolSeats := make([]Seat, 0, len(seats))
	byID := make(map[string]Gateway, len(seats))
	for _, seat := range seats {
		if seat.ID == "" || seat.Gateway == nil || seat.Scheduler == nil {
			return nil, errors.New("microagent: gateway seat requires ID, gateway, and scheduler")
		}
		if _, exists := byID[seat.ID]; exists {
			return nil, errors.New("microagent: duplicate gateway seat ID")
		}
		byID[seat.ID] = seat.Gateway
		poolSeats = append(poolSeats, Seat{ID: seat.ID, Scheduler: seat.Scheduler})
	}
	pool, err := NewSeatPool(poolSeats)
	if err != nil {
		return nil, err
	}
	return &SessionAffinityGateway{pool: pool, byID: byID, table: table, disable: cfg.DisableAffinity, observe: cfg.Observe}, nil
}

func (g *SessionAffinityGateway) Model() string {
	if g == nil || len(g.byID) == 0 {
		return ""
	}
	// Seats may intentionally expose different account routes while serving the
	// same requested model. Return any configured model for Planner provenance;
	// each completion still carries the actual selected gateway's model.
	for _, gateway := range g.byID {
		return gateway.Model()
	}
	return ""
}

func (g *SessionAffinityGateway) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	if g == nil || g.pool == nil {
		return nil, ErrNoSeatAvailable
	}
	trace := TraceFromContext(ctx)
	key := ""
	if !g.disable {
		key = AffinityFromContext(ctx)
		if key == "" && g.table != nil && trace != "" {
			key = strings.TrimSpace(g.table.Get(trace).CacheAffinity.AffinityKey)
		}
		if key == "" {
			key = trace
		}
	}
	lease, err := g.pool.TryAcquire(key)
	if err != nil {
		return nil, err
	}
	defer lease.Release()
	if g.observe != nil {
		g.observe(AffinitySelection{Trace: trace, AffinityKey: key, SeatID: lease.SeatID})
	}
	return g.byID[lease.SeatID].Complete(ctx, messages, tools, opts...)
}

type affinityContextKey struct{}

func WithAffinity(ctx context.Context, key string) context.Context {
	return context.WithValue(ctx, affinityContextKey{}, strings.TrimSpace(key))
}

func AffinityFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	key, _ := ctx.Value(affinityContextKey{}).(string)
	return strings.TrimSpace(key)
}
