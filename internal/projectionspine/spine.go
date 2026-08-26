package projectionspine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/supervisionpolicy"
)

// Snapshot is the durable identity and progress observed by a projection.
type Snapshot struct {
	AuthorityPID     int    `json:"authority_pid"`
	SessionID        string `json:"session_id"`
	WriterEpoch      uint64 `json:"writer_epoch"`
	TranscriptMarker string `json:"transcript_marker"`
	EffectCount      uint64 `json:"effect_count"`
}

// EffectReceipt proves whether an idempotent effect key executed.
type EffectReceipt struct {
	Key      string `json:"key"`
	Executed bool   `json:"executed"`
	Count    uint64 `json:"count"`
}

// Authority owns state that must survive replacement of disposable projections.
type Authority struct {
	mu       sync.Mutex
	snapshot Snapshot
	effects  map[string]EffectReceipt
}

// NewAuthority creates one durable writer. Identity fields must be non-zero.
func NewAuthority(pid int, sessionID string, writerEpoch uint64, transcriptMarker string) (*Authority, error) {
	if pid <= 0 || sessionID == "" || writerEpoch == 0 || transcriptMarker == "" {
		return nil, errors.New("projectionspine: invalid authority identity")
	}
	return &Authority{
		snapshot: Snapshot{
			AuthorityPID:     pid,
			SessionID:        sessionID,
			WriterEpoch:      writerEpoch,
			TranscriptMarker: transcriptMarker,
		},
		effects: make(map[string]EffectReceipt),
	}, nil
}

// Snapshot returns a consistent copy of the authority state.
func (a *Authority) Snapshot() Snapshot {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.snapshot
}

// ApplyEffect executes key at most once. Session and epoch fence stale writers.
func (a *Authority) ApplyEffect(sessionID string, writerEpoch uint64, key string) (EffectReceipt, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if sessionID != a.snapshot.SessionID || writerEpoch != a.snapshot.WriterEpoch {
		return EffectReceipt{}, errors.New("projectionspine: authority identity fence rejected effect")
	}
	if key == "" {
		return EffectReceipt{}, errors.New("projectionspine: empty effect key")
	}
	if prior, ok := a.effects[key]; ok {
		prior.Executed = false
		return prior, nil
	}
	a.snapshot.EffectCount++
	receipt := EffectReceipt{Key: key, Executed: true, Count: a.snapshot.EffectCount}
	a.effects[key] = receipt
	return receipt, nil
}

type request struct {
	Operation   string `json:"operation"`
	SessionID   string `json:"session_id,omitempty"`
	WriterEpoch uint64 `json:"writer_epoch,omitempty"`
	EffectKey   string `json:"effect_key,omitempty"`
}

type response struct {
	Snapshot *Snapshot      `json:"snapshot,omitempty"`
	Receipt  *EffectReceipt `json:"receipt,omitempty"`
	Error    string         `json:"error,omitempty"`
}

// Serve exposes authority state over a loopback listener. The caller owns l.
func (a *Authority) Serve(l net.Listener) error {
	for {
		conn, err := l.Accept()
		if err != nil {
			return err
		}
		go a.serveConn(conn)
	}
}

func (a *Authority) serveConn(conn net.Conn) {
	defer conn.Close()
	var req request
	if err := json.NewDecoder(conn).Decode(&req); err != nil {
		return
	}
	enc := json.NewEncoder(conn)
	switch req.Operation {
	case "attach":
		snapshot := a.Snapshot()
		if enc.Encode(response{Snapshot: &snapshot}) == nil {
			_, _ = io.Copy(io.Discard, conn)
		}
	case "effect":
		receipt, err := a.ApplyEffect(req.SessionID, req.WriterEpoch, req.EffectKey)
		if err != nil {
			_ = enc.Encode(response{Error: err.Error()})
			return
		}
		_ = enc.Encode(response{Receipt: &receipt})
	default:
		_ = enc.Encode(response{Error: "projectionspine: unknown operation"})
	}
}

// Projection is a live attachment. Closing it releases the authority connection.
type Projection struct {
	Snapshot Snapshot
	conn     net.Conn
}

// Attach observes authority state and keeps an attachment open until closed.
func Attach(ctx context.Context, address string) (*Projection, error) {
	conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", address)
	if err != nil {
		return nil, err
	}
	if deadline, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	}
	if err := json.NewEncoder(conn).Encode(request{Operation: "attach"}); err != nil {
		conn.Close()
		return nil, err
	}
	var resp response
	if err := json.NewDecoder(conn).Decode(&resp); err != nil {
		conn.Close()
		return nil, err
	}
	_ = conn.SetDeadline(time.Time{})
	if resp.Error != "" || resp.Snapshot == nil {
		conn.Close()
		return nil, fmt.Errorf("projectionspine: attach: %s", resp.Error)
	}
	return &Projection{Snapshot: *resp.Snapshot, conn: conn}, nil
}

// Wait blocks until the authority connection closes or ctx expires.
func (p *Projection) Wait(ctx context.Context) error {
	done := make(chan error, 1)
	go func() {
		_, err := io.Copy(io.Discard, p.conn)
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		_ = p.conn.Close()
		<-done
		return ctx.Err()
	}
}

// Close releases the projection attachment.
func (p *Projection) Close() error { return p.conn.Close() }

// ExecuteEffect asks the authority to execute key under the supplied identity fence.
func ExecuteEffect(ctx context.Context, address, sessionID string, writerEpoch uint64, key string) (EffectReceipt, error) {
	conn, err := (&net.Dialer{}).DialContext(ctx, "tcp", address)
	if err != nil {
		return EffectReceipt{}, err
	}
	defer conn.Close()
	if deadline, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(deadline)
	}
	req := request{Operation: "effect", SessionID: sessionID, WriterEpoch: writerEpoch, EffectKey: key}
	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return EffectReceipt{}, err
	}
	var resp response
	if err := json.NewDecoder(conn).Decode(&resp); err != nil {
		return EffectReceipt{}, err
	}
	if resp.Error != "" || resp.Receipt == nil {
		return EffectReceipt{}, fmt.Errorf("projectionspine: effect: %s", resp.Error)
	}
	return *resp.Receipt, nil
}

// DecideProjectionFailure applies the shared bounded supervision policy.
func DecideProjectionFailure(snapshot Snapshot, member string, generation uint64, failures []time.Time, now time.Time, budget supervisionpolicy.Budget) supervisionpolicy.Decision {
	return supervisionpolicy.Decide(supervisionpolicy.Request{
		Role:               supervisionpolicy.RoleProjection,
		Strategy:           supervisionpolicy.StrategyOneForOne,
		Domain:             supervisionpolicy.DomainID(snapshot.SessionID),
		Session:            supervisionpolicy.LogicalSessionID(snapshot.SessionID),
		Member:             supervisionpolicy.MemberID(member),
		Generation:         generation,
		ObservedGeneration: generation,
		WriterEpoch:        snapshot.WriterEpoch,
		EffectCertain:      true,
		Now:                now,
		Failures:           failures,
		Budget:             budget,
	})
}
