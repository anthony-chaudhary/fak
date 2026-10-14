package ggufload

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime medium est=1s lane=default
func TestV41ExpertParallelSessionRetirementEvidence(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "0")
	t.Setenv("FAK_PAGED_KV", "0")
	shards, digest := ds41AuthenticatedEPShards(t)
	if len(shards) != 2 {
		t.Fatalf("bounded fixture changed: %d shards", len(shards))
	}
	retirementFailure := errors.New("retained reader retirement sentinel")
	for _, injected := range []error{nil, retirementFailure} {
		conn := &ds41RetirementConn{}
		group, err := model.Join(conn, 1, 3)
		if err != nil {
			t.Fatal(err)
		}
		defer group.Close()
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		// Only capability preflight is reached. This recorder has no model
		// tensors, and cancellation prevents materialization or native calls.
		backend := newDS41EPRecorder(&model.Model{})
		request := V41ExpertParallelSessionRequest{
			Shards: shards, ExpectedShardSetSHA256: digest,
			SourceSHA256: strings.Repeat("a", 64), Group: group, Backend: backend,
		}
		var readers []*ds41RetirementReader
		opened := 0
		session, coordinator, cause := loadV41ExpertParallelSession(ctx, request,
			func(ctx context.Context, manifest []ShardIdentity, set string) (*WeightSource, error) {
				opened++
				source, err := OpenWeightsAuthenticated(ctx, manifest, set)
				if err != nil {
					return nil, err
				}
				for i, closer := range source.closers {
					reader := &ds41RetirementReader{inner: closer}
					if i == 0 {
						reader.injected = injected
					}
					source.closers[i] = reader
					readers = append(readers, reader)
				}
				cancel()
				return source, nil
			})
		cancel()
		if err := group.Close(); err != nil {
			t.Fatal(err)
		}
		if session != nil || coordinator != nil || !errors.Is(cause, context.Canceled) || opened != 1 || len(readers) != len(shards) {
			t.Fatalf("post-authentication cancellation changed: session=%v coordinator=%v error=%v opens=%d readers=%d",
				session, coordinator, cause, opened, len(readers))
		}
		for i, reader := range readers {
			if reader.attempts != 1 || reader.completed != 1 || reader.innerErr != nil {
				t.Fatalf("reader %d: attempts=%d completions=%d actual close error=%v", i, reader.attempts, reader.completed, reader.innerErr)
			}
		}
		if conn.reads != 0 || conn.writes != 1 || !bytes.Equal(conn.announcement, []byte{1, 0, 0, 0}) || conn.closes != 1 {
			t.Fatalf("unexpected group traffic or ownership: %+v", conn)
		}
		if len(backend.live) != 0 || len(backend.order) != 0 || backend.routed != 0 {
			t.Fatal("canceled loader performed materialization or forward work")
		}
		if injected == nil {
			t.Log("LOADER_HEALTHY_CANCELED_RETIREMENT_CONFIRMED")
			continue
		}
		// The wrapper closed the actual reader before returning its error.
		// This is lost retirement-error evidence, not an OS-descriptor leak.
		if !errors.Is(cause, retirementFailure) {
			t.Fatalf("LOADER_RETIREMENT_ERROR_EVIDENCE_LOST: cancellation returned without the retained-reader Close error: %v", cause)
		}
	}
}

type ds41RetirementReader struct {
	inner               io.Closer
	injected, innerErr  error
	attempts, completed int
	panicOnClose        bool
	panicValue          any
}

func (r *ds41RetirementReader) Close() error {
	r.attempts++
	r.innerErr = r.inner.Close()
	r.completed++
	if r.panicOnClose {
		panic(r.panicValue)
	}
	return errors.Join(r.innerErr, r.injected)
}

// fak-test:runtime medium est=2s lane=default
func TestV41ExpertParallelSessionLoadResult(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "0")
	t.Setenv("FAK_PAGED_KV", "0")
	t.Setenv("FAK_NUMA_INTERLEAVE", "off")
	t.Setenv("FAK_NUMA_REPLICAS", "off")
	shards, digest := ds41AuthenticatedEPShards(t)
	primary := errors.New("construction primary")
	closeError := errors.New("observed outer close error")
	closePanic := errors.New("observed outer close panic")
	t.Run("preflight-no-owned-work", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		got := LoadV41ExpertParallelSessionWithResult(ctx, V41ExpertParallelSessionRequest{})
		if got.Err != context.Canceled || got.Panicked || got.OwnershipTransferred || got.Session != nil || got.Coordinator != nil ||
			!got.Retirement.Available || !got.Retirement.Completed || got.Retirement.Err != nil {
			t.Fatalf("preflight result: %+v", got)
		}
	})
	for _, tc := range []struct {
		name, fault        string
		closeErr           bool
		closePanic         bool
		known, complete    bool
		panicked, transfer bool
	}{
		{name: "clean-post-auth-cancel", fault: "cancel", known: true, complete: true},
		{name: "post-auth-close-error", fault: "cancel", closeErr: true, known: true},
		{name: "primary-error-and-cleanup-panic", fault: "cancel", closePanic: true, known: true},
		{name: "primary-panic-and-cleanup-panic", fault: "panic", closePanic: true, known: true, panicked: true},
		{name: "partial-open-error-stays-unknown", fault: "open-error"},
		{name: "known-close-error-overrides-unknown", fault: "open-error", closeErr: true, known: true},
		{name: "nil-open-result-stays-unknown", fault: "nil-open"},
		{name: "materializer-cancel-stays-unknown", fault: "materializer-cancel"},
		{name: "Session-constructor-panic-stays-unknown", fault: "Session-panic", panicked: true},
		{name: "post-Session-cancel-model-close-error", fault: "Session-cancel", closeErr: true, known: true},
		{name: "transfer-is-not-retirement", transfer: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rawCtx, cancel := context.WithCancel(t.Context())
			defer cancel()
			armed, afterOpenChecks := false, 0
			ctx := ds41RetirementContext{Context: rawCtx, check: func() error {
				if armed {
					if tc.fault == "panic" {
						panic(primary)
					}
					if tc.fault == "materializer-cancel" {
						afterOpenChecks++ // Cancels before materializer workers start.
						if afterOpenChecks == 2 {
							cancel()
						}
					}
				}
				return rawCtx.Err()
			}}
			var backend compute.Backend = newDS41EPRecorder(&model.Model{})
			var factoryBackend *ds41FactoryBackend
			if tc.fault == "Session-panic" || tc.fault == "Session-cancel" {
				factoryBackend = &ds41FactoryBackend{ds41EPRecorder: newDS41EPRecorder(&model.Model{}), reset: func() {
					if tc.fault == "Session-panic" {
						panic(primary)
					}
					cancel()
				}}
				backend = factoryBackend
			}
			req := ds41RetirementRequest(t, shards, digest, backend)
			var readers []*ds41RetirementReader
			got := loadV41ExpertParallelSessionWithResult(ctx, req, func(ctx context.Context, manifest []ShardIdentity, set string) (*WeightSource, error) {
				if tc.fault == "nil-open" {
					return nil, nil
				}
				ws, err := OpenWeightsAuthenticated(ctx, manifest, set)
				if err != nil {
					return nil, err
				}
				for i, closer := range ws.closers {
					reader := &ds41RetirementReader{inner: closer}
					if tc.closeErr && i == 0 {
						reader.injected = closeError
					}
					// The last closer panics only after closing its actual reader,
					// so this control leaves no unretired fixture descriptor.
					if tc.closePanic && i == len(ws.closers)-1 {
						reader.panicOnClose, reader.panicValue = true, closePanic
					}
					ws.closers[i] = reader
					readers = append(readers, reader)
					t.Cleanup(func() {
						if reader.attempts == 0 {
							_ = reader.inner.Close()
						}
					})
				}
				armed = true
				if tc.fault == "cancel" {
					cancel()
				}
				if tc.fault == "open-error" {
					return ws, primary
				}
				return ws, nil
			})
			ownedSession := got.Session
			var ownedModel *model.Model
			if ownedSession != nil {
				ownedModel = ownedSession.M
			}
			defer func() {
				if ownedModel != nil {
					_ = ownedModel.CloseWeights()
				}
			}()
			defer func() {
				if ownedSession != nil {
					ownedSession.Close()
				}
			}()
			if got.Panicked != tc.panicked || got.OwnershipTransferred != tc.transfer || got.Coordinator != nil ||
				got.Retirement.Available != tc.known || got.Retirement.Completed != tc.complete {
				t.Fatalf("result state: %+v", got)
			}
			if tc.transfer {
				if got.Err != nil || got.Session == nil || got.Retirement.Err != nil {
					t.Fatalf("successful transfer: %+v", got)
				}
				session := ownedSession
				ownedSession = nil
				session.Close()
				m := ownedModel
				ownedModel = nil
				if err := m.CloseWeights(); err != nil {
					t.Fatal(err)
				}
			} else {
				if got.Session != nil || got.Err == nil {
					t.Fatalf("failed construction transferred ownership: %+v", got)
				}
				wantPrimary := error(context.Canceled)
				if tc.panicked || tc.fault == "open-error" {
					wantPrimary = primary
				}
				if tc.fault != "nil-open" && !errors.Is(got.Err, wantPrimary) {
					t.Fatalf("primary disappeared: %v", got.Err)
				}
				if !tc.closeErr && !tc.closePanic && !tc.panicked && tc.fault != "nil-open" && got.Err != wantPrimary {
					t.Fatal("clean cleanup changed returned error identity")
				}
				if (tc.closeErr || tc.closePanic) && !tc.panicked {
					joined, ok := got.Err.(interface{ Unwrap() []error })
					if !ok || len(joined.Unwrap()) != 2 || joined.Unwrap()[0] != wantPrimary {
						t.Fatal("ordinary primary no longer precedes cleanup failures")
					}
				}
				if tc.panicked {
					var carrier *V41ExpertParallelLoadPanicError
					if got.PrimaryPanic != primary || !errors.As(got.Err, &carrier) || carrier.PanicValue() != primary || carrier.Operation != "construction" {
						t.Fatal("primary panic value/order changed")
					}
				}
			}
			if tc.closeErr && (!errors.Is(got.Err, closeError) || !errors.Is(got.Retirement.Err, closeError)) {
				t.Fatal("observed Close error is missing")
			}
			if tc.closePanic {
				var carrier *V41ExpertParallelLoadPanicError
				if !errors.Is(got.Err, closePanic) || !errors.As(got.Retirement.Err, &carrier) || carrier.PanicValue() != closePanic || carrier.Operation != "WeightSource.Close" {
					t.Fatal("cleanup panic is not independently inspectable")
				}
			}
			if !tc.closeErr && !tc.closePanic && got.Retirement.Err != nil {
				t.Fatal("construction failure mislabeled as a release failure")
			}
			if errors.Is(got.Retirement.Err, primary) || errors.Is(got.Retirement.Err, context.Canceled) {
				t.Fatal("retirement evidence contains the construction failure")
			}
			if tc.fault != "nil-open" && len(readers) != len(shards) {
				t.Fatal("authentication did not retain every fixture reader")
			}
			for i, reader := range readers {
				if reader.attempts != 1 || reader.completed != 1 || reader.innerErr != nil {
					t.Fatalf("reader %d retirement: %+v", i, reader)
				}
			}
			if factoryBackend != nil && factoryBackend.kvFrees != 1 {
				t.Fatalf("actual Session constructor/retirement KV frees=%d", factoryBackend.kvFrees)
			}
		})
	}
	t.Run("compatibility-panic-payload-including-nil", func(t *testing.T) {
		// Test the legacy runtime mode explicitly as well as ordinary payloads.
		// Presence is observed via completion state, never recover()!=nil.
		t.Setenv("GODEBUG", "panicnil=1")
		for _, payload := range []any{primary, "string panic", (*int)(new(int)), nil} {
			req := ds41RetirementRequest(t, shards, digest, newDS41EPRecorder(&model.Model{}))
			opener := func(context.Context, []ShardIdentity, string) (*WeightSource, error) { panic(payload) }
			got := loadV41ExpertParallelSessionWithResult(t.Context(), req, opener)
			var carrier *V41ExpertParallelLoadPanicError
			if !got.Panicked || got.PrimaryPanic != payload || !errors.As(got.Err, &carrier) || carrier.PanicValue() != payload || got.Retirement.Available || got.OwnershipTransferred {
				t.Fatalf("captured panic changed: %+v", got)
			}
			_, legacyValue, legacyPanicked := v41EPLoadInvoke("compatibility witness", func() error {
				_, _, err := loadV41ExpertParallelSession(t.Context(), req, opener)
				return err
			})
			if !legacyPanicked || legacyValue != payload {
				t.Fatalf("compatibility panic presence=%t value=%v want=%v", legacyPanicked, legacyValue, payload)
			}
		}
		_, cleanupValue, cleanupPanicked := v41EPLoadInvoke("nil cleanup witness", func() error { panic(nil) })
		if !cleanupPanicked || cleanupValue != nil {
			t.Fatal("cleanup capture lost nil panic presence")
		}
	})
}

type ds41RetirementContext struct {
	context.Context
	check func() error
}

func (c ds41RetirementContext) Err() error { return c.check() }

func ds41RetirementRequest(t *testing.T, shards []ShardIdentity, digest string, backend compute.Backend) V41ExpertParallelSessionRequest {
	t.Helper()
	conn := &ds41RetirementConn{}
	group, err := model.Join(conn, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := group.Close(); err != nil {
			t.Error(err)
		}
		if conn.reads != 0 || conn.writes != 1 || !bytes.Equal(conn.announcement, []byte{1, 0, 0, 0}) || conn.closes != 1 {
			t.Errorf("unexpected group traffic/retirement: %+v", conn)
		}
	})
	return V41ExpertParallelSessionRequest{Shards: shards, ExpectedShardSetSHA256: digest,
		SourceSHA256: strings.Repeat("a", 64), Group: group, Backend: backend}
}

// Join only needs a four-byte announcement for this non-forwarding witness.
// No network endpoint or asynchronous user is created by this connection.
type ds41RetirementConn struct {
	announcement          []byte
	reads, writes, closes int
}

func (c *ds41RetirementConn) Read([]byte) (int, error) {
	c.reads++
	return 0, errors.New("retirement witness unexpected read")
}
func (c *ds41RetirementConn) Write(p []byte) (int, error) {
	c.writes++
	if c.writes != 1 || !bytes.Equal(p, []byte{1, 0, 0, 0}) {
		return 0, errors.New("retirement witness unexpected announcement")
	}
	c.announcement = append([]byte(nil), p...)
	return len(p), nil
}
func (c *ds41RetirementConn) Close() error                   { c.closes++; return nil }
func (*ds41RetirementConn) LocalAddr() net.Addr              { return &net.TCPAddr{} }
func (*ds41RetirementConn) RemoteAddr() net.Addr             { return &net.TCPAddr{} }
func (*ds41RetirementConn) SetDeadline(time.Time) error      { return nil }
func (*ds41RetirementConn) SetReadDeadline(time.Time) error  { return nil }
func (*ds41RetirementConn) SetWriteDeadline(time.Time) error { return nil }
