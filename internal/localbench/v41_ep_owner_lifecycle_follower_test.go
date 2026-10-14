package localbench

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41EPOwnerLifecycleFollowerRetirement(t *testing.T) {
	// Run the healthy control first in this same root. The expected-failure
	// witness cannot reach its unique marker unless every control below passes.
	healthy := v41EPRunFollowerRetirement(t, nil)
	if healthy.out.Err != nil || healthy.out.Receipt.Status != "complete" ||
		healthy.out.Receipt.CleanupCompleted == nil || !*healthy.out.Receipt.CleanupCompleted ||
		healthy.backend.recycles != 1 || !healthy.backend.retirementComplete {
		t.Fatalf("healthy follower retirement did not complete: outcome=%+v recycles=%d complete=%v", healthy.out, healthy.backend.recycles, healthy.backend.retirementComplete)
	}
	t.Log("FOLLOWER_HEALTHY_RETIREMENT_CONFIRMED")

	primary := errors.New("fixture FlushBatch interrupted Session retirement")
	failed := v41EPRunFollowerRetirement(t, primary)
	var carrier *model.EPFollowerPanicError
	if !errors.Is(failed.out.Err, primary) || !errors.As(failed.out.Err, &carrier) || carrier.PanicValue() != primary ||
		failed.out.PrimaryPanic != nil || failed.out.Receipt.Status != "incomplete" {
		t.Fatalf("qualified returned panic carrier changed: outcome=%+v carrier=%v", failed.out, carrier)
	}
	// Recycle is a real later Session.Close hook, after FlushBatch. Its absence
	// proves the public Close body was interrupted, rather than merely throwing
	// after successful retirement. No allocated device resource is involved.
	if failed.backend.recycles != 0 || failed.backend.retirementComplete || errors.Is(failed.out.Err, context.Canceled) || errors.Is(failed.out.Err, context.DeadlineExceeded) {
		t.Fatalf("failure discriminator changed: recycles=%d complete=%v unrelated cleanup=%v", failed.backend.recycles, failed.backend.retirementComplete, failed.out.Receipt.CleanupFailures)
	}
	if failed.out.Receipt.CleanupCompleted != nil && *failed.out.Receipt.CleanupCompleted {
		t.Fatal("FOLLOWER_SESSION_RETIREMENT_UNPROVEN: returned Close panic skipped Recycle but lifecycle reported CleanupCompleted=true")
	}
	if failed.out.Receipt.CleanupCompleted == nil || len(failed.out.Receipt.CleanupFailures) != 1 {
		t.Fatal("known interrupted Session retirement must be false with its explicit cleanup failure")
	}
}

type v41EPFollowerRetirementResult struct {
	out     v41EPOwnerLifecycleOutcome
	backend *v41EPFollowerRetirementBackend
}

func v41EPRunFollowerRetirement(t *testing.T, closePanic error) v41EPFollowerRetirementResult {
	t.Helper()
	f := newV41EPLifecycleFake()
	f.base = time.Now()
	f.deps.Now = time.Now
	backend := &v41EPFollowerRetirementBackend{closePanic: closePanic}
	session := &model.Session{Backend: backend}
	conn := &v41EPFollowerRetirementConn{reader: bytes.NewReader(v41EPFollowerRetirementFrames())}
	group, err := model.Join(conn, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	retired := false
	// Emergency cleanup is bounded in-memory only. Expected paths perform all
	// retirement explicitly before returning and never repeat Session.Close.
	t.Cleanup(func() {
		if retired {
			return
		}
		if backend.flushes == 0 {
			_, _ = v41EPOwnerInvoke(func() error { session.Close(); return nil })
		}
		_ = group.Close()
	})
	f.group.Value, f.group.Rank = group, 1
	f.backend.Identity.Device = f.req.Owners[1].ExpectedDevice
	f.group.StopIO = func(error) error { f.event("stop-io"); return conn.Close() }
	f.group.Close = func() error { f.event("group"); return group.Close() }
	f.loaded.Value = session
	runnerCloses, factories, claims, observations := 0, 0, 0, 0
	var followerResult model.EPFollowerResult
	f.loaded.FollowerRetirement = func() (model.EPFollowerRetirement, error) {
		observations++
		return followerResult.Retirement, nil
	}
	f.loaded.CloseSession = func() error {
		runnerCloses++
		session.Close()
		return nil
	}
	f.deps.Follower = func(_ context.Context, loaded *v41EPOwnerLifecycleLoaded, claim func() error) error {
		f.event("follower")
		followerResult = model.RunEPFollowerWithResult(group, func() *model.Session {
			factories++
			claims++
			if err := claim(); err != nil {
				panic(err)
			}
			return loaded.Value.(*model.Session)
		})
		return followerResult.Err
	}
	out := runV41EPOwnerLifecycle(context.Background(), f.req, 1, f.deps)
	if factories != 1 || claims != 1 || observations != 1 || runnerCloses != 0 || backend.flushes != 1 || backend.ownerCalls != 4 {
		t.Fatalf("follower transfer or public Close count changed: factories=%d claims=%d runnerCloses=%d flushes=%d ownerCalls=%d", factories, claims, runnerCloses, backend.flushes, backend.ownerCalls)
	}
	if out.Receipt.ExecutionJoined == nil || !*out.Receipt.ExecutionJoined ||
		f.count("model") != 1 || f.count("group") != 1 || f.count("backend") != 1 || conn.closes != 1 || conn.reader.Len() != 0 || conn.written != 14 {
		t.Fatalf("joined loop or independent releases changed: outcome=%+v events=%v conn=%+v", out, f.events(), conn)
	}
	if !backend.owner.TryLock() {
		t.Fatal("Session execution owner remained locked after follower return")
	}
	backend.owner.Unlock()
	retired = true
	return v41EPFollowerRetirementResult{out: out, backend: backend}
}

// No math methods are supplied. Empty Prefill acquires/releases the existing
// execution owner; Session.Close then invokes the actual batch/recycle hooks.
type v41EPFollowerRetirementBackend struct {
	compute.Backend
	owner              sync.Mutex
	ownerCalls         int
	flushes, recycles  int
	retirementComplete bool
	closePanic         error
}

func (b *v41EPFollowerRetirementBackend) ForwardExecutionOwner() *sync.Mutex {
	b.ownerCalls++
	return &b.owner
}
func (*v41EPFollowerRetirementBackend) BeginBatch() { panic("unexpected model compute") }
func (b *v41EPFollowerRetirementBackend) FlushBatch() {
	b.flushes++
	if b.closePanic != nil {
		panic(b.closePanic)
	}
}
func (b *v41EPFollowerRetirementBackend) Recycle() {
	b.recycles++
	b.retirementComplete = true
}

// Two independently encoded bounded replies: empty PREFILL, then SHUTDOWN.
// Each is a four-byte length + successful status + nine-byte decode control.
func v41EPFollowerRetirementFrames() []byte {
	var wire [28]byte
	for i, op := range []byte{1, 3} {
		frame := wire[i*14 : (i+1)*14]
		binary.LittleEndian.PutUint32(frame[:4], 10)
		frame[5] = op
	}
	return wire[:]
}

type v41EPFollowerRetirementConn struct {
	reader  *bytes.Reader
	once    sync.Once
	closes  int
	written int
}

func (c *v41EPFollowerRetirementConn) Read(p []byte) (int, error) { return c.reader.Read(p) }
func (c *v41EPFollowerRetirementConn) Write(p []byte) (int, error) {
	if len(p) > 14-c.written {
		return 0, io.ErrShortWrite
	}
	c.written += len(p)
	return len(p), nil
}
func (c *v41EPFollowerRetirementConn) Close() error {
	c.once.Do(func() { c.closes++ })
	return nil
}
func (*v41EPFollowerRetirementConn) LocalAddr() net.Addr              { return nil }
func (*v41EPFollowerRetirementConn) RemoteAddr() net.Addr             { return nil }
func (*v41EPFollowerRetirementConn) SetDeadline(time.Time) error      { return nil }
func (*v41EPFollowerRetirementConn) SetReadDeadline(time.Time) error  { return nil }
func (*v41EPFollowerRetirementConn) SetWriteDeadline(time.Time) error { return nil }
