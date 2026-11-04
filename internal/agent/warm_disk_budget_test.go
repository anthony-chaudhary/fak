package agent

import (
	"errors"
	"io/fs"
	"math"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3kv"
)

// warmDiskFakeClock drives admission backoff windows deterministically.
// The drain test advances time while admission reads it from another goroutine.
type warmDiskFakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *warmDiskFakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *warmDiskFakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func warmDiskLadderState(ad *warmDiskWriteAdmission) (fails int, backoff time.Duration) {
	ad.mu.Lock()
	defer ad.mu.Unlock()
	return ad.fails, ad.backoff
}

func warmDiskCapturing(p *InKernelPlanner) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.warmDiskCapture
}

// fak-test:runtime fast est=50ms lane=default
func TestWriteBudgetReserveSkipBackoff(t *testing.T) {
	const gib = int64(1) << 30

	t.Run("reserve skips unaffordable writes and backs off", func(t *testing.T) {
		clock := &warmDiskFakeClock{t: time.Unix(0, 0)}
		ad := newWarmDiskWriteAdmission(100, true, clock.now)

		if err := ad.reserve(60); err != nil {
			t.Fatalf("first reserve: %v", err)
		}
		if got := ad.pending(); got != 60 {
			t.Fatalf("pending = %d, want 60", got)
		}
		// The last 40 bytes cannot fit: the write is refused, not truncated.
		if err := ad.reserve(41); !errors.Is(err, errWarmDiskBudgetExhausted) {
			t.Fatalf("oversize reserve = %v, want budget exhausted", err)
		}
		if fails, backoff := warmDiskLadderState(ad); fails != 1 || backoff != time.Second {
			t.Fatalf("ladder after first refusal = (%d, %v), want (1, 1s)", fails, backoff)
		}
		// Refusal armed a 1 s backoff: retries inside the window refuse faster.
		clock.advance(999 * time.Millisecond)
		if err := ad.reserve(40); !errors.Is(err, errWarmDiskBackoffActive) {
			t.Fatalf("in-window retry = %v, want backoff active", err)
		}
		// After the window the retry is re-checked against the pool and fails
		// again, doubling the window to 2 s.
		clock.advance(time.Millisecond)
		if err := ad.reserve(41); !errors.Is(err, errWarmDiskBudgetExhausted) {
			t.Fatalf("post-window retry = %v, want budget exhausted", err)
		}
		// Finish releases the reservation; once the ladder cools the budget fits.
		ad.release(60)
		clock.advance(2 * time.Second)
		if err := ad.reserve(60); err != nil {
			t.Fatalf("recovered reserve: %v", err)
		}
		if got := ad.pending(); got != 60 {
			t.Fatalf("pending = %d, want 60", got)
		}
		if fails, backoff := warmDiskLadderState(ad); fails != 0 || backoff != 0 {
			t.Fatalf("success did not clear the ladder: (%d, %v)", fails, backoff)
		}
		// Success cleared the ladder: the next refusal restarts at 1 s, not the
		// pre-success 4 s rung.
		ad.release(60)
		if err := ad.reserve(101); !errors.Is(err, errWarmDiskBudgetExhausted) {
			t.Fatalf("oversize reserve = %v, want budget exhausted", err)
		}
		clock.advance(3 * time.Second / 2)
		if err := ad.reserve(1); err != nil {
			t.Fatalf("ladder restarted above 1 s: %v", err)
		}
	})

	t.Run("unknown free space refuses fail-closed", func(t *testing.T) {
		clock := &warmDiskFakeClock{}
		ad := newWarmDiskWriteAdmission(0, false, clock.now)
		if err := ad.reserve(1); !errors.Is(err, errWarmDiskSpaceUnknown) {
			t.Fatalf("unknown-space reserve = %v, want space unknown", err)
		}
		cases := []struct {
			maxBytes, free int64
			known          bool
			want           int64
		}{
			{maxBytes: 8 * gib, free: 2 * gib, known: true, want: 2 * gib},
			{maxBytes: 8 * gib, free: 64 * gib, known: true, want: 8 * gib},
			{maxBytes: 8 * gib, free: 0, known: true, want: 0},
			{maxBytes: 8 * gib, free: 5 * gib, known: false, want: 0},
			{maxBytes: 0, free: 5 * gib, known: true, want: 0},
		}
		for _, tc := range cases {
			if got := warmDiskWritePoolLimit(tc.maxBytes, tc.free, tc.known); got != tc.want {
				t.Fatalf("pool limit(max=%d, free=%d, known=%v) = %d, want %d", tc.maxBytes, tc.free, tc.known, got, tc.want)
			}
		}
	})

	t.Run("overflow and negative requests refuse without panic", func(t *testing.T) {
		clock := &warmDiskFakeClock{}
		ad := newWarmDiskWriteAdmission(100, true, clock.now)
		for _, bytes := range []int64{-1, math.MaxInt64, 101} {
			clock.advance(70 * time.Second) // cool any armed window
			if err := ad.reserve(bytes); !errors.Is(err, errWarmDiskBudgetExhausted) {
				t.Fatalf("reserve(%d) = %v, want budget exhausted", bytes, err)
			}
		}
		if got := ad.pending(); got != 0 {
			t.Fatalf("pending = %d, want 0 after refusals", got)
		}
	})

	t.Run("backoff caps at 60 s after 7 failures", func(t *testing.T) {
		clock := &warmDiskFakeClock{}
		ad := newWarmDiskWriteAdmission(100, true, clock.now)
		want := []time.Duration{
			time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second,
			16 * time.Second, 32 * time.Second, warmDiskWriteBackoffCap, warmDiskWriteBackoffCap,
		}
		for i, wantBackoff := range want {
			clock.advance(70 * time.Second) // cool any open window
			if err := ad.reserve(101); !errors.Is(err, errWarmDiskBudgetExhausted) {
				t.Fatalf("refusal %d = %v, want budget exhausted", i+1, err)
			}
			if _, backoff := warmDiskLadderState(ad); backoff != wantBackoff {
				t.Fatalf("refusal %d backoff = %v, want %v", i+1, backoff, wantBackoff)
			}
		}
		if fails, _ := warmDiskLadderState(ad); fails != warmDiskWriteBackoffMaxFail {
			t.Fatalf("fails = %d, want capped at %d", fails, warmDiskWriteBackoffMaxFail)
		}
	})

	t.Run("drain honors the shutdown deadline", func(t *testing.T) {
		clock := &warmDiskFakeClock{}
		idle := newWarmDiskWriteAdmission(1000, true, clock.now)
		if !idle.drain(clock.now()) {
			t.Fatal("empty pool must drain immediately")
		}
		ad := newWarmDiskWriteAdmission(1000, true, clock.now)
		if err := ad.reserve(100); err != nil {
			t.Fatalf("reserve: %v", err)
		}
		if ad.drain(clock.now()) {
			t.Fatal("outstanding reservation must not drain at a passed deadline")
		}
		deadline := clock.now().Add(50 * time.Millisecond)
		go func() {
			clock.advance(20 * time.Millisecond) // still inside the deadline
			ad.release(100)
		}()
		if !ad.drain(deadline) {
			t.Fatal("reservation released before the deadline must drain")
		}
	})

	t.Run("capture seam skips unaffordable budgets", func(t *testing.T) {
		oldProbe := warmDiskDiskInfo
		defer func() { warmDiskDiskInfo = oldProbe }()

		// Unknown volume: the pool stays unknown and capture is skipped.
		t.Setenv(l3kv.EnvSpec, t.TempDir())
		warmDiskDiskInfo = func(string) (int64, int64, bool) { return 0, 0, false }
		pUnknown := &InKernelPlanner{}
		pUnknown.beginWarmDiskCapture(50)
		if warmDiskCapturing(pUnknown) {
			t.Fatal("capture must stay off when free space is unknown")
		}
		pUnknown.endWarmDiskCapture()

		// Measured 100-byte pool: a 500-byte capture is unaffordable and skipped,
		// leaving nothing reserved and nothing on disk.
		rootSkip := t.TempDir()
		t.Setenv(l3kv.EnvSpec, rootSkip)
		warmDiskDiskInfo = func(string) (int64, int64, bool) { return 0, 100, true }
		pSkip := &InKernelPlanner{}
		pSkip.beginWarmDiskCapture(500)
		if warmDiskCapturing(pSkip) {
			t.Fatal("unaffordable capture must be skipped")
		}
		if got := pSkip.warmDiskWriteGate().pending(); got != 0 {
			t.Fatalf("skipped capture reserved %d bytes, want 0", got)
		}
		pSkip.endWarmDiskCapture()

		// An affordable capture reserves up front and finish releases it.
		rootOK := t.TempDir()
		t.Setenv(l3kv.EnvSpec, rootOK)
		pOK := &InKernelPlanner{}
		pOK.beginWarmDiskCapture(50)
		if !warmDiskCapturing(pOK) {
			t.Fatal("affordable capture must stay on")
		}
		gate := pOK.warmDiskWriteGate()
		if got := gate.pending(); got != 50 {
			t.Fatalf("pending = %d, want 50", got)
		}
		pOK.endWarmDiskCapture()
		if got := gate.pending(); got != 0 {
			t.Fatalf("finish did not release the reservation: pending = %d", got)
		}

		// Skipped staging leaves no publishable file behind.
		for _, root := range []string{rootSkip, rootOK} {
			files := 0
			filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
				if err == nil && d != nil && !d.IsDir() {
					files++
				}
				return nil
			})
			if files != 0 {
				t.Fatalf("skipped staging left %d files under %s", files, root)
			}
		}
	})
}
