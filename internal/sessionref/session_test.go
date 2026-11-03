package sessionref

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestSessionDescriptorRef(t *testing.T) {
	t.Parallel()
	for _, id := range []string{"abc-123", "A_b.c"} {
		want := "refs/fak/locks/session-" + id
		if got := (SessionDescriptor{ID: id}).Ref(); got != want {
			t.Errorf("Ref(%q) = %q, want %q", id, got, want)
		}
	}
}

func TestSessionDescriptorExpired(t *testing.T) {
	t.Parallel()
	const updatedAt = int64(1700000000)
	for _, tc := range []struct {
		name    string
		ttl     int64
		offset  time.Duration // now = UpdatedAt + offset
		expired bool
	}{
		{name: "zero ttl never expires", ttl: 0, offset: 1000 * time.Hour},
		{name: "negative ttl never expires", ttl: -5, offset: 1000 * time.Hour},
		// The >= boundary is load-bearing: a reader at exactly the expiry instant
		// must already drop the descriptor, or a crashed node's ghost lingers one
		// extra poll cycle on every machine.
		{name: "at expiry boundary expires", ttl: 60, offset: 60 * time.Second, expired: true},
		{name: "just before boundary does not", ttl: 60, offset: 59 * time.Second},
		{name: "past boundary expires", ttl: 60, offset: 61 * time.Second, expired: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			d := SessionDescriptor{UpdatedAt: updatedAt, TTLSecs: tc.ttl}
			now := time.Unix(updatedAt, 0).Add(tc.offset)
			if got := d.Expired(now); got != tc.expired {
				t.Errorf("Expired(now=%d) = %v, want %v", now.Unix(), got, tc.expired)
			}
		})
	}
}

func TestValidSessionID(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		id    string
		valid bool
	}{
		{id: "abc-123", valid: true},
		{id: "A_b.c", valid: true},
		{id: "session", valid: true},   // the marker only collides with the "session-" prefix
		{id: "SESSION-x", valid: true}, // prefix check is case-sensitive
		{id: strings.Repeat("a", 200), valid: true},
		{id: ""},
		{id: "-x"},
		{id: ".x"},
		{id: "a/b"},
		{id: "a b"},
		{id: "session-x"},
		{id: strings.Repeat("a", 201)},
	} {
		if got := ValidSessionID(tc.id); got != tc.valid {
			label := tc.id
			if len(label) > 24 {
				label = label[:24] + "...(len=" + strconv.Itoa(len(tc.id)) + ")"
			}
			t.Errorf("ValidSessionID(%q) = %v, want %v", label, got, tc.valid)
		}
	}
}

func TestPublishSession(t *testing.T) {
	t.Parallel()
	sentinel := errors.New("sentinel put failure")

	t.Run("invalid id fails before any write", func(t *testing.T) {
		t.Parallel()
		var calls int
		ref, err := PublishSession(context.Background(), SessionDescriptor{ID: "bad/id"}, func(ctx context.Context, ref string, v any) (string, error) {
			calls++
			return "ok-ref", nil
		})
		if err == nil {
			t.Fatal("want error for invalid id")
		}
		if ref != "" {
			t.Errorf("ref = %q, want empty on error", ref)
		}
		if calls != 0 {
			t.Errorf("fake putBlobRef called %d times, want 0 (validation precedes any write)", calls)
		}
	})

	t.Run("zero UpdatedAt is restamped", func(t *testing.T) {
		t.Parallel()
		before := time.Now().Unix()
		var calls int
		var gotRef string
		var payload SessionDescriptor
		ref, err := PublishSession(context.Background(),
			SessionDescriptor{ID: "abc-123", AgentUUID: "uuid-1"},
			func(ctx context.Context, ref string, v any) (string, error) {
				calls++
				gotRef = ref
				p, ok := v.(SessionDescriptor)
				if !ok {
					t.Fatalf("payload type %T, want SessionDescriptor", v)
				}
				payload = p
				return "ok-ref", nil
			})
		if err != nil {
			t.Fatal(err)
		}
		if ref != "ok-ref" {
			t.Errorf("ref = %q, want ok-ref", ref)
		}
		if calls != 1 {
			t.Fatalf("fake putBlobRef called %d times, want 1", calls)
		}
		if want := (SessionDescriptor{ID: "abc-123"}).Ref(); gotRef != want {
			t.Errorf("put ref = %q, want %q", gotRef, want)
		}
		after := time.Now().Unix()
		if payload.UpdatedAt < before-5 || payload.UpdatedAt > after+5 {
			t.Errorf("payload UpdatedAt = %d, want within +/-5s of now (%d..%d)", payload.UpdatedAt, before, after)
		}
		if payload.AgentUUID != "uuid-1" {
			t.Errorf("payload AgentUUID = %q, want uuid-1 to round-trip", payload.AgentUUID)
		}
	})

	t.Run("preset UpdatedAt is preserved", func(t *testing.T) {
		t.Parallel()
		var payload SessionDescriptor
		_, err := PublishSession(context.Background(),
			SessionDescriptor{ID: "abc-123", UpdatedAt: 1700000000},
			func(ctx context.Context, ref string, v any) (string, error) {
				payload = v.(SessionDescriptor)
				return "ok-ref", nil
			})
		if err != nil {
			t.Fatal(err)
		}
		if payload.UpdatedAt != 1700000000 {
			t.Errorf("payload UpdatedAt = %d, want 1700000000 preserved exactly", payload.UpdatedAt)
		}
	})

	t.Run("put error is wrapped and surfaced", func(t *testing.T) {
		t.Parallel()
		_, err := PublishSession(context.Background(),
			SessionDescriptor{ID: "abc-123"},
			func(ctx context.Context, ref string, v any) (string, error) {
				return "", sentinel
			})
		if !errors.Is(err, sentinel) {
			t.Errorf("error %v does not satisfy errors.Is(_, sentinel)", err)
		}
	})
}
