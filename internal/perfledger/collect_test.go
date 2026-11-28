// fak-test:runtime fast est=1s lane=default
package perfledger

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestSummaryRace(t *testing.T) {
	t.Run("bounded re-read catches delayed summary before touching fallback", func(t *testing.T) {
		primaryCalls := 0
		fallbackCalls := 0
		sleeps := 0

		cfg := SummaryCollectConfig{
			MaxAttempts:   4,
			RetryInterval: 10 * time.Millisecond,
			Sleep: func(ctx context.Context, d time.Duration) error {
				sleeps++
				return nil
			},
		}

		res, err := CollectTerminalSummary(
			context.Background(),
			cfg,
			func(ctx context.Context) (SummarySnapshot, error) {
				primaryCalls++
				if primaryCalls < 3 {
					// Child is already terminal, but summary has not landed yet.
					return SummarySnapshot{Terminal: true, HasSummary: false}, nil
				}
				return SummarySnapshot{
					Terminal:   true,
					HasSummary: true,
					Summary:    map[string]any{"total_requests": int64(50)},
				}, nil
			},
			func(ctx context.Context) (SummarySnapshot, error) {
				fallbackCalls++
				return SummarySnapshot{}, nil
			},
		)
		if err != nil {
			t.Fatalf("CollectTerminalSummary: %v", err)
		}
		if res.Source != SummarySourcePrimary || res.UsedFallback || res.Attempts != 3 || sleeps != 2 {
			t.Fatalf("unexpected result=%+v sleeps=%d", res, sleeps)
		}
		if fallbackCalls != 0 {
			t.Fatalf("fallback called %d times, want 0", fallbackCalls)
		}
	})

	t.Run("exhausted retries fall back to secondary source", func(t *testing.T) {
		primaryCalls := 0
		fallbackCalls := 0

		cfg := SummaryCollectConfig{
			MaxAttempts:   3,
			RetryInterval: 5 * time.Millisecond,
			Sleep:         func(ctx context.Context, d time.Duration) error { return nil },
		}

		res, err := CollectTerminalSummary(
			context.Background(),
			cfg,
			func(ctx context.Context) (SummarySnapshot, error) {
				primaryCalls++
				return SummarySnapshot{Terminal: true, HasSummary: false}, nil
			},
			func(ctx context.Context) (SummarySnapshot, error) {
				fallbackCalls++
				return SummarySnapshot{
					Terminal:   true,
					HasSummary: true,
					Summary:    map[string]any{"source": "operator-api"},
				}, nil
			},
		)
		if err != nil {
			t.Fatalf("CollectTerminalSummary fallback: %v", err)
		}
		if primaryCalls != 3 || fallbackCalls != 1 || res.Source != SummarySourceFallback || !res.UsedFallback {
			t.Fatalf("unexpected fallback result=%+v primaryCalls=%d fallbackCalls=%d", res, primaryCalls, fallbackCalls)
		}
		if res.Summary["source"] != "operator-api" {
			t.Fatalf("unexpected summary payload: %+v", res.Summary)
		}
	})

	t.Run("failure when both primary and fallback lack summary, plus env parsing and cancellation", func(t *testing.T) {
		cfg := SummaryCollectConfig{
			MaxAttempts:   2,
			RetryInterval: time.Millisecond,
			Sleep:         func(ctx context.Context, d time.Duration) error { return nil },
		}
		_, err := CollectTerminalSummary(
			context.Background(),
			cfg,
			func(ctx context.Context) (SummarySnapshot, error) {
				return SummarySnapshot{}, errors.New("primary timeout")
			},
			func(ctx context.Context) (SummarySnapshot, error) {
				return SummarySnapshot{Terminal: true, HasSummary: false}, nil
			},
		)
		if !errors.Is(err, ErrSummaryUnavailable) {
			t.Fatalf("err = %v, want ErrSummaryUnavailable", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := CollectTerminalSummary(ctx, cfg, func(ctx context.Context) (SummarySnapshot, error) {
			return SummarySnapshot{}, nil
		}, nil); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled err = %v, want context.Canceled", err)
		}

		parsed, err := SummaryCollectConfigFromLookup(func(k string) string {
			switch k {
			case "SUMMARY_RACE_MAX_ATTEMPTS":
				return "7"
			case "SUMMARY_RACE_RETRY_INTERVAL":
				return "25ms"
			}
			return ""
		})
		if err != nil || parsed.MaxAttempts != 7 || parsed.RetryInterval != 25*time.Millisecond {
			t.Fatalf("SummaryCollectConfigFromLookup = (%+v, %v)", parsed, err)
		}
		if _, err := SummaryCollectConfigFromLookup(func(k string) string { return "0" }); err == nil {
			t.Fatal("expected error for 0 max attempts")
		}
	})
}
