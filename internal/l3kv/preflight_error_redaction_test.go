package l3kv

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// Estimate only: synthetic errors and in-memory JSON; no network requests.
func TestProbeRemoteKVURLDisplayBoundary(t *testing.T) {
	t.Parallel()
	t.Run("url_failure", func(t *testing.T) {
		const endpoint = "https://fake-user:fake-password@example.invalid/store?token=fake-query-secret&safe_flag=true"
		const token = "fake-bearer-secret"
		unavailable := errors.New("connection refused")
		for _, mode := range []RemoteKVMode{RemoteKVModeRequired, RemoteKVModeOptional} {
			for _, tc := range []struct {
				name    string
				cause   error
				wrapped bool
				outcome RemoteKVOutcome
			}{
				{"unavailable", unavailable, false, RemoteKVOutcomeUnavailable},
				{"wrapped_unavailable", unavailable, true, RemoteKVOutcomeUnavailable},
				{"deadline", context.DeadlineExceeded, false, RemoteKVOutcomeTimeout},
				{"wrapped_deadline", context.DeadlineExceeded, true, RemoteKVOutcomeTimeout},
			} {
				t.Run(string(mode)+"/"+tc.name, func(t *testing.T) {
					cfg := RemoteKVConfig{Backend: "l3kv-blobhttp", RemoteURL: endpoint, Token: token, Mode: mode}
					urlErr := &url.Error{Op: "Head", URL: endpoint, Err: tc.cause}
					var original error = urlErr
					if tc.wrapped {
						original = fmt.Errorf("probe %s failed: %w", token, urlErr)
					}
					originalText := original.Error()
					calls := 0
					receipt, err := ProbeRemoteKV(context.Background(), cfg, func(ctx context.Context, got RemoteKVConfig) error {
						calls++
						if got != cfg || ctx == nil {
							t.Fatalf("probe inputs changed: cfg=%+v ctx=%v", got, ctx)
						}
						return original
					})
					if calls != 1 || receipt.Outcome != tc.outcome || receipt.Error != "URL failure (details redacted)" {
						t.Fatalf("calls=%d receipt=%+v", calls, receipt)
					}
					if mode == RemoteKVModeRequired {
						if err == nil || !errors.Is(err, original) || !errors.Is(err, tc.cause) || receipt.FallbackReason != "" {
							t.Fatalf("required result: err=%v fallback=%q", err, receipt.FallbackReason)
						}
						var found *url.Error
						if !errors.As(err, &found) || found != urlErr {
							t.Fatal("original typed URL cause was not preserved")
						}
					} else if err != nil || receipt.FallbackReason == "" {
						t.Fatalf("optional result: err=%v fallback=%q", err, receipt.FallbackReason)
					}
					encoded, marshalErr := json.Marshal(receipt)
					if marshalErr != nil {
						t.Fatal(marshalErr)
					}
					display := string(encoded) + fmt.Sprint(err) + fmt.Sprintf("%+v", err)
					for _, secret := range []string{"fake-user", "fake-password", "fake-query-secret", token} {
						if strings.Contains(display, secret) {
							t.Fatalf("display contains synthetic credential %q", secret)
						}
					}
					if !strings.Contains(receipt.SanitizedEndpoint, "safe_flag=true") || original.Error() != originalText || urlErr.URL != endpoint {
						t.Fatal("safe endpoint metadata or original error was changed")
					}
				})
			}
		}
	})
	t.Run("malformed_url", func(t *testing.T) {
		const endpoint = "https://fake-user:fake-password@example.invalid/%zz?token=fake-query-secret"
		for _, mode := range []RemoteKVMode{RemoteKVModeRequired, RemoteKVModeOptional} {
			t.Run(string(mode), func(t *testing.T) {
				called := false
				receipt, err := ProbeRemoteKV(context.Background(), RemoteKVConfig{
					Backend: "l3kv-blobhttp", RemoteURL: endpoint, Mode: mode,
				}, func(context.Context, RemoteKVConfig) error { called = true; return nil })
				if called || err == nil || receipt.Outcome != RemoteKVOutcomeMalformedConfig || receipt.SanitizedEndpoint != "" || receipt.FallbackReason != "" || receipt.Error != err.Error() || !strings.Contains(err.Error(), "parse url") {
					t.Fatalf("called=%t err=%v receipt=%+v", called, err, receipt)
				}
				var cause *url.Error
				if !errors.As(err, &cause) || cause.URL != endpoint {
					t.Fatal("original malformed URL cause was not preserved")
				}
				encoded, marshalErr := json.Marshal(receipt)
				if marshalErr != nil {
					t.Fatal(marshalErr)
				}
				for _, secret := range []string{"fake-user", "fake-password", "fake-query-secret"} {
					if strings.Contains(string(encoded)+err.Error(), secret) {
						t.Fatalf("display contains synthetic credential %q", secret)
					}
				}
			})
		}
	})
	t.Run("preserved_controls", func(t *testing.T) {
		plain := errors.New("connection refused")
		if sanitizeProbeError(nil) != nil || sanitizeProbeError(plain) != plain {
			t.Fatal("nil or non-URL error identity changed")
		}
		for _, mode := range []RemoteKVMode{RemoteKVModeRequired, RemoteKVModeOptional, RemoteKVModeDisabled} {
			for _, probeErr := range []error{nil, plain} {
				calls := 0
				receipt, err := ProbeRemoteKV(context.Background(), RemoteKVConfig{
					Backend: "l3kv-blobhttp", RemoteURL: "https://example.invalid/store", Mode: mode,
				}, func(context.Context, RemoteKVConfig) error { calls++; return probeErr })
				switch {
				case mode == RemoteKVModeDisabled:
					if calls != 0 || err != nil || receipt.Outcome != RemoteKVOutcomeDisabled || receipt.Error != "" || receipt.FallbackReason != "" {
						t.Fatalf("disabled result: calls=%d err=%v receipt=%+v", calls, err, receipt)
					}
				case probeErr == nil:
					if calls != 1 || err != nil || receipt.Outcome != RemoteKVOutcomeReady || receipt.Error != "" || receipt.FallbackReason != "" {
						t.Fatalf("ready result: calls=%d err=%v receipt=%+v", calls, err, receipt)
					}
				default:
					if calls != 1 || receipt.Outcome != RemoteKVOutcomeUnavailable || receipt.Error != plain.Error() {
						t.Fatalf("plain failure result: calls=%d err=%v receipt=%+v", calls, err, receipt)
					}
					if mode == RemoteKVModeRequired && (err == nil || !errors.Is(err, plain) || receipt.FallbackReason != "") {
						t.Fatalf("required failure: err=%v receipt=%+v", err, receipt)
					}
					if mode == RemoteKVModeOptional && (err != nil || receipt.FallbackReason != "remote kv probe unavailable: connection refused; falling back to local residency") {
						t.Fatalf("optional failure: err=%v receipt=%+v", err, receipt)
					}
				}
			}
		}
	})
}
