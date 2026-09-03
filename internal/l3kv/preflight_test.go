package l3kv

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestMalformedURLRefusal(t *testing.T) {
	cases := []struct {
		name    string
		cfg     RemoteKVConfig
		wantErr string
	}{
		{
			name: "invalid URL format",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "://bad-url",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "parse url",
		},
		{
			name: "missing scheme",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "localhost/blobs",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "unsupported scheme",
		},
		{
			name: "colon without scheme",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "127.0.0.1:8080/blobs",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "parse url",
		},
		{
			name: "unsupported ftp scheme",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "ftp://127.0.0.1:8080/blobs",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "unsupported scheme",
		},
		{
			name: "missing host",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "http:///path",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "missing host",
		},
		{
			name: "invalid port range",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "http://127.0.0.1:999999/blobs",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "invalid port",
		},
		{
			name: "unsupported backend",
			cfg: RemoteKVConfig{
				Backend:   "unsupported-s3",
				RemoteURL: "http://127.0.0.1:8080/blobs",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "unsupported backend",
		},
		{
			name: "invalid mode",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "http://127.0.0.1:8080/blobs",
				Mode:      "speculative",
			},
			wantErr: "invalid mode",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			probeCalled := false
			receipt, err := ProbeRemoteKV(context.Background(), tc.cfg, func(ctx context.Context, cfg RemoteKVConfig) error {
				probeCalled = true
				return nil
			})
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("expected error containing %q, got %q", tc.wantErr, err.Error())
			}
			if receipt.Outcome != RemoteKVOutcomeMalformedConfig {
				t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeMalformedConfig)
			}
			if receipt.Error == "" {
				t.Errorf("receipt.Error must not be empty")
			}
			if probeCalled {
				t.Errorf("probe must not be called on malformed configuration")
			}
		})
	}
}

func TestIncompleteConfigRefusal(t *testing.T) {
	cases := []struct {
		name    string
		cfg     RemoteKVConfig
		wantErr string
	}{
		{
			name: "missing URL required mode",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "remote URL is required",
		},
		{
			name: "missing URL optional mode",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "   ",
				Mode:      RemoteKVModeOptional,
			},
			wantErr: "remote URL is required",
		},
		{
			name: "missing backend",
			cfg: RemoteKVConfig{
				Backend:   "",
				RemoteURL: "http://127.0.0.1:8080/blobs",
				Mode:      RemoteKVModeRequired,
			},
			wantErr: "backend is required",
		},
		{
			name: "missing mode",
			cfg: RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "http://127.0.0.1:8080/blobs",
				Mode:      "",
			},
			wantErr: "mode is required",
		},
		{
			name:    "empty config",
			cfg:     RemoteKVConfig{},
			wantErr: "mode is required",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			probeCalled := false
			receipt, err := ProbeRemoteKV(context.Background(), tc.cfg, func(ctx context.Context, cfg RemoteKVConfig) error {
				probeCalled = true
				return nil
			})
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("expected error containing %q, got %q", tc.wantErr, err.Error())
			}
			if receipt.Outcome != RemoteKVOutcomeIncompleteConfig {
				t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeIncompleteConfig)
			}
			if receipt.Error == "" {
				t.Errorf("receipt.Error must not be empty")
			}
			if probeCalled {
				t.Errorf("probe must not be called on incomplete configuration")
			}
		})
	}
}

func TestCredentialTokenSanitization(t *testing.T) {
	token := "bearer-token-secret-999"
	password := "user-pass-secret-888"
	querySecret := "query-token-777"
	queryKey := "secret-key-666"

	cfg := RemoteKVConfig{
		Backend:   "l3kv-blobhttp",
		RemoteURL: "http://kvuser:" + password + "@127.0.0.1:8080/v1/store?token=" + querySecret + "&api_key=" + queryKey + "&safe_flag=true",
		Token:     token,
		Mode:      RemoteKVModeRequired,
	}

	receipt, err := ProbeRemoteKV(context.Background(), cfg, func(ctx context.Context, c RemoteKVConfig) error {
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify receipt fields and outcome.
	if receipt.Outcome != RemoteKVOutcomeReady {
		t.Fatalf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeReady)
	}
	if receipt.Backend != "l3kv-blobhttp" {
		t.Errorf("backend = %q, want %q", receipt.Backend, "l3kv-blobhttp")
	}

	// Verify credentials are absent from SanitizedEndpoint.
	for _, secret := range []string{token, password, querySecret, queryKey, "kvuser:"} {
		if strings.Contains(receipt.SanitizedEndpoint, secret) {
			t.Errorf("sanitized endpoint %q contains forbidden secret %q", receipt.SanitizedEndpoint, secret)
		}
	}
	if !strings.Contains(receipt.SanitizedEndpoint, "safe_flag=true") {
		t.Errorf("sanitized endpoint %q missing non-secret query param safe_flag=true", receipt.SanitizedEndpoint)
	}

	// Verify credentials are absent from ConfigDigest.
	if !strings.HasPrefix(receipt.ConfigDigest, "sha256:") {
		t.Errorf("config digest %q must start with 'sha256:'", receipt.ConfigDigest)
	}
	for _, secret := range []string{token, password, querySecret, queryKey} {
		if strings.Contains(receipt.ConfigDigest, secret) {
			t.Errorf("config digest %q contains forbidden secret %q", receipt.ConfigDigest, secret)
		}
	}
}

func TestProbeSuccessOptionalAndRequired(t *testing.T) {
	modes := []RemoteKVMode{RemoteKVModeOptional, RemoteKVModeRequired}

	for _, mode := range modes {
		t.Run(string(mode), func(t *testing.T) {
			probeRan := false
			cfg := RemoteKVConfig{
				Backend:   "l3kv-blobhttp",
				RemoteURL: "http://127.0.0.1:8080/blobs",
				Mode:      mode,
				Timeout:   50 * time.Millisecond,
			}

			receipt, err := ProbeRemoteKV(context.Background(), cfg, func(ctx context.Context, c RemoteKVConfig) error {
				probeRan = true
				return nil
			})
			if err != nil {
				t.Fatalf("expected nil error on probe success, got: %v", err)
			}
			if !probeRan {
				t.Fatalf("probe function was not called")
			}
			if receipt.Outcome != RemoteKVOutcomeReady {
				t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeReady)
			}
			if receipt.FallbackReason != "" {
				t.Errorf("expected empty FallbackReason, got %q", receipt.FallbackReason)
			}
			if receipt.Error != "" {
				t.Errorf("expected empty Error, got %q", receipt.Error)
			}
			if receipt.ProbeDurationMS < 0 {
				t.Errorf("invalid probe duration %v", receipt.ProbeDurationMS)
			}
		})
	}
}

func TestProbeTimeoutOptionalAndRequired(t *testing.T) {
	cfg := RemoteKVConfig{
		Backend:   "l3kv-blobhttp",
		RemoteURL: "http://127.0.0.1:8080/blobs",
		Timeout:   10 * time.Millisecond,
	}

	timeoutProbe := func(ctx context.Context, c RemoteKVConfig) error {
		select {
		case <-time.After(50 * time.Millisecond):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	// Optional mode: probe timeout must fail soft (nil error, FallbackReason set).
	t.Run("optional mode fails soft", func(t *testing.T) {
		optCfg := cfg
		optCfg.Mode = RemoteKVModeOptional
		receipt, err := ProbeRemoteKV(context.Background(), optCfg, timeoutProbe)
		if err != nil {
			t.Fatalf("optional mode timeout must return nil error, got: %v", err)
		}
		if receipt.Outcome != RemoteKVOutcomeTimeout {
			t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeTimeout)
		}
		if receipt.FallbackReason == "" {
			t.Errorf("optional mode timeout must set FallbackReason")
		}
		if receipt.Error == "" {
			t.Errorf("receipt.Error must record timeout failure")
		}
	})

	// Required mode: probe timeout must fail closed (non-nil error).
	t.Run("required mode fails closed", func(t *testing.T) {
		reqCfg := cfg
		reqCfg.Mode = RemoteKVModeRequired
		receipt, err := ProbeRemoteKV(context.Background(), reqCfg, timeoutProbe)
		if err == nil {
			t.Fatalf("required mode timeout must return error, got nil")
		}
		if receipt.Outcome != RemoteKVOutcomeTimeout {
			t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeTimeout)
		}
		if receipt.FallbackReason != "" {
			t.Errorf("required mode timeout should not set FallbackReason, got %q", receipt.FallbackReason)
		}
		if receipt.Error == "" {
			t.Errorf("receipt.Error must record timeout failure")
		}
	})
}

func TestProbeFailureOptionalAndRequired(t *testing.T) {
	cfg := RemoteKVConfig{
		Backend:   "l3kv-blobhttp",
		RemoteURL: "http://127.0.0.1:8080/blobs",
		Timeout:   100 * time.Millisecond,
	}

	errConnRefused := errors.New("dial tcp 127.0.0.1:8080: connect: connection refused")
	failingProbe := func(ctx context.Context, c RemoteKVConfig) error {
		return errConnRefused
	}

	// Optional mode: probe failure must fail soft (nil error, FallbackReason set).
	t.Run("optional mode fails soft", func(t *testing.T) {
		optCfg := cfg
		optCfg.Mode = RemoteKVModeOptional
		receipt, err := ProbeRemoteKV(context.Background(), optCfg, failingProbe)
		if err != nil {
			t.Fatalf("optional mode probe failure must return nil error, got: %v", err)
		}
		if receipt.Outcome != RemoteKVOutcomeUnavailable {
			t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeUnavailable)
		}
		if receipt.FallbackReason == "" {
			t.Errorf("optional mode failure must set FallbackReason")
		}
		if receipt.Error != errConnRefused.Error() {
			t.Errorf("receipt.Error = %q, want %q", receipt.Error, errConnRefused.Error())
		}
	})

	// Required mode: probe failure must fail closed (non-nil error).
	t.Run("required mode fails closed", func(t *testing.T) {
		reqCfg := cfg
		reqCfg.Mode = RemoteKVModeRequired
		receipt, err := ProbeRemoteKV(context.Background(), reqCfg, failingProbe)
		if err == nil {
			t.Fatalf("required mode failure must return error, got nil")
		}
		if receipt.Outcome != RemoteKVOutcomeUnavailable {
			t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeUnavailable)
		}
		if receipt.FallbackReason != "" {
			t.Errorf("required mode failure should not set FallbackReason, got %q", receipt.FallbackReason)
		}
		if receipt.Error != errConnRefused.Error() {
			t.Errorf("receipt.Error = %q, want %q", receipt.Error, errConnRefused.Error())
		}
	})
}

func TestDisabledMode(t *testing.T) {
	probeCalled := false
	cfg := RemoteKVConfig{
		Backend:   "l3kv-blobhttp",
		RemoteURL: "http://127.0.0.1:8080/blobs",
		Mode:      RemoteKVModeDisabled,
	}

	receipt, err := ProbeRemoteKV(context.Background(), cfg, func(ctx context.Context, c RemoteKVConfig) error {
		probeCalled = true
		return nil
	})
	if err != nil {
		t.Fatalf("disabled mode must return nil error, got: %v", err)
	}
	if probeCalled {
		t.Errorf("probe must not be invoked when mode is disabled")
	}
	if receipt.Outcome != RemoteKVOutcomeDisabled {
		t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeDisabled)
	}
	if receipt.Backend != "none" {
		t.Errorf("backend = %q, want 'none'", receipt.Backend)
	}
	if receipt.FallbackReason != "" {
		t.Errorf("expected empty FallbackReason, got %q", receipt.FallbackReason)
	}
}

func TestDefaultRemoteKVProbe(t *testing.T) {
	var gotAuthHeader string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuthHeader = r.Header.Get("Authorization")
		if r.URL.Path == "/auth-fail" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		if r.URL.Path == "/server-error" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	ctx := context.Background()

	// 1. Success with bearer token.
	cfg := RemoteKVConfig{
		Backend:   "l3kv-blobhttp",
		RemoteURL: ts.URL + "/ok",
		Token:     "test-secret-token",
		Mode:      RemoteKVModeRequired,
	}
	receipt, err := ProbeRemoteKV(ctx, cfg, nil)
	if err != nil {
		t.Fatalf("expected nil error on live server probe, got: %v", err)
	}
	if receipt.Outcome != RemoteKVOutcomeReady {
		t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeReady)
	}
	if gotAuthHeader != "Bearer test-secret-token" {
		t.Errorf("expected auth header 'Bearer test-secret-token', got %q", gotAuthHeader)
	}

	// 2. Auth failure.
	cfgAuthFail := cfg
	cfgAuthFail.RemoteURL = ts.URL + "/auth-fail"
	receipt, err = ProbeRemoteKV(ctx, cfgAuthFail, nil)
	if err == nil {
		t.Fatalf("expected error on 401 Unauthorized, got nil")
	}
	if receipt.Outcome != RemoteKVOutcomeUnavailable {
		t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeUnavailable)
	}

	// 3. Server error.
	cfgServerError := cfg
	cfgServerError.RemoteURL = ts.URL + "/server-error"
	receipt, err = ProbeRemoteKV(ctx, cfgServerError, nil)
	if err == nil {
		t.Fatalf("expected error on 500 Server Error, got nil")
	}
	if receipt.Outcome != RemoteKVOutcomeUnavailable {
		t.Errorf("outcome = %q, want %q", receipt.Outcome, RemoteKVOutcomeUnavailable)
	}
}
