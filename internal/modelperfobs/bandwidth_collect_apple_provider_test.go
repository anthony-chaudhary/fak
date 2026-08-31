//go:build darwin

package modelperfobs

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

type appleMemoryCurrentProviderFixture struct {
	Provider        string                      `json:"provider"`
	ProviderVersion string                      `json:"provider_version"`
	Platform        map[string]string           `json:"platform"`
	Command         []string                    `json:"command"`
	RawHelp         string                      `json:"raw_help"`
	Expected        AppleMemoryProviderEvidence `json:"expected"`
}

func TestDarwinAppleUnifiedMemoryCurrentProviderUnavailable(t *testing.T) {
	fixture := readAppleMemoryCurrentProviderFixture(t)
	now := mustAppleProviderTime(t, fixture.Expected.ObservedAt)
	run := func(_ context.Context, name string, args ...string) ([]byte, error) {
		got := append([]string{name}, args...)
		if strings.Join(got, "\x00") != strings.Join(fixture.Command, "\x00") {
			t.Fatalf("command=%q want %q", got, fixture.Command)
		}
		return []byte(fixture.RawHelp), nil
	}
	_, err := collectCurrentAppleMemoryBandwidth(context.Background(), AppleMemoryImportOptions{
		Scope: AppleMemoryScope{Kind: "system"}, Interval: 100 * time.Millisecond, ProviderVersion: fixture.ProviderVersion,
	}, run, func() time.Time { return now }, time.Second)
	var unavailable *AppleMemoryProviderUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("error=%T %v, want AppleMemoryProviderUnavailableError", err, err)
	}
	if got, want := unavailable.Evidence, fixture.Expected; !appleProviderEvidenceEqual(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", "  ")
		wantJSON, _ := json.MarshalIndent(want, "", "  ")
		t.Fatalf("evidence mismatch\ngot: %s\nwant: %s", gotJSON, wantJSON)
	}
}

func TestDarwinAppleUnifiedMemoryProviderUnavailableClasses(t *testing.T) {
	help := func(names ...string) []byte {
		var b strings.Builder
		b.WriteString("The following samplers are supported by --samplers:\n\n")
		for _, name := range names {
			b.WriteString("    " + name + "             fixture description\n")
		}
		b.WriteString("\nand the following sampler groups are supported by --samplers:\n")
		return []byte(b.String())
	}
	tests := []struct {
		name   string
		output []byte
		runErr error
		want   string
	}{
		{name: "permission denied", output: []byte("powermetrics must be invoked as the superuser"), runErr: errors.New("exit status 1"), want: AppleMemoryUnavailablePermissionDenied},
		{name: "unsupported", output: help("tasks", "gpu_power"), want: AppleMemoryUnavailableUnsupported},
		{name: "malformed", output: []byte("usage changed without sampler heading"), want: AppleMemoryUnavailableMalformed},
		{name: "ambiguous future sampler", output: help("memory_bandwidth"), want: AppleMemoryUnavailableAmbiguous},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			run := func(context.Context, string, ...string) ([]byte, error) { return tc.output, tc.runErr }
			_, err := collectCurrentAppleMemoryBandwidth(context.Background(), AppleMemoryImportOptions{}, run, func() time.Time { return time.Unix(0, 0) }, time.Second)
			var unavailable *AppleMemoryProviderUnavailableError
			if !errors.As(err, &unavailable) {
				t.Fatalf("error=%T %v", err, err)
			}
			if unavailable.Evidence.UnavailableReason != tc.want {
				t.Fatalf("reason=%q want %q: %v", unavailable.Evidence.UnavailableReason, tc.want, err)
			}
		})
	}
}

func TestDarwinAppleUnifiedMemoryProviderCancellationClasses(t *testing.T) {
	tests := []struct {
		name         string
		parent       func(*testing.T) (context.Context, context.CancelFunc)
		probeTimeout time.Duration
		wantReason   string
		wantCause    error
	}{
		{
			name: "parent cancellation",
			parent: func(t *testing.T) (context.Context, context.CancelFunc) {
				t.Helper()
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx, func() {}
			},
			probeTimeout: time.Hour,
			wantReason:   AppleMemoryUnavailableCanceled,
			wantCause:    context.Canceled,
		},
		{
			name: "earlier parent deadline",
			parent: func(t *testing.T) (context.Context, context.CancelFunc) {
				t.Helper()
				return context.WithDeadline(context.Background(), time.Unix(0, 0))
			},
			probeTimeout: time.Hour,
			wantReason:   AppleMemoryUnavailableCanceled,
			wantCause:    context.DeadlineExceeded,
		},
		{
			name: "provider timeout",
			parent: func(t *testing.T) (context.Context, context.CancelFunc) {
				t.Helper()
				return context.WithCancel(context.Background())
			},
			probeTimeout: 0,
			wantReason:   AppleMemoryUnavailableTimeout,
			wantCause:    context.DeadlineExceeded,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := tc.parent(t)
			defer cancel()
			run := func(probeCtx context.Context, name string, args ...string) ([]byte, error) {
				if name != "/usr/bin/powermetrics" || len(args) != 1 || args[0] != "--help" {
					t.Fatalf("command=%q %q, want /usr/bin/powermetrics --help", name, args)
				}
				<-probeCtx.Done()
				return nil, probeCtx.Err()
			}
			collection, err := collectCurrentAppleMemoryBandwidth(ctx, AppleMemoryImportOptions{}, run, func() time.Time {
				return time.Unix(0, 0)
			}, tc.probeTimeout)
			var unavailable *AppleMemoryProviderUnavailableError
			if !errors.As(err, &unavailable) {
				t.Fatalf("error=%T %v, want AppleMemoryProviderUnavailableError", err, err)
			}
			if unavailable.Evidence.UnavailableReason != tc.wantReason {
				t.Fatalf("reason=%q want %q: %v", unavailable.Evidence.UnavailableReason, tc.wantReason, err)
			}
			if !errors.Is(err, tc.wantCause) {
				t.Fatalf("error=%v, want cause %v", err, tc.wantCause)
			}
			if !reflect.DeepEqual(collection, BandwidthCollection{}) {
				t.Fatalf("interrupted probe emitted bandwidth collection: %+v", collection)
			}
		})
	}
}

func TestDarwinAppleUnifiedMemoryProviderUsesStrictImporter(t *testing.T) {
	valid, err := os.ReadFile("testdata/apple-memory-direct-rate.json")
	if err != nil {
		t.Fatal(err)
	}
	collection, err := importCurrentAppleProviderJSON(valid, AppleMemoryImportOptions{
		Format: AppleMemoryFormatGenericJSON, Provider: "fixture-apple-counter-export", ProviderVersion: "1.0",
		Scope: AppleMemoryScope{Kind: "package"}, Interval: 500 * time.Millisecond, Phase: PhaseDecode, Shape: ShapeSmall,
	})
	if err != nil {
		t.Fatal(err)
	}
	if collection.AppleMemoryArtifact == nil || collection.AppleMemoryArtifact.ReadBytesPerSecond == nil {
		t.Fatalf("strict importer artifact missing: %+v", collection.AppleMemoryArtifact)
	}

	for _, input := range [][]byte{
		[]byte(`<plist><dict><key>memory_bandwidth</key><integer>42</integer></dict></plist>`),
		[]byte(`{"memory_capacity_bytes": 34359738368, "gpu_utilization": 99}`),
	} {
		_, err := importCurrentAppleProviderJSON(input, AppleMemoryImportOptions{})
		var unavailable *AppleMemoryProviderUnavailableError
		if !errors.As(err, &unavailable) || unavailable.Evidence.UnavailableReason != AppleMemoryUnavailableMalformed {
			t.Fatalf("unsafe input error=%T %v", err, err)
		}
	}
}

func readAppleMemoryCurrentProviderFixture(t *testing.T) appleMemoryCurrentProviderFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/apple-memory-current-provider.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture appleMemoryCurrentProviderFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func mustAppleProviderTime(t *testing.T, value string) time.Time {
	t.Helper()
	got, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func appleProviderEvidenceEqual(a, b AppleMemoryProviderEvidence) bool {
	left, _ := json.Marshal(a)
	right, _ := json.Marshal(b)
	return string(left) == string(right)
}
