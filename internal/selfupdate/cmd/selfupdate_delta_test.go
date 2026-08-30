package selfupdatecmd

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAcquireSelfUpdateArtifactUsesExactSourceZstdDelta(t *testing.T) {
	source := bytes.Repeat([]byte("old"), 24)
	targetBody := bytes.Repeat([]byte("new-release"), 16)
	patch := []byte("small-zstd-patch")
	server := selfUpdateArtifactServer(t, targetBody, patch)
	installed := filepath.Join(t.TempDir(), "fak")
	if err := os.WriteFile(installed, source, 0o755); err != nil {
		t.Fatal(err)
	}
	target := deltaFixtureTarget(server+"/full", server+"/delta", source, targetBody, patch)
	withSelfUpdateTransportClock(t)
	path, receipt, err := acquireSelfUpdateArtifact(context.Background(), zstdFixtureRunner(targetBody, true, true), installed, target, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(path)
	got, _ := os.ReadFile(path)
	if !bytes.Equal(got, targetBody) {
		t.Fatalf("patched artifact = %q", got)
	}
	if receipt.ChosenPath != "delta" || receipt.DeltaBytes != int64(len(patch)) || receipt.FullBytes != 0 ||
		receipt.Verification != "signed_target_size_sha256_verified" || receipt.FallbackReason != "" ||
		receipt.TotalMS <= 0 || receipt.FallbackBytes != 0 || receipt.FallbackMS != 0 {
		t.Fatalf("delta receipt = %+v", receipt)
	}
}

func TestAcquireSelfUpdateArtifactFallsBackToSignedFull(t *testing.T) {
	source := bytes.Repeat([]byte("old"), 24)
	targetBody := bytes.Repeat([]byte("new-release"), 16)
	goodPatch := []byte("small-zstd-patch")
	largePatch := bytes.Repeat([]byte("p"), (len(targetBody)*4+4)/5)

	tests := []struct {
		name       string
		patch      []byte
		edit       func(*selfUpdateArtifactTarget)
		zstdFound  bool
		patchOK    bool
		wantReason string
		wantDelta  int64
	}{
		{
			name: "wrong source", patch: goodPatch, zstdFound: true, patchOK: true,
			edit: func(target *selfUpdateArtifactTarget) {
				target.Deltas[0].SourceSHA256 = strings.Repeat("0", 64)
			},
			wantReason: "source_digest_mismatch",
		},
		{
			name: "corrupt patch", patch: goodPatch, zstdFound: true, patchOK: true,
			edit: func(target *selfUpdateArtifactTarget) {
				target.Deltas[0].SHA256 = strings.Repeat("0", 64)
			},
			wantReason: "delta_download_verification_failed", wantDelta: int64(len(goodPatch)),
		},
		{
			name: "zstd unavailable", patch: goodPatch, zstdFound: false, patchOK: false,
			edit:       func(*selfUpdateArtifactTarget) {},
			wantReason: "zstd_unavailable", wantDelta: int64(len(goodPatch)),
		},
		{
			name: "patch failure", patch: goodPatch, zstdFound: true, patchOK: false,
			edit:       func(*selfUpdateArtifactTarget) {},
			wantReason: "zstd_patch_failed", wantDelta: int64(len(goodPatch)),
		},
		{
			name: "poor ratio", patch: largePatch, zstdFound: true, patchOK: true,
			edit:       func(*selfUpdateArtifactTarget) {},
			wantReason: "poor_delta_ratio",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := selfUpdateArtifactServer(t, targetBody, tc.patch)
			installed := filepath.Join(t.TempDir(), "fak")
			if err := os.WriteFile(installed, source, 0o755); err != nil {
				t.Fatal(err)
			}
			target := deltaFixtureTarget(server+"/full", server+"/delta", source, targetBody, tc.patch)
			tc.edit(&target)
			withSelfUpdateTransportClock(t)
			path, receipt, err := acquireSelfUpdateArtifact(context.Background(), zstdFixtureRunner(targetBody, tc.zstdFound, tc.patchOK), installed, target, t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer os.Remove(path)
			got, _ := os.ReadFile(path)
			if !bytes.Equal(got, targetBody) {
				t.Fatalf("fallback artifact = %q", got)
			}
			if receipt.ChosenPath != "full" || receipt.FallbackReason != tc.wantReason ||
				receipt.DeltaBytes != tc.wantDelta || receipt.FullBytes != int64(len(targetBody)) ||
				receipt.FallbackBytes != int64(len(targetBody)) || receipt.FallbackMS <= 0 ||
				receipt.TotalMS <= receipt.FallbackMS || receipt.Verification != "signed_full_size_sha256_verified" {
				t.Fatalf("fallback receipt = %+v", receipt)
			}
		})
	}
}

func deltaFixtureTarget(fullURL, deltaURL string, source, target, patch []byte) selfUpdateArtifactTarget {
	return selfUpdateArtifactTarget{
		URL: fullURL, SHA256: digestManifestFixture(target), Size: int64(len(target)),
		Deltas: []selfUpdateArtifactDelta{{
			URL: deltaURL, Format: selfUpdateDeltaFormat, SourceSHA256: digestManifestFixture(source),
			SHA256: digestManifestFixture(patch), Size: int64(len(patch)),
		}},
	}
}

func selfUpdateArtifactServer(t *testing.T, full, delta []byte) string {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/full":
			_, _ = w.Write(full)
		case "/delta":
			_, _ = w.Write(delta)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server.URL
}

func zstdFixtureRunner(target []byte, found, patchOK bool) func(context.Context, string, string, ...string) (string, bool) {
	return func(_ context.Context, _ string, name string, args ...string) (string, bool) {
		if name != "zstd" {
			return "unexpected command", false
		}
		if len(args) == 1 && args[0] == "--version" {
			return "zstd fixture", found
		}
		if !patchOK {
			return "fixture patch failure", false
		}
		for i := range args {
			if args[i] == "-o" && i+1 < len(args) {
				if err := os.WriteFile(args[i+1], target, 0o600); err != nil {
					return err.Error(), false
				}
				return "", true
			}
		}
		return "missing output", false
	}
}

func withSelfUpdateTransportClock(t *testing.T) {
	t.Helper()
	old := selfUpdateTransportNow
	now := time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC)
	selfUpdateTransportNow = func() time.Time {
		now = now.Add(10 * time.Millisecond)
		return now
	}
	t.Cleanup(func() { selfUpdateTransportNow = old })
}
