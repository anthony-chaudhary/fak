package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// This exercises the real acceptance orchestrator and existing injected command
// seam, not a replacement parser, native build, or hardware execution.
// fak-test:runtime fast est=20ms lane=default
func TestCUDAAcceptanceUsesCanonicalProfileTags(t *testing.T) {
	for _, profile := range []string{"default", "async", "flash-decode"} {
		t.Run(profile, func(t *testing.T) {
			root, _ := cudaFirstLightFixture(t)
			cfg := &computebuild.CUDAConfig{RepoRoot: root, PkgDir: filepath.Join(root, "internal", "compute"), Arch: "sm_120", ReceiptPath: ".fak/accept-profile.json"}
			switch profile {
			case "async":
				cfg.EnableV41Compressor = true
				cfg.EnableV41CompressorPending = true
				cfg.EnableQ4KQKVQualification = true
				cfg.EnableQ4KQKVAsync = true
				cfg.EnableCaptureOnceQualification = true
			case "flash-decode":
				cfg.Arch = "sm_100"
				cfg.EnableArchSpecific = true
				cfg.EnableFlashMLADecode = true
			}
			wantTags := computebuild.CUDAGoTags(cfg)
			var out, stderr bytes.Buffer
			built, tests, benches := false, 0, 0
			build := func(_ context.Context, got *computebuild.CUDAConfig) error {
				built = true
				if got != cfg || got.Command != "binary" || computebuild.CUDAGoTags(got) != wantTags {
					t.Fatal("build profile changed")
				}
				got.Toolchain = &computebuild.Toolchain{VsDevEnv: map[string]string{"FAK_ACCEPT_PROFILE_FIXTURE": "preserved"}}
				return nil
			}
			command := func(_ context.Context, name string, args []string, dir string, env []string, stdout, stderr io.Writer) error {
				if !built || dir != root {
					t.Fatal("command ran outside resolved build context")
				}
				foundEnv := false
				for _, item := range env {
					if item == "FAK_ACCEPT_PROFILE_FIXTURE=preserved" {
						foundEnv = true
					}
				}
				if !foundEnv {
					t.Fatal("resolved toolchain environment lost")
				}
				if name == cfg.OutBin {
					parity := cudaFirstLightAcceptanceFixture()
					parity["observed_tier"] = cfg.Arch
					return json.NewEncoder(stdout).Encode(parity)
				}
				if name != "go" || len(args) == 0 || args[0] != "test" {
					t.Fatalf("unexpected command %s %v", name, args)
				}
				value := func(flag string) string {
					return cudaAcceptanceFlagValue(args, flag)
				}
				if got := value("-tags"); got != wantTags {
					t.Fatalf("acceptance profile=%q built=%q args=%v", got, wantTags, args)
				}
				if value("-count") != "1" {
					t.Fatal("lost uncached acceptance")
				}
				if value("-bench") != "" {
					benches++
					if value("-run") != "^$" || value("-benchtime") != "2s" {
						t.Fatal("benchmark arguments changed")
					}
					return nil
				}
				tests++
				run := value("-run")
				if !strings.HasPrefix(run, "^(") || !strings.HasSuffix(run, ")$") {
					t.Fatal("family selector changed")
				}
				for _, test := range strings.Split(strings.TrimSuffix(strings.TrimPrefix(run, "^("), ")$"), "|") {
					if err := json.NewEncoder(stdout).Encode(map[string]string{"Action": "pass", "Test": test}); err != nil {
						return err
					}
				}
				return nil
			}
			code := runCUDABuildAcceptance(&out, &stderr, cfg, "weights", 2, true, build, command)
			if code != 0 || tests != len(cudaAcceptanceFamilies()) || benches == 0 {
				t.Fatalf("code=%d tests=%d benches=%d stderr=%s", code, tests, benches, stderr.String())
			}
		})
	}
}

// cudaAcceptanceFlagValue reads the two value forms emitted by the real
// acceptance command without matching longer flags that share a prefix.
func cudaAcceptanceFlagValue(args []string, flag string) string {
	for i, arg := range args {
		if arg == flag && i+1 < len(args) {
			return args[i+1]
		}
		if value, ok := strings.CutPrefix(arg, flag+"="); ok {
			return value
		}
	}
	return ""
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAAcceptanceFlagValue(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		flag string
		want string
	}{
		{"split-count", []string{"test", "-count", "1"}, "-count", "1"},
		{"joined-count-at-end", []string{"test", "-count=1"}, "-count", "1"},
		{"split-benchtime", []string{"-benchtime", "2s"}, "-benchtime", "2s"},
		{"joined-benchtime", []string{"-benchtime=2s"}, "-benchtime", "2s"},
		{"longer-prefix", []string{"-counter=9", "-countdown", "8", "-count=1"}, "-count", "1"},
		{"bench-not-benchtime", []string{"-benchtime=2s"}, "-bench", ""},
		{"value-with-equals", []string{"-run=^a=b$"}, "-run", "^a=b$"},
		{"missing", []string{"test"}, "-count", ""},
		{"trailing-bare", []string{"test", "-count"}, "-count", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := cudaAcceptanceFlagValue(tc.args, tc.flag); got != tc.want {
				t.Fatalf("value(%v, %q) = %q; want %q", tc.args, tc.flag, got, tc.want)
			}
		})
	}
}
