package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/rawdecode"
)

func setRawDecodeShardFlags(t *testing.T, manifest, digest, single string) {
	t.Helper()
	for _, input := range []struct{ name, value string }{
		{"raw-artifact-shards", manifest},
		{"raw-artifact-shard-set-sha256", digest},
		{"raw-artifact-sha256", single},
	} {
		registered := flag.Lookup(input.name)
		if registered == nil {
			t.Fatalf("missing CLI flag -%s", input.name)
		}
		previous := registered.Value.String()
		if err := flag.Set(input.name, input.value); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := flag.Set(input.name, previous); err != nil {
				t.Error(err)
			}
		})
	}
}

func writeRawDecodeShardManifest(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "shards.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func rawDecodeShardFixture(t *testing.T) (rawdecode.Request, []byte) {
	t.Helper()
	req := rawDecodeOutputTextFixture(t, []string{"a", "b", "ab"}, []string{"a b"})
	info, err := os.Stat(req.ArtifactPath)
	if err != nil {
		t.Fatal(err)
	}
	req.ArtifactShards = []ggufload.ShardIdentity{{
		Path: req.ArtifactPath, Size: info.Size(), RawFileSHA256: req.ExpectedArtifactSHA256,
	}}
	req.ExpectedArtifactSHA256 = ""
	data, err := json.Marshal(req.ArtifactShards)
	if err != nil {
		t.Fatal(err)
	}
	req.ExpectedArtifactShardSetSHA256 = fmt.Sprintf("%x", sha256.Sum256(data))
	return req, data
}

// fak-test:runtime fast est=100ms lane=default
func TestRawDecodeShardAdapterForwardsIdentityAndProjectsSealedObservation(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "0")
	t.Setenv("FAK_PAGED_KV", "0")
	defer setRawDecodeTestFlags(true, "0", 2, true, false)()
	observedRequest, _ := rawDecodeShardFixture(t)
	requestShards := append([]ggufload.ShardIdentity(nil), observedRequest.ArtifactShards...)
	requestShards = append(requestShards, ggufload.ShardIdentity{
		Path: observedRequest.ArtifactPath + ".second", Size: 123, RawFileSHA256: strings.Repeat("b", 64),
	})
	data, err := json.Marshal(requestShards)
	if err != nil {
		t.Fatal(err)
	}
	trustedDigest := strings.Repeat("a", 64)
	setRawDecodeShardFlags(t, writeRawDecodeShardManifest(t, data), trustedDigest, "")
	f := testRawDecodeFlags(1, 2)
	*f.gguf = requestShards[0].Path
	*f.out = filepath.Join(t.TempDir(), "raw.json")
	calls := 0
	var observed rawdecode.Execution
	execute := func(ctx context.Context, req rawdecode.Request) (rawdecode.Execution, error) {
		calls++
		if req.ArtifactPath != requestShards[0].Path || req.ExpectedArtifactSHA256 != "" ||
			req.ExpectedArtifactShardSetSHA256 != trustedDigest || !reflect.DeepEqual(req.ArtifactShards, requestShards) {
			t.Fatal("adapter changed the independently supplied shard identity")
		}
		var executeErr error
		observed, executeErr = rawdecode.Execute(ctx, observedRequest)
		if executeErr == nil {
			if _, ok := observed.GenerationObservation(0); !ok {
				t.Fatal("fixture execution lacks generation authority")
			}
		}
		return observed, executeErr
	}
	if err := runRawDecodeArtifactWith(f, nil, execute); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("executor calls=%d, want 1", calls)
	}
	data, err = os.ReadFile(*f.out)
	if err != nil {
		t.Fatal(err)
	}
	var report struct {
		Shards   []ggufload.ShardIdentity `json:"artifact_shards"`
		SetSHA   string                   `json:"artifact_shard_set_sha256"`
		SHA      string                   `json:"artifact_sha256"`
		Physical struct {
			Status string `json:"status"`
			Credit bool   `json:"credit_eligible"`
		} `json:"canonical_physical_receipt"`
	}
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(report.Shards, observed.ArtifactShards) ||
		report.SetSHA != observed.ArtifactShardSetSHA256 || report.SHA != "" {
		t.Fatal("ordinary report lost or replaced executor-authenticated shard identity")
	}
	if report.Physical.Status != "UNAVAILABLE" || report.Physical.Credit {
		t.Fatal("CPU fixture was promoted to physical evidence")
	}
	observed.ArtifactShards = append([]ggufload.ShardIdentity(nil), observed.ArtifactShards...)
	observed.ArtifactShards[0].Size++
	forged, err := rawDecodeReport(f, observed, nil, observed.ModelName)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"artifact_shards", "artifact_shard_set_sha256"} {
		if _, present := forged[key]; present {
			t.Fatalf("unsealed execution projected %s", key)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRawDecodeShardRejectsInvalidManifestBeforeExecutor(t *testing.T) {
	defer setRawDecodeTestFlags(true, "0", 2, true, false)()
	shards := []ggufload.ShardIdentity{{Path: "first.gguf", Size: 1, RawFileSHA256: strings.Repeat("b", 64)}}
	valid, err := json.Marshal(shards)
	if err != nil {
		t.Fatal(err)
	}
	unknown := strings.TrimSuffix(string(valid), "}]") + ",\"unexpected\":true}]"
	for _, tc := range []struct {
		name, body, digest, single, gguf string
		noManifest                       bool
	}{
		{name: "manifest without digest", body: string(valid), gguf: "first.gguf"},
		{name: "digest without manifest", digest: strings.Repeat("a", 64), gguf: "first.gguf", noManifest: true},
		{name: "empty array", body: "[]", digest: strings.Repeat("a", 64), gguf: "first.gguf"},
		{name: "null", body: "null", digest: strings.Repeat("a", 64), gguf: "first.gguf"},
		{name: "single hash conflict", body: string(valid), digest: strings.Repeat("a", 64), single: strings.Repeat("b", 64), gguf: "first.gguf"},
		{name: "first path mismatch", body: string(valid), digest: strings.Repeat("a", 64), gguf: "second.gguf"},
		{name: "unknown member", body: unknown, digest: strings.Repeat("a", 64), gguf: "first.gguf"},
		{name: "trailing JSON", body: string(valid) + " []", digest: strings.Repeat("a", 64), gguf: "first.gguf"},
		{name: "oversized", body: strings.Repeat(" ", 1<<20) + string(valid), digest: strings.Repeat("a", 64), gguf: "first.gguf"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			manifest := ""
			if !tc.noManifest {
				manifest = writeRawDecodeShardManifest(t, []byte(tc.body))
			}
			setRawDecodeShardFlags(t, manifest, tc.digest, tc.single)
			f := testRawDecodeFlags(1, 1)
			*f.gguf = tc.gguf
			*f.out = filepath.Join(t.TempDir(), "rejected.json")
			calls := 0
			err := runRawDecodeArtifactWith(f, nil, func(context.Context, rawdecode.Request) (rawdecode.Execution, error) {
				calls++
				return rawdecode.Execution{}, errors.New("executor must not be reached")
			})
			if err == nil || calls != 0 {
				t.Fatalf("invalid manifest admission: refused=%v executor_calls=%d", err != nil, calls)
			}
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestRawDecodeShardCLIEntrypoint(t *testing.T) {
	if os.Getenv("FAK_MODELBENCH_SHARD_TEST_CHILD") == "1" {
		for i, arg := range os.Args {
			if arg == "--" {
				os.Args = append([]string{os.Args[0]}, os.Args[i+1:]...)
				main()
				return
			}
		}
		t.Fatal("missing child CLI arguments")
	}
	t.Setenv("FAK_GGUF_MMAP", "0")
	t.Setenv("FAK_PAGED_KV", "0")
	req, data := rawDecodeShardFixture(t)
	manifest := writeRawDecodeShardManifest(t, data)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, digest string
		accept       bool
	}{
		{"authenticated", req.ExpectedArtifactShardSetSHA256, true},
		{"wrong independent digest", strings.Repeat("0", 64), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "cli.json")
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, executable, "-test.run=^TestRawDecodeShardCLIEntrypoint$", "--",
				"-raw-decode", "-gguf="+req.ArtifactPath, "-raw-artifact-shards="+manifest,
				"-raw-artifact-shard-set-sha256="+tc.digest, "-raw-prompt-ids=0", "-raw-context=2",
				"-raw-ignore-eos", "-decode-steps=1", "-decode-reps=1", "-backend=cpu-ref", "-out="+out)
			cmd.Env = append(os.Environ(), "FAK_MODELBENCH_SHARD_TEST_CHILD=1")
			output, runErr := cmd.CombinedOutput()
			if ctx.Err() != nil {
				t.Fatal(ctx.Err())
			}
			if !tc.accept {
				var exitErr *exec.ExitError
				if !errors.As(runErr, &exitErr) || exitErr.ExitCode() != 1 {
					t.Fatalf("wrong digest exit=%v, want 1; output=%s", runErr, output)
				}
				if _, statErr := os.Stat(out); !os.IsNotExist(statErr) {
					t.Fatal("rejected artifact produced an ordinary report")
				}
				return
			}
			if runErr != nil {
				t.Fatalf("ordinary CLI rejected authenticated manifest: %v; output=%s", runErr, output)
			}
			data, err := os.ReadFile(out)
			if err != nil {
				t.Fatal(err)
			}
			var report struct {
				Shards []ggufload.ShardIdentity `json:"artifact_shards"`
				SetSHA string                   `json:"artifact_shard_set_sha256"`
				SHA    string                   `json:"artifact_sha256"`
				Tokens int                      `json:"actual_tokens_generated"`
			}
			if err := json.Unmarshal(data, &report); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(report.Shards, req.ArtifactShards) ||
				report.SetSHA != req.ExpectedArtifactShardSetSHA256 || report.SHA != "" || report.Tokens != 1 {
				t.Fatal("ordinary CLI failed to preserve authenticated execution identity and token count")
			}
		})
	}
}
