package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/localbench"
)

// fak-test:runtime fast est=15ms
func TestBenchmarksV41EPOwnerRegisteredAdapter(t *testing.T) {
	t.Run("describe-is-explicitly-unavailable", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := runBenchmarks(&stdout, &stderr, []string{"describe", "v41-ep-owner"}); code != 0 || stderr.Len() != 0 {
			t.Fatalf("describe code=%d stderr=%s", code, stderr.String())
		}
		for _, want := range []string{"native execution is unavailable", "--validate", "tools/v41epguard", "package initialization"} {
			if !strings.Contains(stdout.String(), want) {
				t.Fatalf("missing diagnostic boundary %q: %s", want, stdout.String())
			}
		}
	})
	t.Run("run-dispatches-in-process-validation", func(t *testing.T) {
		req := localbench.V41EPOwnerRequest{
			Schema: localbench.V41EPOwnerRequestSchema, RunID: "registered-diagnostic", WorldSize: 3,
			Shards:       []ggufload.ShardIdentity{{Path: "/synthetic/owner.gguf", Size: 64, RawFileSHA256: strings.Repeat("a", 64)}},
			SourceSHA256: strings.Repeat("c", 64), CoordinatorAddress: "127.0.0.1:19341",
			Owners: []localbench.V41EPOwnerSelection{
				{Rank: 0, Backend: "vulkan", ExpectedDevice: "device 0"},
				{Rank: 1, Backend: "vulkan", ExpectedDevice: "device 1"},
				{Rank: 2, Backend: "vulkan", ExpectedDevice: "device 2"},
			},
			PromptTokenIDs: []int{0, 1}, ContextLimit: 4, GeneratedTokenLimit: 2, Repetitions: 1,
			Budgets:    localbench.V41EPOwnerBudgets{JoinNS: time.Second, LoadNS: time.Second, ForwardNS: time.Second, CleanupNS: time.Second, TotalNS: 5 * time.Second},
			Supervisor: localbench.V41EPOwnerSupervisorContract{Kind: "external-process-group", ContractID: "declared-only", TerminateNS: 5 * time.Second, KillGraceNS: time.Second},
		}
		manifest, _ := json.Marshal(req.Shards)
		digest := sha256.Sum256(manifest)
		req.ExpectedShardSetSHA256 = hex.EncodeToString(digest[:])
		raw, err := json.Marshal(req)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "request.json")
		if err := os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		var stdout, stderr bytes.Buffer
		code := runBenchmarks(&stdout, &stderr, []string{"run", "v41-ep-owner", "--", "--validate", "--request", path, "--rank", "2"})
		var got struct {
			Schema             string `json:"schema"`
			Status             string `json:"status"`
			RequestSHA256      string `json:"request_sha256"`
			Rank               int    `json:"rank"`
			ExecutionAvailable bool   `json:"execution_available"`
			ExecutionPerformed bool   `json:"execution_performed"`
		}
		want, _ := localbench.V41EPOwnerRequestDigest(req)
		if code != 0 || stderr.Len() != 0 || json.Unmarshal(stdout.Bytes(), &got) != nil || got.Schema != "fak.v41-ep-owner.command-diagnostic/v1" || got.Status != "request_valid" || got.Rank != 2 || got.RequestSHA256 != want || got.ExecutionAvailable || got.ExecutionPerformed {
			t.Fatalf("registered validation failed: code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
		}
	})
	t.Run("reserved-owner-and-default-do-not-launch", func(t *testing.T) {
		for _, extra := range [][]string{nil, {"--supervised-owner"}, {"--supervised-owner", "--approve-native-artifacts"}} {
			var stdout, stderr bytes.Buffer
			args := append([]string{"run", "v41-ep-owner", "--"}, extra...)
			code := runBenchmarks(&stdout, &stderr, args)
			if code == 0 || stderr.Len() == 0 || strings.Contains(stdout.String(), "running: go") || strings.Contains(stdout.String(), "Run it directly") {
				t.Fatalf("reserved route escaped refusal: args=%v code=%d stdout=%s stderr=%s", args, code, stdout.String(), stderr.String())
			}
			if len(extra) == 1 && (!strings.Contains(stdout.String(), `"status":"refused"`) || !strings.Contains(stdout.String(), `"inherited_descriptors_adopted":false`)) {
				t.Fatalf("owner entry did not reach explicit pre-adoption refusal: %s", stdout.String())
			}
		}
	})
}
