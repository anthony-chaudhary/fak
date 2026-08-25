package witness

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

type row struct {
	ExperimentID string `json:"experiment_id"`
	Arm          string `json:"arm"`
	Rep          int    `json:"rep"`
	Content      string `json:"content"`
	QualityPass  bool   `json:"quality_pass"`
	Usage        struct {
		PromptTokensDetails struct {
			CachedTokens *int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

type summary struct {
	Schema      string `json:"schema"`
	Engine      string `json:"engine"`
	Backend     string `json:"backend"`
	ForwardPath string `json:"forward_path"`
	Artifact    struct {
		SHA256 string `json:"sha256"`
	} `json:"artifact"`
	Source struct {
		ArchiveSHA256   string `json:"archive_sha256"`
		RowsSHA256      string `json:"rows_sha256"`
		IdentitySHA256  string `json:"identity_sha256"`
		ServerLogSHA256 string `json:"server_log_sha256"`
	} `json:"source"`
	Verdict string `json:"verdict"`
	Parity  string `json:"parity"`
}

func readJSON[T any](t *testing.T, path string) T {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var v T
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	return v
}

func fileSHA256(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func TestCacheAttributionWitness(t *testing.T) {
	s := readJSON[summary](t, "summary.json")
	if s.Schema != "fak.qwen38.cache_attribution.v1" {
		t.Fatalf("schema=%q", s.Schema)
	}
	if s.Engine != "fak-native" || s.Backend != "cuda" || s.ForwardPath != "cuda/qwen35-gdn-ssm-decode-v1" {
		t.Fatalf("native identity mismatch: engine=%q backend=%q path=%q", s.Engine, s.Backend, s.ForwardPath)
	}
	if s.Artifact.SHA256 != "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169" {
		t.Fatalf("artifact hash=%q", s.Artifact.SHA256)
	}
	if s.Source.ArchiveSHA256 != "5644b66807fc3fc5eb69036808986079acfdf75588aec622a07621f198ab2a97" || s.Source.ServerLogSHA256 != "db21e023d2f4a2d62803fb63db4ee4c0e6f8c588149548dc9b0d88d7a3c05479" {
		t.Fatalf("source hashes changed: %+v", s.Source)
	}
	if got := fileSHA256(t, "rows.json"); got != s.Source.RowsSHA256 {
		t.Fatalf("rows hash=%s want=%s", got, s.Source.RowsSHA256)
	}
	if got := fileSHA256(t, "identity.txt"); got != s.Source.IdentitySHA256 {
		t.Fatalf("identity hash=%s want=%s", got, s.Source.IdentitySHA256)
	}
	b, err := os.ReadFile("identity.txt")
	if err != nil {
		t.Fatal(err)
	}
	identity := string(b)
	for _, marker := range []string{"BACKEND_FORWARD_PREFLIGHT_OK backend=cuda", "forward_path=cuda/qwen35-gdn-ssm-decode-v1", "q4k=true"} {
		if !strings.Contains(identity, marker) {
			t.Fatalf("identity missing %q", marker)
		}
	}
	for _, forbidden := range []string{"engine=llama", "engine=vllm", "delegated=true"} {
		if strings.Contains(identity, forbidden) {
			t.Fatalf("identity contains reference marker %q", forbidden)
		}
	}
	if s.Verdict != "HOLD_CACHE_RESTORE_REGRESSION" || s.Parity != "HOLD_BELOW_PARITY" {
		t.Fatalf("unsafe verdict: %q %q", s.Verdict, s.Parity)
	}
}

func TestRowsProveColdQualityAndCacheFailure(t *testing.T) {
	rows := readJSON[[]row](t, "rows.json")
	if len(rows) != 10 {
		t.Fatalf("rows=%d want=10", len(rows))
	}
	ids := make(map[string]bool, len(rows))
	coldPass, cachePass, cacheHits := 0, 0, 0
	var cacheHitReps []int
	for _, r := range rows {
		if r.ExperimentID == "" || ids[r.ExperimentID] {
			t.Fatalf("missing or duplicate id=%q", r.ExperimentID)
		}
		ids[r.ExperimentID] = true
		switch r.Arm {
		case "cold-unique":
			if r.Content != "Q38" || !r.QualityPass {
				t.Fatalf("cold %s content=%q pass=%v", r.ExperimentID, r.Content, r.QualityPass)
			}
			if r.Usage.PromptTokensDetails.CachedTokens != nil && *r.Usage.PromptTokensDetails.CachedTokens != 0 {
				t.Fatalf("cold %s unexpectedly cached", r.ExperimentID)
			}
			coldPass++
		case "cache-identical":
			if r.Content != "Stable" || r.QualityPass {
				t.Fatalf("cache %s content=%q pass=%v", r.ExperimentID, r.Content, r.QualityPass)
			}
			if r.QualityPass {
				cachePass++
			}
			if r.Rep >= 2 {
				if r.Usage.PromptTokensDetails.CachedTokens == nil || *r.Usage.PromptTokensDetails.CachedTokens != 24 {
					t.Fatalf("cache hit %s tokens=%v", r.ExperimentID, r.Usage.PromptTokensDetails.CachedTokens)
				}
				cacheHits++
				cacheHitReps = append(cacheHitReps, r.Rep)
			}
		default:
			t.Fatalf("unexpected arm %q", r.Arm)
		}
	}
	if coldPass != 5 || cachePass != 0 || cacheHits != 4 {
		t.Fatalf("cold_pass=%d cache_pass=%d cache_hits=%d", coldPass, cachePass, cacheHits)
	}
	if !reflect.DeepEqual(cacheHitReps, []int{2, 3, 4, 5}) {
		t.Fatalf("cache hit reps=%v", cacheHitReps)
	}
}
