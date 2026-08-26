package issue8385witness

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
)

const minimumDecodeTokens = 2048

func TestPartialNativeHoldReadback(t *testing.T) {
	var response struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens        int            `json:"prompt_tokens"`
			CompletionTokens    int            `json:"completion_tokens"`
			TotalTokens         int            `json:"total_tokens"`
			PromptTokensDetails map[string]any `json:"prompt_tokens_details"`
		} `json:"usage"`
		Fak struct {
			DecodeTrace struct {
				Schema string `json:"schema"`
				Engine string `json:"engine"`
				Events []struct {
					TokenIndex int   `json:"token_index"`
					ElapsedNS  int64 `json:"elapsed_ns"`
				} `json:"events"`
			} `json:"decode_trace"`
			TokenIDs struct {
				Schema   string `json:"schema"`
				Engine   string `json:"engine"`
				TokenIDs []int  `json:"token_ids"`
			} `json:"native_decode_token_ids"`
			HeavyReceipt json.RawMessage `json:"native_inference_receipt"`
		} `json:"fak"`
	}
	raw := readFile(t, "native-response.json")
	if err := json.Unmarshal(raw, &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Choices) != 1 || response.Choices[0].FinishReason != "stop" {
		t.Fatalf("choices/finish = %d/%q, want 1/stop", len(response.Choices), response.Choices[0].FinishReason)
	}
	if response.Usage.PromptTokens != 82 || response.Usage.CompletionTokens != 251 || response.Usage.TotalTokens != 333 || response.Usage.PromptTokensDetails == nil {
		t.Fatalf("realistic usage = %+v", response.Usage)
	}
	if response.Usage.CompletionTokens >= minimumDecodeTokens {
		t.Fatalf("completion_tokens=%d unexpectedly eligible for long-decode verdict", response.Usage.CompletionTokens)
	}
	if response.Fak.DecodeTrace.Schema != "fak.native-decode-trace/1" || response.Fak.DecodeTrace.Engine != "fak-native" {
		t.Fatalf("trace provenance = %q/%q", response.Fak.DecodeTrace.Schema, response.Fak.DecodeTrace.Engine)
	}
	if response.Fak.TokenIDs.Schema != "fak.native-decode-token-ids/1" || response.Fak.TokenIDs.Engine != "fak-native" {
		t.Fatalf("token-ID provenance = %q/%q", response.Fak.TokenIDs.Schema, response.Fak.TokenIDs.Engine)
	}
	if len(response.Fak.DecodeTrace.Events) != response.Usage.CompletionTokens || len(response.Fak.TokenIDs.TokenIDs) != response.Usage.CompletionTokens {
		t.Fatalf("events/IDs/completion = %d/%d/%d", len(response.Fak.DecodeTrace.Events), len(response.Fak.TokenIDs.TokenIDs), response.Usage.CompletionTokens)
	}
	if len(response.Fak.HeavyReceipt) != 0 || bytes.Contains(raw, []byte("native_inference_receipt")) {
		t.Fatal("partial native response unexpectedly contains heavyweight receipt")
	}
	var previous int64
	for i, event := range response.Fak.DecodeTrace.Events {
		if event.TokenIndex != i+1 || event.ElapsedNS < 0 || (i > 0 && event.ElapsedNS < previous) || response.Fak.TokenIDs.TokenIDs[i] < 0 {
			t.Fatalf("event/ID %d = %+v/%d", i, event, response.Fak.TokenIDs.TokenIDs[i])
		}
		previous = event.ElapsedNS
	}

	var verdict struct {
		Verdict    string `json:"verdict"`
		Promotion  string `json:"promotion"`
		Comparator struct {
			Started     bool `json:"started"`
			Repetitions int  `json:"repetitions"`
		} `json:"comparator"`
		Eligible struct {
			Decay   bool `json:"decay_verdict"`
			Parity  bool `json:"parity_verdict"`
			Matched bool `json:"matched_3x2_verdict"`
		} `json:"eligible"`
	}
	readJSON(t, "verdict.json", &verdict)
	if verdict.Verdict != "HOLD" || verdict.Promotion != "DEMOTE" || verdict.Comparator.Started || verdict.Comparator.Repetitions != 0 || verdict.Eligible.Decay || verdict.Eligible.Parity || verdict.Eligible.Matched {
		t.Fatalf("ineligible partial verdict = %+v", verdict)
	}
}

func TestCleanupAndMemorySafetyReadback(t *testing.T) {
	var cleanup struct {
		CleanupStatus int `json:"cleanup_status"`
		Restoration   struct {
			Required  bool     `json:"required_at_cleanup"`
			Attempted bool     `json:"attempted"`
			Exact     bool     `json:"exact_service_restored_or_preserved"`
			Type      string   `json:"service_type"`
			Checks    []string `json:"verification"`
		} `json:"restoration"`
	}
	readJSON(t, "abort-restoration.json", &cleanup)
	if cleanup.CleanupStatus != 0 || !cleanup.Restoration.Required || !cleanup.Restoration.Attempted || !cleanup.Restoration.Exact || cleanup.Restoration.Type != "Submitted" || len(cleanup.Restoration.Checks) != 6 {
		t.Fatalf("cleanup receipt = %+v", cleanup)
	}
	verifySingleHashFile(t, "abort-restoration.sha256")

	var contract struct {
		During struct {
			MaxSwap   int64 `json:"maximum_swap_growth_bytes"`
			MinFree   int64 `json:"minimum_free_memory_percent"`
			MinStatus int64 `json:"minimum_memorystatus_percent"`
		} `json:"during_load_and_requests"`
	}
	readJSON(t, "memory-safety-contract.json", &contract)
	if contract.During.MaxSwap != 12*1024*1024*1024 || contract.During.MinFree != 10 || contract.During.MinStatus != 10 {
		t.Fatalf("memory contract = %+v", contract.During)
	}

	scanner := bufio.NewScanner(bytes.NewReader(readFile(t, "native-memory-samples.tsv")))
	rows := 0
	for scanner.Scan() {
		if rows == 0 {
			if scanner.Text() != "timestamp\trss_bytes\tfree_percent\tmemorystatus_percent\tswap_bytes\tswap_growth_bytes\tcrossing" {
				t.Fatalf("memory header = %q", scanner.Text())
			}
			rows++
			continue
		}
		fields := strings.Split(scanner.Text(), "\t")
		if len(fields) != 7 {
			t.Fatalf("memory row fields=%d: %q", len(fields), scanner.Text())
		}
		free := parseInt(t, fields[2])
		status := parseInt(t, fields[3])
		growth := parseInt(t, fields[5])
		crossing := parseInt(t, fields[6])
		if free < contract.During.MinFree || status < contract.During.MinStatus || growth > contract.During.MaxSwap || crossing != 0 {
			t.Fatalf("unsafe memory row: %q", scanner.Text())
		}
		rows++
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if rows < 2 {
		t.Fatal("memory evidence has no samples")
	}
}

func TestArtifactSHA256(t *testing.T) {
	verifySingleHashFile(t, "SHA256SUMS")
	var provenance struct {
		Source struct {
			Manifest string `json:"file_manifest_sha256"`
		} `json:"source"`
		Comparator struct {
			Started bool `json:"started"`
		} `json:"comparator"`
	}
	readJSON(t, "provenance.json", &provenance)
	if provenance.Source.Manifest != fileSHA256(t, "source-files.sha256") || provenance.Comparator.Started {
		t.Fatalf("provenance manifest/comparator = %q/%v", provenance.Source.Manifest, provenance.Comparator.Started)
	}
}

func verifySingleHashFile(t *testing.T, name string) {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(string(readFile(t, name))), "\n")
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			t.Fatalf("%s malformed line %q", name, line)
		}
		if got := fileSHA256(t, fields[1]); got != fields[0] {
			t.Fatalf("%s: %s hash=%s want=%s", name, fields[1], got, fields[0])
		}
	}
}

func readJSON(t *testing.T, name string, dst any) {
	t.Helper()
	if err := json.Unmarshal(readFile(t, name), dst); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

func readFile(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func fileSHA256(t *testing.T, name string) string {
	t.Helper()
	digest := sha256.Sum256(readFile(t, name))
	return hex.EncodeToString(digest[:])
}

func parseInt(t *testing.T, raw string) int64 {
	t.Helper()
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		t.Fatalf("parse %q: %v", raw, err)
	}
	return value
}
