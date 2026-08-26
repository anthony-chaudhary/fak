package metalprof

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

type packet struct {
	Schema   string `json:"schema"`
	Issue    int    `json:"issue"`
	Status   string `json:"status"`
	Envelope struct {
		ModelSHA      string `json:"model_sha256"`
		PromptTokens  int    `json:"prompt_tokens"`
		FakEngine     string `json:"fak_engine"`
		Fallback      int    `json:"fak_fallback_count"`
		LlamaRevision string `json:"llama_cpp_revision"`
	} `json:"envelope"`
	Fak struct {
		Runs []struct {
			Wall, Q4Wall, Q4GPU, Q4Roundtrip, Q8CPU, Q6Wall, Residual float64 `json:"-"`
		} `json:"runs"`
		Raw map[string]string `json:"raw_hashes"`
	} `json:"fak"`
	Llama struct {
		Runs []float64 `json:"runs_tok_s"`
		Raw  string    `json:"raw_json_sha256"`
	} `json:"llama_cpp"`
	Unavailable     []string `json:"unavailable"`
	Instrumentation struct {
		Available bool   `json:"available"`
		Reason    string `json:"reason"`
	} `json:"instrumentation"`
	Decision struct {
		Close  bool   `json:"close_issue"`
		Reason string `json:"reason"`
	} `json:"decision"`
}

type rawPacket struct {
	Fak struct {
		Runs []map[string]float64 `json:"runs"`
	} `json:"fak"`
}

func validHash(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size
}

func TestPartialPacketIsHonestAndReconciled(t *testing.T) {
	b, err := os.ReadFile("partial-summary.json")
	if err != nil {
		t.Fatal(err)
	}
	var p packet
	if err := json.Unmarshal(b, &p); err != nil {
		t.Fatal(err)
	}
	var r rawPacket
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if p.Schema != "fak.metal-prefill-attribution.v1" || p.Issue != 9236 || p.Status != "partial_not_closeable" {
		t.Fatalf("bad identity: %+v", p)
	}
	if p.Envelope.ModelSHA != "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169" || p.Envelope.PromptTokens != 32 || p.Envelope.FakEngine != "fak-native Metal" || p.Envelope.Fallback != 0 || p.Envelope.LlamaRevision != "ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0" {
		t.Fatal("envelope drift")
	}
	if len(r.Fak.Runs) != 3 || len(p.Llama.Runs) != 3 {
		t.Fatalf("need three runs per arm: fak=%d llama=%d", len(r.Fak.Runs), len(p.Llama.Runs))
	}
	for i, run := range r.Fak.Runs {
		wall := run["wall_ms"]
		sum := run["q4k_wall_ms"] + run["q8_cpu_ms"] + run["q6k_wall_ms"] + run["residual_ms"]
		if d := sum - wall; d < -0.6 || d > 0.6 {
			t.Fatalf("run %d does not reconcile: wall %.1f sum %.1f", i+1, wall, sum)
		}
		if d := run["q4k_gpu_execute_ms"] + run["q4k_roundtrip_ms"] - run["q4k_wall_ms"]; d < -0.2 || d > 0.2 {
			t.Fatalf("run %d q4 split mismatch", i+1)
		}
	}
	for k, v := range p.Fak.Raw {
		if !validHash(v) {
			t.Fatalf("bad fak %s hash", k)
		}
	}
	if !validHash(p.Llama.Raw) {
		t.Fatal("bad llama hash")
	}
	if p.Instrumentation.Available || p.Instrumentation.Reason == "" || len(p.Unavailable) < 10 || p.Decision.Close || p.Decision.Reason == "" {
		t.Fatal("partial limitations must be explicit and issue must remain open")
	}
}
