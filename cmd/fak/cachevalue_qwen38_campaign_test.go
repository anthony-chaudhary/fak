package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunCachevalueQwen38CampaignSpine(t *testing.T) {
	input := filepath.Join(t.TempDir(), "campaign.json")
	body := `{"schema":"fak.qwen38_cache_campaign.v1","corpus":"workflow-v1","hardware":"macbook-m4-max","workload":{"turns":5,"repeated_system_prompt":true,"repeated_tool_schema":true,"growing_conversation":true,"correlated_tool_calls":true,"prefix_mutation":true,"restart_boundary":true},"identity":{"alias":"qwen38:27b","ref":"hf://unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-Q4_K_M.gguf","revision":"rev","sha256":"weights","tokenizer_sha256":"tok","chat_template_hash":"template","quant":"Q4_K_M","backend":"metal","tool_schema_hash":"tools","policy_hash":"policy"},"observations":[{"mode":"cold","trial":1,"wall_ms":100,"ttft_ms":50,"prompt_tokens":1000,"reused_prompt_tokens":0,"output_hash":"text","tool_call_hash":"tool","structured_json_hash":"json","cache_hit":false},{"mode":"fak","trial":1,"wall_ms":60,"ttft_ms":25,"prompt_tokens":1000,"reused_prompt_tokens":700,"cache_lookup_ms":1,"serialization_ms":1,"output_hash":"text","tool_call_hash":"tool","structured_json_hash":"json","cache_hit":true},{"mode":"combined","trial":1,"wall_ms":50,"ttft_ms":20,"prompt_tokens":1000,"reused_prompt_tokens":800,"cache_lookup_ms":1,"serialization_ms":1,"output_hash":"text","tool_call_hash":"tool","structured_json_hash":"json","cache_hit":true},{"mode":"fak","trial":2,"wall_ms":95,"ttft_ms":45,"prompt_tokens":1000,"reused_prompt_tokens":0,"output_hash":"mutated","tool_call_hash":"tool2","structured_json_hash":"json2","expected_invalidation":true,"cache_hit":false}]}`
	if err := os.WriteFile(input, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runCachevalueQwen38Campaign(&stdout, &stderr, []string{"--input", input}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"verdict": "PASS"`) || !strings.Contains(stdout.String(), `"status": "N/A"`) {
		t.Fatalf("output=%s", stdout.String())
	}
}
