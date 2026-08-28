package devcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStudyAdjacencyValidateAndRender(t *testing.T) {
	manifestPath := writeStudyAdjacencyFixture(t)
	var stdout, stderr bytes.Buffer
	if code := RunStudyAdjacency(&stdout, &stderr, []string{"validate", "--manifest", manifestPath}); code != 0 {
		t.Fatalf("validate code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "members=6 candidates=6") {
		t.Fatalf("validate stdout=%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	outPath := filepath.Join(t.TempDir(), "adjacency.md")
	if code := RunStudyAdjacency(&stdout, &stderr, []string{"render", "--manifest", manifestPath, "--out", outPath}); code != 0 {
		t.Fatalf("render code=%d stderr=%s", code, stderr.String())
	}
	rendered, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(rendered), "Processed members (6/6)") {
		t.Fatalf("rendered summary=%s", rendered)
	}
}

func TestStudyAdjacencyRejectsInvalidManifest(t *testing.T) {
	manifestPath := writeStudyAdjacencyFixture(t)
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	raw = bytes.Replace(raw, []byte(`"revision":"1111111111111111111111111111111111111111"`), []byte(`"revision":""`), 1)
	if err := os.WriteFile(manifestPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := RunStudyAdjacency(&stdout, &stderr, []string{"validate", "--manifest", manifestPath}); code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "revision must be a full 40-hex commit") {
		t.Fatalf("stderr=%s", stderr.String())
	}
}

func writeStudyAdjacencyFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "adjacency.json")
	data := `{
  "schema":"fak-study-adjacency/1",
  "id":"test-adjacency",
  "title":"Test adjacency",
  "scope":{
    "bounded_meaning":"Only declared decision-changing runtime peers.",
    "inclusion_criteria":["Named by issue or directly changes a FAK runtime decision."],
    "exclusion_criteria":["Unrelated or governance-only repository."]
  },
  "anchor":{
    "repository":{"owner":"vllm-project","repo":"vllm"},
    "pin":{"revision":"1111111111111111111111111111111111111111","cutoff":"2026-08-26T22:35:00Z","observed_at":"2026-08-27T00:00:00Z"},
    "normalized_records":53848,
    "sha256":"2a66d4876aee3811eb200c0884c6558a5f3ac86c90b6c7f8b92f45b85fe671b2"
  },
  "declared_repositories":[
    {"owner":"sgl-project","repo":"sglang"},
    {"owner":"NVIDIA","repo":"TensorRT-LLM"},
    {"owner":"ai-dynamo","repo":"dynamo"},
    {"owner":"ggml-org","repo":"llama.cpp"},
    {"owner":"flashinfer-ai","repo":"flashinfer"},
    {"owner":"llm-d","repo":"llm-d"}
  ],
  "members":[
    ` + studyAdjacencyFixtureMember("SGLang", "sgl-project", "sglang", "sglang-candidate") + `,
    ` + studyAdjacencyFixtureMember("TensorRT-LLM", "NVIDIA", "TensorRT-LLM", "tensorrt-candidate") + `,
    ` + studyAdjacencyFixtureMember("Dynamo", "ai-dynamo", "dynamo", "dynamo-candidate") + `,
    ` + studyAdjacencyFixtureMember("llama.cpp", "ggml-org", "llama.cpp", "llamacpp-candidate") + `,
    ` + studyAdjacencyFixtureMember("FlashInfer", "flashinfer-ai", "flashinfer", "flashinfer-candidate") + `,
    ` + studyAdjacencyFixtureMember("llm-d", "llm-d", "llm-d", "llmd-candidate") + `
  ]
}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func studyAdjacencyFixtureMember(name, owner, repo, candidateID string) string {
	return `{
      "name":"` + name + `",
      "repository":{"owner":"` + owner + `","repo":"` + repo + `"},
      "pin":{"revision":"2222222222222222222222222222222222222222","cutoff":"2026-08-26T22:35:00Z","observed_at":"2026-08-27T00:00:00Z"},
      "processed":true,
      "inclusion_rationale":"Decision-changing runtime peer.",
      "decision_relation":"Changes a vLLM-derived FAK runtime decision.",
      "freshness_notes":"Revision is pinned at the shared cutoff.",
      "partial_notes":"Forge enrichment remains partial.",
      "source_class_receipts":[{"class":"repository_metadata","status":"complete","terminal_receipt":"fixture","notes":"Terminal fixture."}],
      "candidates":[{
        "id":"` + candidateID + `",
        "title":"Candidate",
        "rationale":"Decision-changing mechanism.",
        "repository_links":[{"owner":"` + owner + `","repo":"` + repo + `"}],
        "vllm_mechanism_link":"vLLM mechanism"
      }]
    }`
}
