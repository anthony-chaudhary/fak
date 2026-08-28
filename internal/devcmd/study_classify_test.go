package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studyclass"
	"github.com/anthony-chaudhary/fak/internal/studyforge"
)

func TestStudyClassifyCLIRoundTripAndDeterminism(t *testing.T) {
	corpusPath := writeStudyClassifyCorpus(t)
	dir := t.TempDir()
	fullA := filepath.Join(dir, "a", "full.json")
	indexA := filepath.Join(dir, "a", "index.json")
	var stdout, stderr bytes.Buffer
	args := []string{"classify", "--corpus", corpusPath, "--out", fullA, "--index-out", indexA, "--related-limit", "1"}
	if code := RunStudyClassify(&stdout, &stderr, args); code != 0 {
		t.Fatalf("classify exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	for _, want := range []string{"records: 1", "source.issues: 1", "disposition.regression_bug: 1", "mechanism.kv_cache: 1", "state.open: 1", "confidence.high: 1"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("summary missing %q:\n%s", want, stdout.String())
		}
	}

	fullFile, err := os.Open(fullA)
	if err != nil {
		t.Fatal(err)
	}
	classified, err := studyclass.ReadJSON(fullFile)
	_ = fullFile.Close()
	if err != nil {
		t.Fatal(err)
	}
	if classified.Summary.RecordCount != 1 || len(classified.Records) != 1 {
		t.Fatalf("classification coverage = %+v", classified.Summary)
	}
	indexFile, err := os.Open(indexA)
	if err != nil {
		t.Fatal(err)
	}
	index, err := studyclass.ReadCompactJSON(indexFile)
	_ = indexFile.Close()
	if err != nil {
		t.Fatal(err)
	}
	if index.RelatedSampleLimit != 1 || !reflect.DeepEqual(index.Summary, classified.Summary) {
		t.Fatalf("compact index mismatch: %+v", index)
	}

	stdout.Reset()
	stderr.Reset()
	if code := RunStudyClassify(&stdout, &stderr, []string{"validate", "--classification", fullA, "--corpus", corpusPath}); code != 0 {
		t.Fatalf("validate exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := RunStudyClassify(&stdout, &stderr, []string{"validate-index", "--index", indexA, "--classification", fullA, "--corpus", corpusPath}); code != 0 {
		t.Fatalf("validate-index exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}

	fullB := filepath.Join(dir, "b", "full.json")
	indexB := filepath.Join(dir, "b", "index.json")
	stdout.Reset()
	stderr.Reset()
	if code := RunStudyClassify(&stdout, &stderr, []string{"classify", "--corpus", corpusPath, "--out", fullB, "--index-out", indexB, "--related-limit", "1", "--json"}); code != 0 {
		t.Fatalf("second classify exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	var summary studyclass.Summary
	if err := json.Unmarshal(stdout.Bytes(), &summary); err != nil {
		t.Fatalf("decode JSON summary: %v: %q", err, stdout.String())
	}
	if summary.RecordCount != 1 {
		t.Fatalf("JSON summary = %+v", summary)
	}
	assertStudyClassifyFilesEqual(t, fullA, fullB)
	assertStudyClassifyFilesEqual(t, indexA, indexB)
}

func TestStudyClassifyCLICorruptionAndUsage(t *testing.T) {
	corpusPath := writeStudyClassifyCorpus(t)
	dir := t.TempDir()
	full := filepath.Join(dir, "full.json")
	index := filepath.Join(dir, "index.json")
	var stdout, stderr bytes.Buffer
	if code := RunStudyClassify(&stdout, &stderr, []string{"classify", "--corpus", corpusPath, "--out", full, "--index-out", index}); code != 0 {
		t.Fatalf("classify exit=%d stderr=%q", code, stderr.String())
	}
	b, err := os.ReadFile(full)
	if err != nil {
		t.Fatal(err)
	}
	b = bytes.Replace(b, []byte("{"), []byte(`{"unexpected":true,`), 1)
	corrupt := filepath.Join(dir, "corrupt.json")
	if err := os.WriteFile(corrupt, b, 0600); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := RunStudyClassify(&stdout, &stderr, []string{"validate", "--classification", corrupt, "--corpus", corpusPath}); code != 1 {
		t.Fatalf("corrupt validate exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "unknown field") {
		t.Fatalf("corruption error did not report strict decode: %q", stderr.String())
	}
	corpusBytes, err := os.ReadFile(corpusPath)
	if err != nil {
		t.Fatal(err)
	}
	corpusBytes = bytes.Replace(corpusBytes, []byte("{"), []byte(`{"unexpected":true,`), 1)
	corruptCorpus := filepath.Join(dir, "corrupt-corpus.json")
	if err := os.WriteFile(corruptCorpus, corpusBytes, 0600); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := RunStudyClassify(&stdout, &stderr, []string{"classify", "--corpus", corruptCorpus, "--out", full, "--index-out", index}); code != 1 {
		t.Fatalf("unknown-field corpus exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "unknown field") {
		t.Fatalf("corpus corruption did not report strict decode: %q", stderr.String())
	}

	indexFile, err := os.Open(index)
	if err != nil {
		t.Fatal(err)
	}
	compact, err := studyclass.ReadCompactJSON(indexFile)
	_ = indexFile.Close()
	if err != nil {
		t.Fatal(err)
	}
	compact.FullOutputChecksum = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	tamperedIndex, err := json.MarshalIndent(compact, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	tamperedIndex = append(tamperedIndex, '\n')
	tamperedIndexPath := filepath.Join(dir, "tampered-index.json")
	if err := os.WriteFile(tamperedIndexPath, tamperedIndex, 0600); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := RunStudyClassify(&stdout, &stderr, []string{"validate-index", "--index", tamperedIndexPath, "--classification", full, "--corpus", corpusPath}); code != 1 {
		t.Fatalf("tampered index exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "does not match full classification") {
		t.Fatalf("tampered index error=%q", stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	if code := RunStudyClassify(&stdout, &stderr, []string{"classify", "--corpus", corpusPath}); code != 2 {
		t.Fatalf("incomplete args exit=%d", code)
	}
	stdout.Reset()
	stderr.Reset()
	if code := RunStudyClassify(&stdout, &stderr, []string{"classify", "--corpus", corpusPath, "--out", full, "--index-out", full}); code != 2 {
		t.Fatalf("same outputs exit=%d", code)
	}
}

func TestStudyClassifyCLISchema(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := RunStudyClassify(&stdout, &stderr, []string{"schema"}); code != 0 {
		t.Fatalf("schema exit=%d stderr=%q", code, stderr.String())
	}
	var schema map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &schema); err != nil {
		t.Fatal(err)
	}
	if schema["$id"] != studyclass.JSONSchemaID {
		t.Fatalf("schema id=%v", schema["$id"])
	}
	if !bytes.Equal(stdout.Bytes(), studyclass.SchemaDocument()) {
		t.Fatal("CLI schema differs from embedded schema")
	}
}

func writeStudyClassifyCorpus(t *testing.T) string {
	t.Helper()
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits":
			fmt.Fprint(w, `[{"sha":"study-classify-revision"}]`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprint(w, `{"sha":"study-classify-revision"}`)
		case "/repos/acme/widget/issues":
			fmt.Fprint(w, `[{"id":7,"node_id":"I_7","number":7,"title":"KV cache regression","state":"open","labels":[{"name":"bug"}],"created_at":"2026-08-25T00:00:00Z"}]`)
		case "/repos/acme/widget/pulls", "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	collector := studyforge.NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.Now = func() time.Time { return cutoff }
	corpus, err := collector.Capture(context.Background(), studyforge.CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: cutoff,
	})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "corpus.json")
	if err := studyforge.Write(path, corpus); err != nil {
		t.Fatal(err)
	}
	return path
}

func assertStudyClassifyFilesEqual(t *testing.T, a, b string) {
	t.Helper()
	aBytes, err := os.ReadFile(a)
	if err != nil {
		t.Fatal(err)
	}
	bBytes, err := os.ReadFile(b)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(aBytes, bBytes) {
		t.Fatalf("files differ: %s %s", a, b)
	}
}
