package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

func TestNewModelManifestCLIEmitsPacketAndTypedRefusal(t *testing.T) {
	fixture := func(name string) string {
		return filepath.Join("..", "..", "internal", "newmodel", "testdata", name)
	}

	var stdout, stderr bytes.Buffer
	if code := runNewModel(&stdout, &stderr, []string{"--from-manifest", fixture("qwen38-valid.json"), "--json"}); code != 0 {
		t.Fatalf("valid manifest exit=%d stderr=%s", code, stderr.String())
	}
	var packet newmodel.Packet
	if err := json.Unmarshal(stdout.Bytes(), &packet); err != nil {
		t.Fatal(err)
	}
	if packet.Schema != newmodel.PacketSchema || packet.Engine != "fak-native" || packet.ExternalRuntimeFallback {
		t.Fatalf("packet execution = %+v", packet)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runNewModel(&stdout, &stderr, []string{"--from-manifest", fixture("qwen38-unknown-delta.json"), "--json"}); code != 3 {
		t.Fatalf("semantic-delta exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var refusal newmodel.Refusal
	if err := json.Unmarshal(stderr.Bytes(), &refusal); err != nil {
		t.Fatal(err)
	}
	if refusal.Reason != newmodel.RefusalUnknownSemanticDelta || refusal.Phase != "pre-allocation" || refusal.Axis != "attention" {
		t.Fatalf("refusal = %+v", refusal)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runNewModel(&stdout, &stderr, []string{"--family", "fixture", "--json"}); code != 0 {
		t.Fatalf("legacy scaffold exit=%d stderr=%s", code, stderr.String())
	}
	var scaffold newmodel.Result
	if err := json.Unmarshal(stdout.Bytes(), &scaffold); err != nil {
		t.Fatal(err)
	}
	if len(scaffold.NextSteps) == 0 || !strings.HasPrefix(scaffold.NextSteps[0], "P0 prerequisite:") || !strings.Contains(scaffold.NextSteps[0], "--from-manifest release.json --json") || !strings.Contains(scaffold.NextSteps[0], "pretrained_drafter") {
		t.Fatalf("legacy JSON scaffold hid the P0 pinned intake: %+v", scaffold)
	}
	stdout.Reset()
	stderr.Reset()
	if code := runNewModel(&stdout, &stderr, []string{"--family", "fixture"}); code != 0 || !strings.Contains(stdout.String(), scaffold.NextSteps[0]) {
		t.Fatalf("legacy text scaffold lost the same P0 prerequisite: exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNewModelHFScaffoldFlagsAndOfflineEmit(t *testing.T) {
	rev := strings.Repeat("d", 40)
	header := []byte(`{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`)
	shard := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	shard = append(append(shard, header...), make([]byte, 4096)...)
	files := map[string][]byte{
		"config.json":       []byte(`{"architectures":["ToyForCausalLM"],"model_type":"toy","hidden_size":2}`),
		"model.safetensors": shard,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name, ok := strings.CutPrefix(r.URL.Path, "/acme/toy/resolve/"+rev+"/")
		if data, found := files[name]; ok && found {
			http.ServeContent(w, r, name, time.Time{}, bytes.NewReader(data))
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	t.Setenv("HF_TOKEN", "")

	out := t.TempDir()
	for _, argv := range [][]string{
		{"--hf", "acme/toy@main", "--family", "toy", "--out", out},
		{"--hf", "acme/toy@" + rev, "--out", out},
		{"--hf", "acme/toy@" + rev, "--family", "toy", "--from-manifest", "m.json", "--json"},
		{"--family", "toy", "--out", out},
	} {
		var stdout, stderr bytes.Buffer
		if code := runNewModel(&stdout, &stderr, argv); code != 2 {
			t.Fatalf("%v exit=%d, want 2; stderr=%s", argv, code, stderr.String())
		}
	}

	var stdout, stderr bytes.Buffer
	argv := []string{"--hf", "acme/toy@" + rev, "--family", "toy", "--hf-endpoint", srv.URL, "--out", out, "--json"}
	if code := runNewModel(&stdout, &stderr, argv); code != 0 {
		t.Fatalf("hf emit exit=%d stderr=%s", code, stderr.String())
	}
	var res struct {
		Files      []string                  `json:"files"`
		Insertions []newmodel.SpineInsertion `json:"insertions"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatal(err)
	}
	if len(res.Files) == 0 || len(res.Insertions) != 2 {
		t.Fatalf("result = %+v", res)
	}
	for _, rel := range res.Files {
		if _, err := os.Stat(filepath.Join(out, filepath.FromSlash(rel))); err != nil {
			t.Fatalf("listed file not written: %v", err)
		}
	}
	got, err := os.ReadFile(filepath.Join(out, "internal", "model", "testdata", "toy", "config.json"))
	if err != nil || !bytes.Equal(got, files["config.json"]) {
		t.Fatalf("emitted config = %q, %v", got, err)
	}
	stdout.Reset()
	stderr.Reset()
	if code := runNewModel(&stdout, &stderr, argv); code != 1 {
		t.Fatalf("re-emit over an existing spine exit=%d, want 1", code)
	}
}
