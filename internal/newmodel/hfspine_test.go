package newmodel

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"go/format"
	"go/parser"
	"go/token"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
)

const (
	inkling2Repo = "thinkingmachines/Inkling-2-Mini"
	inkling2Rev  = "b85c3cfa615fc413cc9af3379f491b963f3b586a"
)

const fakeHubToken = "hf_fake_test_token"

// fakeHub serves repo files under /{repo}/resolve/{rev}/{file} with native
// Range support and records what it sent.
type fakeHub struct {
	repo, rev   string
	files       map[string][]byte
	ignoreRange bool
	// listing serves the Hub revision file listing; listingStatus, when set,
	// answers it with that status instead.
	listing       bool
	listingStatus int

	mu           sync.Mutex
	bytesServed  int64
	unrangedGets []string
	listingAuth  []string
}

func (h *fakeHub) serveListing(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	h.listingAuth = append(h.listingAuth, r.Header.Get("Authorization"))
	h.mu.Unlock()
	if h.listingStatus != 0 {
		http.Error(w, "listing unavailable", h.listingStatus)
		return
	}
	if !h.listing {
		http.NotFound(w, r)
		return
	}
	type sibling struct {
		Rfilename string `json:"rfilename"`
	}
	var siblings []sibling
	for _, name := range sortedKeys(h.files) {
		siblings = append(siblings, sibling{name})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"id": h.repo, "sha": h.rev, "siblings": siblings})
}

type countingWriter struct {
	http.ResponseWriter
	n int64
}

func (w *countingWriter) Write(p []byte) (int, error) {
	n, err := w.ResponseWriter.Write(p)
	w.n += int64(n)
	return n, err
}

func (h *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/api/models/"+h.repo+"/revision/"+h.rev {
		h.serveListing(w, r)
		return
	}
	name, ok := strings.CutPrefix(r.URL.Path, "/"+h.repo+"/resolve/"+h.rev+"/")
	data, found := h.files[name]
	if !ok || !found {
		http.NotFound(w, r)
		return
	}
	if h.ignoreRange {
		r.Header.Del("Range")
	}
	cw := &countingWriter{ResponseWriter: w}
	http.ServeContent(cw, r, name, time.Time{}, bytes.NewReader(data))
	h.mu.Lock()
	defer h.mu.Unlock()
	h.bytesServed += cw.n
	if strings.HasSuffix(name, ".safetensors") && r.Header.Get("Range") == "" {
		h.unrangedGets = append(h.unrangedGets, name)
	}
}

func (h *fakeHub) source(t *testing.T) (HFRangeSource, func()) {
	t.Helper()
	srv := httptest.NewServer(h)
	ref, err := ParseHFPin(h.repo + "@" + h.rev)
	if err != nil {
		srv.Close()
		t.Fatal(err)
	}
	return HubRangeSource{Client: &hfhub.Client{BaseURL: srv.URL, HTTP: srv.Client(), Token: fakeHubToken}, Ref: ref}, srv.Close
}

func safetensorsFile(t *testing.T, tensors map[string]HFTensor, payload int) []byte {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	for name, ts := range tensors {
		header[name] = map[string]any{"dtype": ts.DType, "shape": ts.Shape, "data_offsets": []int{0, 0}}
	}
	js, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(js)%8 != 0 {
		js = append(js, ' ')
	}
	out := make([]byte, 8, 8+len(js)+payload)
	binary.LittleEndian.PutUint64(out, uint64(len(js)))
	out = append(out, js...)
	return append(out, make([]byte, payload)...)
}

// shardedHub splits tensors across shards by assign and writes the index.
func shardedHub(t *testing.T, repo, rev string, config []byte, tensors map[string]HFTensor, assign func(string) string, payload int) *fakeHub {
	t.Helper()
	perShard := map[string]map[string]HFTensor{}
	weightMap := map[string]string{}
	for name, ts := range tensors {
		shard := assign(name)
		if perShard[shard] == nil {
			perShard[shard] = map[string]HFTensor{}
		}
		perShard[shard][name] = ts
		weightMap[name] = shard
	}
	index, err := json.Marshal(map[string]any{"metadata": map[string]any{"total_size": 0}, "weight_map": weightMap})
	if err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{"config.json": config, "model.safetensors.index.json": index}
	for shard, ts := range perShard {
		files[shard] = safetensorsFile(t, ts, payload)
	}
	return &fakeHub{repo: repo, rev: rev, files: files}
}

func inkling2Fixture(t *testing.T) ([]byte, map[string]HFTensor) {
	t.Helper()
	cfg, tensors, _ := inkling2FixtureSource(t)
	return cfg, tensors
}

// inkling2FixtureSource also returns the hand fixture's source line.
func inkling2FixtureSource(t *testing.T) ([]byte, map[string]HFTensor, string) {
	t.Helper()
	dir := filepath.Join("..", "model", "testdata", "inkling2")
	cfg, err := os.ReadFile(filepath.Join(dir, "inkling_2_mini_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "inkling_2_mini_tensor_index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var inv struct {
		Source  string              `json:"source"`
		Tensors map[string]HFTensor `json:"tensors"`
	}
	if err := json.Unmarshal(raw, &inv); err != nil {
		t.Fatal(err)
	}
	return cfg, inv.Tensors, inv.Source
}

// inkling2Hub lays the hand fixture out as upstream does: one indexed main
// shard plus an mtp.safetensors sidecar the weight_map does not name, whose
// tensors the fixture records under an "mtp:" prefix.
func inkling2Hub(t *testing.T) *fakeHub {
	t.Helper()
	cfg, tensors := inkling2Fixture(t)
	main, mtp := map[string]HFTensor{}, map[string]HFTensor{}
	for name, ts := range tensors {
		if bare, ok := strings.CutPrefix(name, "mtp:"); ok {
			mtp[bare] = ts
			continue
		}
		main[name] = ts
	}
	if len(mtp) == 0 {
		t.Fatal("hand fixture has no mtp: sidecar tensors")
	}
	hub := shardedHub(t, inkling2Repo, inkling2Rev, cfg, main, func(string) string { return "model-00001-of-00001.safetensors" }, 1<<20)
	hub.files["mtp.safetensors"] = safetensorsFile(t, mtp, 1<<10)
	hub.files["README.md"] = []byte("# not a checkpoint file\n")
	hub.listing = true
	return hub
}

func emitInkling2x(t *testing.T) (*HFSpine, *fakeHub) {
	t.Helper()
	return emitHFHub(t, inkling2Hub(t), "inkling2x")
}

func emitHFHub(t *testing.T, hub *fakeHub, family string) (*HFSpine, *fakeHub) {
	t.Helper()
	src, closeSrv := hub.source(t)
	defer closeSrv()
	census, err := CensusHF(context.Background(), src, hub.repo, hub.rev)
	if err != nil {
		t.Fatalf("CensusHF: %v", err)
	}
	spine, err := EmitHFSpine(census, family)
	if err != nil {
		t.Fatalf("EmitHFSpine: %v", err)
	}
	return spine, hub
}

func wantRefusal(t *testing.T, err error, reason RefusalReason) {
	t.Helper()
	var r *Refusal
	if !errors.As(err, &r) || r.Reason != reason {
		t.Fatalf("err = %v, want refusal %s", err, reason)
	}
}

// fak-test:runtime fast est=40ms lane=default
func TestCensusHFRangeReadsHeadersOnlyOverHTTP(t *testing.T) {
	tensors := map[string]HFTensor{
		"model.embed_tokens.weight":              {Shape: []int{32, 8}, DType: "BF16"},
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{8, 8}, DType: "BF16"},
		"model.layers.1.self_attn.q_proj.weight": {Shape: []int{8, 8}, DType: "F8_E4M3"},
		"lm_head.weight":                         {Shape: []int{32, 8}, DType: "BF16"},
	}
	assign := func(name string) string {
		if strings.Contains(name, "layers") {
			return "model-00001-of-00002.safetensors"
		}
		return "model-00002-of-00002.safetensors"
	}
	const payload = 4 << 20
	cfg := []byte(`{"architectures":["ToyForCausalLM"],"model_type":"toy","hidden_size":8}`)
	hub := shardedHub(t, "acme/toy", strings.Repeat("a", 40), cfg, tensors, assign, payload)
	src, closeSrv := hub.source(t)
	defer closeSrv()

	census, err := CensusHF(context.Background(), src, "acme/toy", strings.Repeat("a", 40))
	if err != nil {
		t.Fatalf("CensusHF: %v", err)
	}
	if !reflect.DeepEqual(census.Tensors, tensors) {
		t.Fatalf("tensors = %+v, want %+v", census.Tensors, tensors)
	}
	if !bytes.Equal(census.Config, cfg) {
		t.Fatalf("config bytes changed: %s", census.Config)
	}
	if len(census.Shards) != 2 {
		t.Fatalf("shards = %+v", census.Shards)
	}
	var headerTotal int64
	for _, sh := range census.Shards {
		if digest, err := hex.DecodeString(sh.HeaderSHA256); err != nil || len(digest) != sha256.Size ||
			sh.ShardBytes != int64(len(hub.files[sh.File])) || sh.HeaderBytes <= 0 {
			t.Fatalf("shard %+v, file is %d bytes", sh, len(hub.files[sh.File]))
		}
		headerTotal += 8 + sh.HeaderBytes
	}
	budget := headerTotal + int64(len(cfg)) + int64(len(hub.files["model.safetensors.index.json"]))
	if hub.bytesServed > budget || len(hub.unrangedGets) != 0 {
		t.Fatalf("served %d bytes (budget %d), unranged shard GETs %v: payload was downloaded", hub.bytesServed, budget, hub.unrangedGets)
	}

	hub.ignoreRange = true
	hub.bytesServed = 0
	if _, err := CensusHF(context.Background(), src, "acme/toy", strings.Repeat("a", 40)); err == nil {
		t.Fatal("a server that ignores Range was accepted")
	}

	single := &fakeHub{repo: "acme/one", rev: strings.Repeat("b", 40), files: map[string][]byte{
		"config.json":       cfg,
		"model.safetensors": safetensorsFile(t, tensors, 64),
	}}
	src1, close1 := single.source(t)
	defer close1()
	one, err := CensusHF(context.Background(), src1, "acme/one", strings.Repeat("b", 40))
	if err != nil {
		t.Fatalf("single-file census: %v", err)
	}
	if len(one.Shards) != 1 || one.Shards[0].File != "model.safetensors" || !reflect.DeepEqual(one.Tensors, tensors) {
		t.Fatalf("single-file census = %+v", one)
	}
}

// fak-test:runtime fast est=120ms lane=default
func TestEmitHFSpineFromInkling2FixtureIsDeterministicAndParses(t *testing.T) {
	spine, hub := emitInkling2x(t)
	if len(hub.unrangedGets) != 0 {
		t.Fatalf("unranged shard GETs: %v", hub.unrangedGets)
	}
	cfg, tensors, source := inkling2FixtureSource(t)
	base := "internal/model/testdata/inkling2x/"

	if got := spine.Files[base+"config.json"]; !bytes.Equal(got, cfg) {
		t.Fatal("emitted config.json is not byte-identical to the upstream config")
	}
	var inv struct {
		Source  string              `json:"source"`
		Tensors map[string]HFTensor `json:"tensors"`
	}
	if err := json.Unmarshal(spine.Files[base+"tensor_inventory.json"], &inv); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(inv.Tensors, tensors) {
		t.Fatal("emitted tensor inventory differs from the committed inkling2 tensor index")
	}
	if inv.Source != source {
		t.Fatalf("source = %q, want the hand fixture's %q", inv.Source, source)
	}
	var prov struct {
		Schema string          `json:"schema"`
		Shards []HFShardHeader `json:"shards"`
	}
	if err := json.Unmarshal(spine.Files[base+"provenance.json"], &prov); err != nil {
		t.Fatal(err)
	}
	if prov.Schema != HFCensusSchema || len(prov.Shards) != 2 || prov.Shards[0].HeaderSHA256 == "" {
		t.Fatalf("provenance = %+v", prov)
	}

	for _, name := range []string{"internal/model/inkling2x_spine.go", "internal/model/inkling2x_spine_test.go"} {
		src, ok := spine.Files[name]
		if !ok {
			t.Fatalf("missing %s; have %v", name, spine.Paths())
		}
		if _, err := parser.ParseFile(token.NewFileSet(), name, src, parser.AllErrors); err != nil {
			t.Fatalf("%s does not parse: %v", name, err)
		}
		formatted, err := format.Source(src)
		if err != nil || !bytes.Equal(formatted, src) {
			t.Fatalf("%s is not gofmt-clean: %v", name, err)
		}
	}
	goSrc := string(spine.Files["internal/model/inkling2x_spine.go"])
	for _, want := range []string{
		"func isExactInkling2xConfig(",
		"type UnsupportedInkling2xExecutionError struct",
		"func (c Config) IsInkling2x() bool",
		"func refuseInkling2xNative(c Config) error",
		"func ValidateInkling2xTensorInventory(",
		`inkling2xEnvelopeType = "inkling2_loop"`,
		`inkling2xTextType     = "inkling2_loop_text"`,
		`"text_config.hidden_size":`,
		`"thinkingmachines/Inkling-2-Mini@b85c3cfa"`,
		`"model.llm.layers.20.mlp.experts.w13_weight":`,
	} {
		if !strings.Contains(goSrc, want) {
			t.Fatalf("spine lacks %q", want)
		}
	}
	testSrc := string(spine.Files["internal/model/inkling2x_spine_test.go"])
	for _, want := range []string{"ClassifyForwardPath(cfg, nil)", "errors.As(err, &unsupported)", "*UnsupportedInkling2xExecutionError", "fak-test:runtime fast"} {
		if !strings.Contains(testSrc, want) {
			t.Fatalf("spine test lacks %q", want)
		}
	}
	insert := string(spine.Files[base+"INSERTION.md"])
	for _, want := range []string{"internal/model/arch_support.go", "if cfg.IsQwen35Hybrid() {", "refuseInkling2xNative(cfg)", "internal/model/qwen35.go", "c.IsInkling2x()"} {
		if !strings.Contains(insert, want) {
			t.Fatalf("INSERTION.md lacks %q", want)
		}
	}
	var rows struct {
		Rows []map[string]any `json:"rows"`
	}
	if err := json.Unmarshal(spine.Files["tools/concept_disambiguation_scorecard.data/rows-model-inkling2x.json"], &rows); err != nil || len(rows.Rows) != 1 || rows.Rows[0]["grounding"] != "UnsupportedInkling2xExecutionError" {
		t.Fatalf("scorecard rows = %+v, %v", rows, err)
	}

	again, _ := emitInkling2x(t)
	if !reflect.DeepEqual(spine, again) {
		t.Fatal("two emissions from the same pinned headers differ")
	}
	outA, outB := t.TempDir(), t.TempDir()
	if err := WriteHFSpine(outA, spine); err != nil {
		t.Fatal(err)
	}
	if err := WriteHFSpine(outB, again); err != nil {
		t.Fatal(err)
	}
	if a, b := treeBytes(t, outA), treeBytes(t, outB); !reflect.DeepEqual(a, b) || len(a) != len(spine.Files) {
		t.Fatalf("written trees differ or are incomplete: %d vs %d files, want %d", len(a), len(b), len(spine.Files))
	}
}

func treeBytes(t *testing.T, root string) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		data, err := os.ReadFile(path)
		out[filepath.ToSlash(rel)] = data
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// fak-test:runtime fast est=60ms lane=default
func TestCensusHFAndEmitRefuseBadInputs(t *testing.T) {
	rev := strings.Repeat("c", 40)
	cfg := []byte(`{"architectures":["ToyForCausalLM"],"model_type":"toy"}`)
	tensors := map[string]HFTensor{"a.weight": {Shape: []int{2}, DType: "F32"}, "b.weight": {Shape: []int{2}, DType: "F32"}}
	census := func(h *fakeHub) error {
		src, closeSrv := h.source(t)
		defer closeSrv()
		_, err := CensusHF(context.Background(), src, h.repo, h.rev)
		return err
	}

	missing := shardedHub(t, "acme/toy", rev, cfg, tensors, func(string) string { return "s.safetensors" }, 0)
	missing.files["s.safetensors"] = safetensorsFile(t, map[string]HFTensor{"a.weight": tensors["a.weight"]}, 0)
	wantRefusal(t, census(missing), RefusalTensorMissing)

	orphan := shardedHub(t, "acme/toy", rev, cfg, tensors, func(string) string { return "s.safetensors" }, 0)
	extra := map[string]HFTensor{"c.weight": {Shape: []int{1}, DType: "F32"}}
	for k, v := range tensors {
		extra[k] = v
	}
	orphan.files["s.safetensors"] = safetensorsFile(t, extra, 0)
	wantRefusal(t, census(orphan), RefusalTensorOrphan)

	absentShard := shardedHub(t, "acme/toy", rev, cfg, tensors, func(string) string { return "s.safetensors" }, 0)
	delete(absentShard.files, "s.safetensors")
	wantRefusal(t, census(absentShard), RefusalTensorMissing)

	oversized := shardedHub(t, "acme/toy", rev, cfg, tensors, func(string) string { return "s.safetensors" }, 0)
	huge := make([]byte, 16)
	binary.LittleEndian.PutUint64(huge, uint64(MaxSafetensorsHeaderBytes)+1)
	oversized.files["s.safetensors"] = huge
	wantRefusal(t, census(oversized), RefusalHeaderConfigInvalid)

	noConfig := shardedHub(t, "acme/toy", rev, cfg, tensors, func(string) string { return "s.safetensors" }, 0)
	delete(noConfig.files, "config.json")
	wantRefusal(t, census(noConfig), RefusalHeaderConfigInvalid)

	for _, bad := range []string{"acme/toy@main", "acme/toy", "acme/toy@" + strings.Repeat("g", 40), "toy@" + rev} {
		if _, err := ParseHFPin(bad); err == nil {
			t.Fatalf("ParseHFPin(%q) accepted an unpinned or malformed ref", bad)
		}
	}

	good := shardedHub(t, "acme/toy", rev, cfg, tensors, func(string) string { return "s.safetensors" }, 0)
	src, closeSrv := good.source(t)
	defer closeSrv()
	c, err := CensusHF(context.Background(), src, "acme/toy", rev)
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range []string{"", "Bad-Name", "9lives", "has_underscore"} {
		if _, err := EmitHFSpine(c, family); err == nil {
			t.Fatalf("family %q accepted", family)
		}
	}
	spine, err := EmitHFSpine(c, "toy")
	if err != nil {
		t.Fatal(err)
	}
	parent := t.TempDir()
	out := filepath.Join(parent, "out")
	if err := WriteHFSpine(out, spine); err != nil {
		t.Fatal(err)
	}
	if err := WriteHFSpine(out, spine); err == nil {
		t.Fatal("rewriting an existing family fixture dir was accepted")
	}
	entries, err := os.ReadDir(parent)
	if err != nil || len(entries) != 1 || entries[0].Name() != "out" {
		t.Fatalf("emitter wrote outside --out: %v %v", entries, err)
	}
	written := treeBytes(t, out)
	paths := make([]string, 0, len(written))
	for p := range written {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	if !reflect.DeepEqual(paths, spine.Paths()) {
		t.Fatalf("written %v, want %v", paths, spine.Paths())
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestEmitHFSpineRefusesConfigIdentityOutsideSafeCharset(t *testing.T) {
	rev := strings.Repeat("d", 40)
	tensors := map[string]HFTensor{"a.weight": {Shape: []int{2}, DType: "F32"}}
	emit := func(cfg string) error {
		_, err := EmitHFSpine(&HFCensus{Repo: "acme/toy", Revision: rev, Config: []byte(cfg), Tensors: tensors}, "toy")
		return err
	}
	for _, cfg := range []string{
		`{"architectures":["ToyForCausalLM"],"model_type":"x\" + func() string { panic(\"pwn\") }() + \""}`,
		`{"architectures":["ToyForCausalLM"],"model_type":"toy%d"}`,
		`{"architectures":["ToyForCausalLM"],"model_type":"toy\n*/"}`,
		`{"architectures":["Toy\"ForCausalLM"],"model_type":"toy"}`,
		`{"architectures":["ToyForCausalLM"],"model_type":"toy","text_config":{"model_type":"toy text"}}`,
	} {
		wantRefusal(t, emit(cfg), RefusalHeaderConfigInvalid)
	}
	if err := emit(`{"architectures":["Toy_v1.5ForCausalLM"],"model_type":"toy-v1.5","text_config":{"model_type":"toy_text"}}`); err != nil {
		t.Fatalf("safe identity refused: %v", err)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestCensusHFMissingSingleFileShardDoesNotBlameAnIndex(t *testing.T) {
	rev := strings.Repeat("e", 40)
	hub := &fakeHub{repo: "acme/toy", rev: rev, files: map[string][]byte{
		"config.json": []byte(`{"architectures":["ToyForCausalLM"],"model_type":"toy"}`),
	}}
	src, closeSrv := hub.source(t)
	defer closeSrv()
	_, err := CensusHF(context.Background(), src, hub.repo, hub.rev)
	wantRefusal(t, err, RefusalTensorMissing)
	var r *Refusal
	errors.As(err, &r)
	if strings.Contains(r.Detail, "index") {
		t.Fatalf("single-file refusal blames an index that does not exist: %q", r.Detail)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCensusHFCensusesUnindexedSidecarLikeHandFixture(t *testing.T) {
	spine, hub := emitInkling2x(t)
	_, tensors, source := inkling2FixtureSource(t)
	base := "internal/model/testdata/inkling2x/"
	var inv struct {
		Source  string              `json:"source"`
		Tensors map[string]HFTensor `json:"tensors"`
	}
	if err := json.Unmarshal(spine.Files[base+"tensor_inventory.json"], &inv); err != nil {
		t.Fatal(err)
	}
	var missing []string
	for name, want := range tensors {
		if got, ok := inv.Tensors[name]; !ok || !reflect.DeepEqual(got, want) {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 || !reflect.DeepEqual(inv.Tensors, tensors) {
		t.Fatalf("inventory has %d tensors, fixture %d; absent or different: %v", len(inv.Tensors), len(tensors), missing)
	}
	if inv.Source != source {
		t.Fatalf("source = %q, want %q", inv.Source, source)
	}
	var prov struct {
		Shards         []HFShardHeader `json:"shards"`
		SidecarListing *int            `json:"sidecar_listing"`
	}
	if err := json.Unmarshal(spine.Files[base+"provenance.json"], &prov); err != nil {
		t.Fatal(err)
	}
	var files []string
	for _, sh := range prov.Shards {
		files = append(files, sh.File)
	}
	if want := []string{"model-00001-of-00001.safetensors", "mtp.safetensors"}; !reflect.DeepEqual(files, want) {
		t.Fatalf("provenance shards = %v, want %v", files, want)
	}
	if prov.SidecarListing != nil {
		t.Fatalf("sidecar_listing = %d recorded on a successful listing", *prov.SidecarListing)
	}
	if len(hub.unrangedGets) != 0 {
		t.Fatalf("unranged shard GETs: %v", hub.unrangedGets)
	}
	if len(hub.listingAuth) != 1 || hub.listingAuth[0] != "Bearer "+fakeHubToken {
		t.Fatalf("listing requests carried Authorization %q, want one bearer", hub.listingAuth)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCensusHFSidecarListingFailureIsRecordedNotFatal(t *testing.T) {
	hub := inkling2Hub(t)
	hub.listingStatus = http.StatusNotFound
	spine, _ := emitHFHub(t, hub, "inkling2x")
	base := "internal/model/testdata/inkling2x/"
	var prov struct {
		Shards         []HFShardHeader `json:"shards"`
		SidecarListing *int            `json:"sidecar_listing"`
	}
	if err := json.Unmarshal(spine.Files[base+"provenance.json"], &prov); err != nil {
		t.Fatal(err)
	}
	if prov.SidecarListing == nil || *prov.SidecarListing != http.StatusNotFound {
		t.Fatalf("sidecar_listing = %v, want %d", prov.SidecarListing, http.StatusNotFound)
	}
	if len(prov.Shards) != 1 || prov.Shards[0].File != "model-00001-of-00001.safetensors" {
		t.Fatalf("shards = %+v, want only the indexed shard", prov.Shards)
	}
	var inv struct {
		Source  string              `json:"source"`
		Tensors map[string]HFTensor `json:"tensors"`
	}
	if err := json.Unmarshal(spine.Files[base+"tensor_inventory.json"], &inv); err != nil {
		t.Fatal(err)
	}
	if want := inkling2Repo + "@" + inkling2Rev + " model-00001-of-00001.safetensors header"; inv.Source != want {
		t.Fatalf("source = %q, want %q", inv.Source, want)
	}
	for name := range inv.Tensors {
		if strings.HasPrefix(name, "mtp:") {
			t.Fatalf("sidecar tensor %q present without a listing", name)
		}
	}
}
