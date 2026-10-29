//go:build cuda

package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type denseWKVArtifact struct {
	Path   string
	Bytes  int64
	SHA256 string
}
type denseWKVPhysicalFixture struct {
	Schema, SourceCommit, ExecutableSHA256                                string
	ModelRevision, DeepInfer, DeepGEMM, Cutlass                           string
	WeightCodesSHA256, TileScalesSHA256                                   string
	ToleranceApproval, ReferenceCommand                                   string
	Config, Shard, InputF32, ExpectedBF16, ReferenceReceipt, BuildReceipt denseWKVArtifact
	Layer, Rows, Iterations, SMs                                          int
	AbsTolerance, RelTolerance                                            *float64
}

// denseWKVReadArtifact validates size and hashes before allocating bounded bytes.
// Hash-only shard/executable verification streams the file, not a shard buffer.
func denseWKVReadArtifact(t *testing.T, root string, a denseWKVArtifact, max int64, materialize bool) []byte {
	t.Helper()
	if a.Path == "" || a.Bytes <= 0 || a.Bytes > max || len(a.SHA256) != 64 {
		t.Fatal("missing or oversized physical artifact")
	}
	if _, err := hex.DecodeString(a.SHA256); err != nil {
		t.Fatal(err)
	}
	p := a.Path
	if !filepath.IsAbs(p) {
		p = filepath.Join(root, p)
	}
	f, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil || !st.Mode().IsRegular() || st.Size() != a.Bytes {
		t.Fatalf("artifact size/type mismatch: %s", p)
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, a.Bytes+1))
	if err != nil || n != a.Bytes || hex.EncodeToString(h.Sum(nil)) != a.SHA256 {
		t.Fatalf("artifact hash mismatch: %s", p)
	}
	if !materialize {
		return nil
	}
	if _, err := f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	b := make([]byte, int(a.Bytes))
	if _, err := io.ReadFull(f, b); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != a.SHA256 {
		t.Fatal("artifact changed during read")
	}
	return b
}

// TestV41DenseWKVPhysical is an explicit on-device witness, never an automatic
// hardware job. The operator supplies independently generated golden bytes and
// approved tolerances; this test does not generate its own expected answers.
// A cuda-only build remains legal: explicit opt-in then fails unless the
// arch-specific WKV image and physical device are actually available.
// fak-test:runtime slow est=30s lane=cuda
func TestV41DenseWKVPhysical(t *testing.T) {
	path := os.Getenv("FAK_WKV_PHYSICAL_MANIFEST")
	if path == "" {
		t.Skip("NOT_RUN: set FAK_WKV_PHYSICAL_MANIFEST on the authorized GPU owner")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	var m denseWKVPhysicalFixture
	d := json.NewDecoder(io.LimitReader(f, 1<<20))
	d.DisallowUnknownFields()
	err = d.Decode(&m)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if m.Schema != "fak.wkv-physical-fixture/v1" || m.ModelRevision != DeepSeekV41FlashRevision || m.DeepInfer != "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01" || m.DeepGEMM != "1fc9c66975ee322d628b3288534ca0e9ce12a64d" || m.Cutlass != "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8" {
		t.Fatal("physical fixture source/model pin mismatch")
	}
	if len(m.SourceCommit) != 40 || m.SourceCommit != os.Getenv("FAK_WKV_SOURCE_COMMIT") || m.ToleranceApproval == "" || m.ReferenceCommand == "" {
		t.Fatal("missing source binding or external reference/tolerance approval")
	}
	if _, err := hex.DecodeString(m.SourceCommit); err != nil {
		t.Fatal(err)
	}
	gitRead := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"--no-optional-locks"}, args...)...)
		cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0", "GIT_NO_LAZY_FETCH=1", "GIT_TERMINAL_PROMPT=0")
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("actual source checkout witness: %v", err)
		}
		return strings.TrimSpace(string(out))
	}
	if gitRead("rev-parse", "HEAD") != m.SourceCommit || gitRead("status", "--porcelain", "--untracked-files=no") != "" {
		t.Fatal("physical witness requires pinned clean tracked source checkout")
	}
	if m.Rows < 128 || m.Rows > 4096 || m.Rows%128 != 0 || m.Iterations < 2 || m.Iterations > 100 || m.Layer < 0 || m.SMs <= 0 {
		t.Fatal("physical fixture shape/iteration/SM budget")
	}
	for _, v := range []*float64{m.AbsTolerance, m.RelTolerance} {
		if v == nil || *v < 0 || math.IsNaN(*v) || math.IsInf(*v, 0) {
			t.Fatal("finite explicit tolerances required")
		}
	}
	root := filepath.Dir(path)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(exe)
	if err != nil {
		t.Fatal(err)
	}
	denseWKVReadArtifact(t, root, denseWKVArtifact{Path: exe, Bytes: st.Size(), SHA256: m.ExecutableSHA256}, 1<<30, false)
	config := denseWKVReadArtifact(t, root, m.Config, 1<<20, true)
	denseWKVReadArtifact(t, root, m.ReferenceReceipt, 4<<20, false)
	denseWKVReadArtifact(t, root, m.BuildReceipt, 4<<20, false)
	denseWKVReadArtifact(t, root, m.Shard, 64<<30, false)
	var cfg Config
	if err := json.Unmarshal(config, &cfg); err != nil {
		t.Fatal(err)
	}
	shardPath := m.Shard.Path
	if !filepath.IsAbs(shardPath) {
		shardPath = filepath.Join(root, shardPath)
	}
	shard, err := OpenV41PackedShard(shardPath, cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := shard.ReadV41DenseWKV(m.Layer, V41DenseWKVPayloadBytes)
	closeErr := shard.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	for _, pair := range []struct {
		b    []byte
		want string
	}{{w.codes, m.WeightCodesSHA256}, {w.scales, m.TileScalesSHA256}} {
		sum := sha256.Sum256(pair.b)
		if hex.EncodeToString(sum[:]) != pair.want {
			t.Fatal("actual retained weight/scale hash mismatch")
		}
	}
	inputRaw := denseWKVReadArtifact(t, root, m.InputF32, int64(m.Rows)*5120*4, true)
	expectedRaw := denseWKVReadArtifact(t, root, m.ExpectedBF16, int64(m.Rows)*512*2, true)
	if len(inputRaw) != m.Rows*5120*4 || len(expectedRaw) != m.Rows*512*2 {
		t.Fatal("physical fixture exact payload sizes required")
	}
	x := make([]float32, m.Rows*5120)
	want := make([]float32, m.Rows*512)
	for i := range x {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(inputRaw[4*i:]))
		if math.IsNaN(float64(x[i])) || math.IsInf(float64(x[i]), 0) {
			t.Fatal("nonfinite input")
		}
	}
	for i := range want {
		want[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(expectedRaw[2*i:])) << 16)
		if math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
			t.Fatal("nonfinite golden")
		}
	}
	be := compute.Pick("cuda")
	b, ok := be.(compute.V41FP8WKVBackend)
	if !ok || !b.V41FP8WKVSupported() {
		t.Fatal("requested WKV native device/kernel unavailable")
	}
	if b.V41FP8WKVQualified() {
		t.Fatal("witness must not silently enable ordinary qualification")
	}
	factsSource, ok := be.(interface {
		V41FP8WKVDeviceFacts() (map[string]any, error)
	})
	if !ok {
		t.Fatal("actual device facts unavailable")
	}
	facts, err := factsSource.V41FP8WKVDeviceFacts()
	if err != nil {
		t.Fatal(err)
	}
	// Decode through JSON to accept integer representations without guessing
	// architecture, runtime version or SM count from a backend label.
	factsBytes, err := json.Marshal(facts)
	if err != nil {
		t.Fatal(err)
	}
	var factsNumbers map[string]any
	if err := json.Unmarshal(factsBytes, &factsNumbers); err != nil {
		t.Fatal(err)
	}
	if factsNumbers["sms"] != float64(m.SMs) || factsNumbers["compiled_sms"] != float64(m.SMs) {
		t.Fatal("actual/compiled SM count mismatch")
	}
	name, _ := factsNumbers["name"].(string)
	if factsNumbers["major"] != float64(10) || factsNumbers["minor"] != float64(3) || !strings.Contains(strings.ToUpper(name), "B300") {
		t.Fatal("this physical fixture requires actual B300 sm103; other devices need a separately scoped witness")
	}
	p, err := w.BindForWitness(b)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := p.Release(); err != nil {
			t.Error(err)
		}
	}()
	input := be.Upload(compute.NewF32(be, []int{m.Rows, 5120}, x), compute.F32)
	inputFreed := false
	defer func() {
		if !inputFreed {
			be.Free(input)
		}
	}()
	start := time.Now()
	maxAbs := 0.0
	for iteration := 0; iteration < m.Iterations; iteration++ {
		func() {
			out, err := p.Project(input, m.Rows)
			if err != nil {
				t.Fatal(err)
			}
			defer be.Free(out)
			if out.Dtype != compute.BF16 || out.Layout != compute.RowMajor || !sameShape(out.Shape, []int{m.Rows, 512}) {
				t.Fatal("native output layout mismatch")
			}
			got := be.Read(out)
			if len(got) != len(want) {
				t.Fatal("native checked read length mismatch")
			}
			for i, v := range got {
				delta := math.Abs(float64(v) - float64(want[i]))
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || delta > *m.AbsTolerance+*m.RelTolerance*math.Abs(float64(want[i])) {
					t.Fatalf("iteration %d output %d outside approved tolerance", iteration, i)
				}
				maxAbs = math.Max(maxAbs, delta)
			}
		}()
	}
	if err := p.Release(); err != nil {
		t.Fatal(err)
	}
	be.Free(input)
	inputFreed = true
	t.Logf("WKV_PROJECTION_COMPLETED source=%s executable=%s manifest=%s device=%s rows=%d attempted=%d returned=%d completed=%d max_abs=%g end_to_end=%s; B300 projection witness only, not full-model qualification or token/s", m.SourceCommit, m.ExecutableSHA256, strings.TrimSpace(path), factsBytes, m.Rows, m.Iterations, m.Iterations, m.Iterations, maxAbs, time.Since(start))
}
