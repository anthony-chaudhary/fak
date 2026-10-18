package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms lane=default
func TestV41DSparkBundledInventoryAndPackedReader(t *testing.T) {
	_, published := readDeepSeekV41Config(t)
	inv, err := DeepSeekV41DSparkInventory(published)
	if err != nil {
		t.Fatal(err)
	}
	for name, shape := range map[string][]int{
		"mtp.0.main_proj.weight":            {5120, 15360},
		"mtp.0.attn.wo_a.weight":            {8192, 4096},
		"mtp.1.ffn.gate.weight":             {128, 5120},
		"mtp.1.ffn.experts.127.w1.weight":   {2304, 2560},
		"mtp.2.markov_head.embed.weight":    {129280, 256},
		"mtp.2.markov_head.head.weight":     {129280, 256},
		"mtp.2.confidence_head.proj.weight": {1, 5376},
		"mtp.2.hc_ffn_fn":                   {24, 20480},
	} {
		spec, ok := inv.Lookup(name)
		if !ok || !sameShape(spec.Shape, shape) {
			t.Fatalf("%s = %+v, present=%v, want shape %v", name, spec, ok, shape)
		}
	}
	for _, name := range []string{"layers.0.attn.wq_a.weight", "mtp.0.markov_head.embed.weight", "mtp.3.norm.weight", "mtp.2.hc_head_fn"} {
		if _, ok := inv.Lookup(name); ok {
			t.Fatalf("foreign/incorrect-stage tensor admitted: %s", name)
		}
	}
	target, err := DeepSeekV41Inventory(published)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := target.Lookup("mtp.0.main_proj.weight"); ok {
		t.Fatal("draft inventory changed the default target inventory")
	}
	if !errors.Is(published.RefuseDeepSeekV41DSpark(), ErrV41DSparkUnsupported) {
		t.Fatal("metadata loader accidentally enabled DSpark execution")
	}

	// Small real safetensors fixture: no model execution and no external artifact.
	cfg := v41TestReducedConfig(t, 3, 2)
	cfg.DeepSeekV41.DSpark.TargetLayerIDs = []int{0, 1, 2}
	cfg.DeepSeekV41.DSpark.NumRoutedExperts = 2
	cfg.DeepSeekV41.DSpark.NumExpertsPerTok = 1
	cfg.DeepSeekV41.DSpark.MarkovRank = 4
	cfg.DeepSeekV41.DSpark.NoiseTokenID = 7
	weightName := "model.mtp.1.mlp.experts.1.w1.weight"
	scaleName := "model.mtp.1.mlp.experts.1.w1.weight_scale_inv"
	weight := v41TestPattern(32*32, 11)
	scale := v41TestPattern(32*2, 127)
	fixture := func() map[string]tinySTTensor {
		return map[string]tinySTTensor{
			weightName:                          {dtype: "I8", shape: []int{32, 32}, data: weight},
			scaleName:                           {dtype: "F8_E8M0", shape: []int{32, 2}, data: scale},
			"mtp.2.confidence_head.proj.weight": {dtype: "BF16", shape: []int{1, 68}, data: make([]byte, 136)},
			"model.embed_tokens.weight":         {dtype: "BF16", shape: []int{8, 64}, data: make([]byte, 8*64*2)},
		}
	}
	path := writeTinySafetensors(t, fixture())
	shard, err := OpenV41DSparkPackedShard(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer shard.Close()
	ref, err := shard.Reference("mtp.1.ffn.experts.1.w1.weight")
	if err != nil || ref.Ref.Name != weightName || ref.Ref.ScaleName != scaleName || !ref.Ref.PackedWeight {
		t.Fatalf("original artifact identity was not retained: %+v, %v", ref, err)
	}
	if got, err := shard.Read("mtp.1.ffn.experts.1.w1.weight", int64(len(weight))); err != nil || !bytes.Equal(got, weight) {
		t.Fatalf("packed E2M1 payload changed: %v", err)
	}
	if got, err := shard.ReadScale("mtp.1.ffn.experts.1.w1.weight", int64(len(scale))); err != nil || !bytes.Equal(got, scale) {
		t.Fatalf("E8M0 exponent bytes changed: %v", err)
	}
	for _, bound := range []int64{0, -1, int64(len(weight) - 1)} {
		if _, err := shard.Read("mtp.1.ffn.experts.1.w1.weight", bound); !errors.Is(err, ErrV41DSparkArtifact) {
			t.Fatalf("read bound %d did not fail closed: %v", bound, err)
		}
	}
	if _, err := shard.Reference("model.embed_tokens.weight"); !errors.Is(err, ErrV41DSparkArtifact) {
		t.Fatalf("draft reader exposed target-owned weights: %v", err)
	}
	if len(shard.Missing()) == 0 {
		t.Fatal("a partial shard was presented as a complete drafter")
	}

	for _, tc := range []struct {
		name string
		edit func(map[string]tinySTTensor)
	}{
		{"missing scale", func(m map[string]tinySTTensor) { delete(m, scaleName) }},
		{"NVFP4 scale", func(m map[string]tinySTTensor) { x := m[scaleName]; x.dtype = "F8_E4M3"; m[scaleName] = x }},
		{"NVFP4 group16", func(m map[string]tinySTTensor) {
			x := m[scaleName]
			x.shape = []int{32, 4}
			x.data = make([]byte, 128)
			m[scaleName] = x
		}},
		{"ambiguous alias", func(m map[string]tinySTTensor) { m["mtp.1.ffn.experts.1.w1.weight"] = m[weightName] }},
		{"unknown DSpark tensor", func(m map[string]tinySTTensor) {
			m["mtp.2.hc_head_fn"] = tinySTTensor{dtype: "F32", shape: []int{1}, data: make([]byte, 4)}
		}},
		{"wrong final stage", func(m map[string]tinySTTensor) {
			m["mtp.0.confidence_head.proj.weight"] = m["mtp.2.confidence_head.proj.weight"]
			delete(m, "mtp.2.confidence_head.proj.weight")
		}},
		{"short payload", func(m map[string]tinySTTensor) {
			x := m[weightName]
			x.data = x.data[:len(x.data)-1]
			m[weightName] = x
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := fixture()
			tc.edit(m)
			s, err := OpenV41DSparkPackedShard(writeTinySafetensors(t, m), cfg)
			if s != nil {
				_ = s.Close()
			}
			if !errors.Is(err, ErrV41DSparkArtifact) {
				t.Fatalf("malformed draft artifact accepted: %v", err)
			}
		})
	}
	if err := shard.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := shard.Read("mtp.1.ffn.experts.1.w1.weight", 1024); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("read after close = %v", err)
	}
}

type v41DSparkTestReaderAt func([]byte, int64) (int, error)

func (f v41DSparkTestReaderAt) ReadAt(p []byte, off int64) (int, error) { return f(p, off) }

type v41DSparkTestCloser struct {
	calls atomic.Int32
	err   error
}

func (c *v41DSparkTestCloser) Close() error {
	c.calls.Add(1)
	return c.err
}

// fak-test:runtime fast est=60ms lane=default
func TestV41DSparkPackedReaderReviewRegressions(t *testing.T) {
	cfg := v41TestReducedConfig(t, 3, 2)
	cfg.DeepSeekV41.DSpark.TargetLayerIDs = []int{0, 1, 2}
	cfg.DeepSeekV41.DSpark.NumRoutedExperts = 2
	cfg.DeepSeekV41.DSpark.NumExpertsPerTok = 1
	cfg.DeepSeekV41.DSpark.MarkovRank = 4
	cfg.DeepSeekV41.DSpark.NoiseTokenID = 7
	inv, err := DeepSeekV41DSparkInventory(cfg)
	if err != nil {
		t.Fatal(err)
	}
	const weightName = "mtp.1.ffn.experts.1.w1.weight"
	const scaleName = "mtp.1.ffn.experts.1.w1.scale"
	fixture := func() map[string]tinySTTensor {
		return map[string]tinySTTensor{
			weightName:                  {dtype: "I8", shape: []int{32, 32}, data: v41TestPattern(1024, 11)},
			scaleName:                   {dtype: "F8_E8M0", shape: []int{32, 2}, data: v41TestPattern(64, 127)},
			"model.embed_tokens.weight": {dtype: "BF16", shape: []int{8, 64}, data: make([]byte, 1024)},
			"model.ignored.weight":      {dtype: "F32", shape: []int{16}, data: make([]byte, 64)},
		}
	}
	openReader := func(r io.ReaderAt, size int64, closer io.Closer) (*V41DSparkPackedShard, error) {
		sf, err := readV41DSparkHeader(r, size, closer)
		if err != nil {
			return nil, err
		}
		s, err := bindV41DSparkPackedShard("fixture.safetensors", inv, sf)
		if err != nil {
			_ = sf.Close()
		}
		return s, err
	}
	file := tinySafetensorsBytes(t, fixture())
	dataBase := 8 + int64(binary.LittleEndian.Uint64(file[:8]))

	t.Run("length prefix is checked and parsed once", func(t *testing.T) {
		prefixReads, headerReads, payloadReads := 0, 0, 0
		reader := v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
			switch off {
			case 0:
				prefixReads++
				if prefixReads > 1 {
					// The old check-then-reparse opener would see a changed,
					// oversized length on its second prefix read.
					binary.LittleEndian.PutUint64(p, v41DSparkMaxHeaderBytes+1)
					return len(p), nil
				}
			case 8:
				headerReads++
				if len(p) != int(dataBase-8) {
					t.Fatalf("header read size %d, checked %d", len(p), dataBase-8)
				}
			default:
				payloadReads++
			}
			return bytes.NewReader(file).ReadAt(p, off)
		})
		s, err := openReader(reader, int64(len(file)), nil)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if prefixReads != 1 || headerReads != 1 || payloadReads != 0 {
			t.Fatalf("reads prefix/header/payload = %d/%d/%d", prefixReads, headerReads, payloadReads)
		}
	})

	t.Run("oversized prefix refuses before header allocation", func(t *testing.T) {
		reads := 0
		reader := v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
			reads++
			if off != 0 || len(p) != binary.Size(uint64(0)) {
				t.Fatalf("read past prefix budget: off=%d len=%d", off, len(p))
			}
			binary.LittleEndian.PutUint64(p, v41DSparkMaxHeaderBytes+1)
			return 8, nil
		})
		closer := &v41DSparkTestCloser{}
		if s, err := openReader(reader, v41DSparkMaxHeaderBytes+9, closer); s != nil || !errors.Is(err, ErrV41DSparkArtifact) || reads != 1 || closer.calls.Load() != 1 {
			t.Fatalf("oversized prefix: shard=%v err=%v reads=%d closes=%d", s, err, reads, closer.calls.Load())
		}
	})

	t.Run("exact target owned aliases are excluded", func(t *testing.T) {
		m := fixture()
		aliases := []string{"mtp.0.embed.weight", "mtp.1.embed.weight", "mtp.2.embed.weight", "model.mtp.0.head.weight", "model.mtp.1.head.weight", "model.mtp.2.head.weight"}
		for _, name := range aliases {
			m[name] = tinySTTensor{dtype: "BF16", shape: []int{8, 64}, data: make([]byte, 1024)}
		}
		blob := tinySafetensorsBytes(t, m)
		s, err := openReader(bytes.NewReader(blob), int64(len(blob)), nil)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if len(s.sources) != 1 {
			t.Fatalf("target alias counted as draft weight: %v", s.sources)
		}
		for _, name := range aliases {
			if _, err := s.Reference(name); !errors.Is(err, ErrV41DSparkArtifact) {
				t.Fatalf("excluded alias exposed: %s, %v", name, err)
			}
		}
		for _, name := range []string{"mtp.3.embed.weight", "mtp.00.head.weight", "mtp.0.extra.head.weight"} {
			bad := fixture()
			bad[name] = m[aliases[0]]
			blob := tinySafetensorsBytes(t, bad)
			if s, err := openReader(bytes.NewReader(blob), int64(len(blob)), nil); s != nil || !errors.Is(err, ErrV41DSparkArtifact) {
				t.Fatalf("non-exact alias admitted: %s, %v", name, err)
			}
		}
	})

	t.Run("all shard spans including ignored target cannot overlap", func(t *testing.T) {
		// The ordinary mixed shard is the positive control.
		s, err := openReader(bytes.NewReader(file), int64(len(file)), nil)
		if err != nil {
			t.Fatal(err)
		}
		_ = s.Close()
		for _, overlap := range [][2]string{
			{"model.embed_tokens.weight", weightName},
			{"model.ignored.weight", "model.embed_tokens.weight"},
		} {
			hdr, base, err := parseSafetensorsHeader(file)
			if err != nil {
				t.Fatal(err)
			}
			var dst, src stEntry
			if err := json.Unmarshal(hdr[overlap[0]], &dst); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(hdr[overlap[1]], &src); err != nil {
				t.Fatal(err)
			}
			width := dst.DataOffsets[1] - dst.DataOffsets[0]
			dst.DataOffsets = []int{src.DataOffsets[0], src.DataOffsets[0] + width}
			hdr[overlap[0]], err = json.Marshal(dst)
			if err != nil {
				t.Fatal(err)
			}
			header, err := json.Marshal(hdr)
			if err != nil {
				t.Fatal(err)
			}
			blob := make([]byte, 8)
			binary.LittleEndian.PutUint64(blob, uint64(len(header)))
			blob = append(blob, header...)
			blob = append(blob, file[base:]...)
			if s, err := openReader(bytes.NewReader(blob), int64(len(blob)), nil); s != nil || !errors.Is(err, ErrV41DSparkArtifact) || !strings.Contains(err.Error(), "overlapping") {
				t.Fatalf("mixed overlap %v admitted: %v", overlap, err)
			}
		}
	})

	for _, scale := range []bool{false, true} {
		name := "weight"
		if scale {
			name = "scale"
		}
		t.Run(name+" actual partial ReadAt error", func(t *testing.T) {
			injected := errors.New("fixture partial payload I/O")
			payloadReads := 0
			reader := v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
				if off >= dataBase {
					payloadReads++
					n, _ := bytes.NewReader(file).ReadAt(p[:len(p)/2], off)
					return n, injected
				}
				return bytes.NewReader(file).ReadAt(p, off)
			})
			s, err := openReader(reader, int64(len(file)), nil)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			var got []byte
			if scale {
				got, err = s.ReadScale(weightName, 64)
			} else {
				got, err = s.Read(weightName, 1024)
			}
			if got != nil || !errors.Is(err, injected) || payloadReads != 1 {
				t.Fatalf("partial payload exposed: bytes=%d err=%v reads=%d", len(got), err, payloadReads)
			}
		})

		t.Run(name+" Close waits for blocked payload read", func(t *testing.T) {
			entered, release := make(chan struct{}), make(chan struct{})
			var releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(release) }) }
			reader := v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
				if off >= dataBase {
					close(entered)
					<-release
				}
				return bytes.NewReader(file).ReadAt(p, off)
			})
			closeError := errors.New("fixture close failure")
			closer := &v41DSparkTestCloser{err: closeError}
			s, err := openReader(reader, int64(len(file)), closer)
			if err != nil {
				t.Fatal(err)
			}
			// Cleanup must release a blocked read before waiting on Close.
			t.Cleanup(func() { _ = s.Close() })
			t.Cleanup(unblock)
			readDone, closeDone := make(chan error, 1), make(chan error, 1)
			go func() {
				if scale {
					_, err := s.ReadScale(weightName, 64)
					readDone <- err
				} else {
					_, err := s.Read(weightName, 1024)
					readDone <- err
				}
			}()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("payload read did not enter")
			}
			// Deterministically witness the lifetime lock while ReadAt is blocked.
			if s.mu.TryLock() {
				s.mu.Unlock()
				t.Fatal("in-flight payload read does not hold the lifetime lock")
			}
			go func() { closeDone <- s.Close() }()
			select {
			case err := <-closeDone:
				t.Fatalf("Close passed blocked payload read: %v", err)
			case <-time.After(10 * time.Millisecond):
			}
			if closer.calls.Load() != 0 {
				t.Fatal("underlying file closed during read")
			}
			unblock()
			select {
			case err := <-readDone:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(time.Second):
				t.Fatal("payload read did not complete")
			}
			select {
			case err := <-closeDone:
				if err != closeError {
					t.Fatalf("first Close lost error identity: %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("Close did not complete")
			}
			if err := s.Close(); err != closeError || closer.calls.Load() != 1 {
				t.Fatalf("repeated Close = %v, underlying calls = %d", err, closer.calls.Load())
			}
			if _, err := s.Reference(weightName); !errors.Is(err, ErrV41DSparkArtifact) {
				t.Fatalf("Reference after Close = %v", err)
			}
		})
	}
}
