package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

type pdbenchGGUFTensor struct {
	name   string
	dims   []uint64
	data   []byte
	offset uint64
}

func pdbenchTinyDenseGGUF(t *testing.T) string {
	t.Helper()
	const dim, vocab = 32, 8
	var tensors []pdbenchGGUFTensor
	addF32 := func(name string, dims ...uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		data := make([]byte, n*4)
		for i := range n {
			v := float32((i*7)%19-9) / 128
			if len(dims) == 1 {
				v = 1
			}
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
		}
		tensors = append(tensors, pdbenchGGUFTensor{name: name, dims: dims, data: data})
	}
	addF32("token_embd.weight", dim, vocab)
	addF32("output.weight", dim, vocab)
	for _, name := range []string{"output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"} {
		addF32(name, dim)
	}
	for _, name := range []string{
		"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
	} {
		addF32(name, dim, dim)
	}

	var b bytes.Buffer
	write := func(v any) {
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	str := func(s string) {
		write(uint64(len(s)))
		_, _ = b.WriteString(s)
	}
	write(uint32(0x46554747))
	write(uint32(3))
	write(uint64(len(tensors)))
	write(uint64(12))
	str("general.architecture")
	write(uint32(ggufload.TypeString))
	str("qwen2")
	for _, kv := range []struct {
		key string
		val uint32
	}{
		{"general.alignment", 32}, {"qwen2.context_length", 16}, {"qwen2.embedding_length", dim},
		{"qwen2.block_count", 1}, {"qwen2.feed_forward_length", dim}, {"qwen2.attention.head_count", 1},
		{"qwen2.attention.head_count_kv", 1}, {"qwen2.rope.dimension_count", dim},
	} {
		str(kv.key)
		write(uint32(ggufload.TypeUint32))
		write(kv.val)
	}
	for _, kv := range []struct {
		key string
		val float32
	}{{"qwen2.attention.layer_norm_rms_epsilon", 1e-5}, {"qwen2.rope.freq_base", 10000}} {
		str(kv.key)
		write(uint32(ggufload.TypeFloat32))
		write(kv.val)
	}
	str("tokenizer.ggml.tokens")
	write(uint32(ggufload.TypeArray))
	write(uint32(ggufload.TypeString))
	write(uint64(vocab))
	for i := range vocab {
		str(string(rune('a' + i)))
	}
	var offset uint64
	for i := range tensors {
		tensor := &tensors[i]
		tensor.offset = offset
		str(tensor.name)
		write(uint32(len(tensor.dims)))
		for _, d := range tensor.dims {
			write(d)
		}
		write(uint32(ggufload.TensorF32))
		write(offset)
		offset = (offset + uint64(len(tensor.data)) + 31) &^ 31
	}
	for b.Len()%32 != 0 {
		_ = b.WriteByte(0)
	}
	dataStart := b.Len()
	for _, tensor := range tensors {
		for b.Len() < dataStart+int(tensor.offset) {
			_ = b.WriteByte(0)
		}
		_, _ = b.Write(tensor.data)
	}
	path := filepath.Join(t.TempDir(), "pdbench-tiny-qwen2.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatalf("write tiny GGUF: %v", err)
	}
	return path
}

func pdbenchHybridFixtureModel() *model.Model {
	cfg := modelengine.SyntheticConfig()
	cfg.NumLayers = 4
	cfg.LayerTypes = []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"}
	cfg.FullAttentionInterval = 4
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = cfg.HeadDim
	cfg.LinearValueHeadDim = cfg.HeadDim
	cfg.LinearNumKeyHeads = cfg.NumKVHeads
	cfg.LinearNumValueHeads = cfg.NumHeads
	cfg.AttnOutputGate = true
	cfg.NormGain1p = true
	m := model.NewSynthetic(cfg)
	m.Quantize()
	return m
}

func pdbenchDenseFixtureModel() *model.Model {
	m := model.NewSynthetic(modelengine.SyntheticConfig())
	m.Quantize()
	return m
}

func pdbenchCPUBackend(t *testing.T) compute.Backend {
	t.Helper()
	be, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend is not registered")
	}
	return be
}

func pdbenchRunConfig(t *testing.T, m *model.Model, prompt []int, steps int) runConfig {
	t.Helper()
	configID, err := identifyConfig(m.Cfg)
	if err != nil {
		t.Fatalf("identify config: %v", err)
	}
	be := pdbenchCPUBackend(t)
	backendID := identifyBackend(be)
	return runConfig{
		artifact:       artifactIdentity{Bytes: 4096, RawSHA256: strings.Repeat("a", 64), QuantFormat: backendID.QuantFormat},
		backend:        backendID,
		configID:       configID,
		expectedPrompt: append([]int(nil), prompt...),
		expectedSteps:  steps,
	}
}

func pdbenchGreedyContinuation(m *model.Model, prompt []int, steps int) []int {
	s := m.NewSession()
	defer s.Close()
	logits := s.Prefill(prompt)
	out := make([]int, 0, steps)
	for range steps {
		next := pdbenchArgmax(logits)
		out = append(out, next)
		logits = s.Step(next)
	}
	return out
}

func pdbenchArgmax(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

func pdbenchLoopbackPair(t *testing.T) (net.Conn, net.Conn) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen loopback: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	type accepted struct {
		conn net.Conn
		err  error
	}
	ch := make(chan accepted, 1)
	go func() {
		conn, acceptErr := ln.Accept()
		ch <- accepted{conn: conn, err: acceptErr}
	}()
	client, err := net.DialTimeout("tcp", ln.Addr().String(), time.Second)
	if err != nil {
		t.Fatalf("dial loopback: %v", err)
	}
	got := <-ch
	if got.err != nil {
		_ = client.Close()
		t.Fatalf("accept loopback: %v", got.err)
	}
	t.Cleanup(func() { _ = client.Close() })
	t.Cleanup(func() { _ = got.conn.Close() })
	return client, got.conn
}

// fak-test:runtime fast est=1s lane=short
func TestFrameRoundTripUsesRealLoopbackTCP(t *testing.T) {
	client, server := pdbenchLoopbackPair(t)
	payload := bytes.Repeat([]byte("complete-hybrid-state"), 17)
	errCh := make(chan error, 1)
	go func() {
		errCh <- writeFrame(client, payload, 4096)
	}()
	got, err := readFrame(server, 4096)
	if err != nil {
		t.Fatalf("read frame: %v", err)
	}
	if err := <-errCh; err != nil {
		t.Fatalf("write frame: %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("loopback frame changed: got %d bytes, want %d", len(got), len(payload))
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestFrameRefusesOversizeAndTruncation(t *testing.T) {
	if err := writeFrame(io.Discard, bytes.Repeat([]byte{1}, 65), 64); err == nil {
		t.Fatal("oversize frame was accepted")
	}
	if got, err := readFrame(bytes.NewReader([]byte{0, 0, 0}), 64); err == nil || got != nil {
		t.Fatalf("truncated header got payload=%x err=%v", got, err)
	}
}

type pdbenchShortWriter struct{}

func (pdbenchShortWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	return len(p) - 1, nil
}

// fak-test:runtime fast est=100ms lane=default
func TestWriteFrameRefusesShortNilWrite(t *testing.T) {
	if err := writeFrame(pdbenchShortWriter{}, []byte("payload"), 64); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short nil write error=%v, want io.ErrShortWrite", err)
	}
}

// fak-test:runtime fast est=1s lane=short
func TestReadFrameHonorsConnectionDeadline(t *testing.T) {
	client, server := pdbenchLoopbackPair(t)
	if err := server.SetReadDeadline(time.Now().Add(50 * time.Millisecond)); err != nil {
		t.Fatalf("set deadline: %v", err)
	}
	started := time.Now()
	got, err := readFrame(server, 64)
	if err == nil || got != nil {
		t.Fatalf("stalled peer got payload=%x err=%v", got, err)
	}
	var netErr net.Error
	if !errors.As(err, &netErr) || !netErr.Timeout() {
		t.Fatalf("stalled peer error=%v, want timeout", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("deadline returned after %v, want bounded", elapsed)
	}
	if err := client.Close(); err != nil && !errors.Is(err, net.ErrClosed) {
		t.Fatalf("close client: %v", err)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestHybridFixtureRequiresRecurrentStateForExactContinuation(t *testing.T) {
	m := pdbenchHybridFixtureModel()
	prompt := []int{3, 7, 11, 5, 17, 19}
	control := pdbenchGreedyContinuation(m, prompt, 4)

	prefilled := m.NewSession()
	logits := append([]float32(nil), prefilled.Prefill(prompt)...)
	blob, err := model.QwenHybridKVCacheToHost(prefilled.Cache, 4)
	prefilled.Close()
	if err != nil {
		t.Fatalf("serialize complete hybrid cache: %v", err)
	}
	restored, err := model.QwenHybridKVCacheFromHost(m.Cfg, blob)
	if err != nil {
		t.Fatalf("restore complete hybrid cache: %v", err)
	}
	consumer := m.NewSession()
	defer consumer.Close()
	consumer.Cache = restored
	if _, err := consumer.RestoreTokenLineage(prompt); err != nil {
		t.Fatalf("restore prompt lineage: %v", err)
	}
	got := make([]int, 0, len(control))
	for range control {
		next := pdbenchArgmax(logits)
		got = append(got, next)
		logits = consumer.Step(next)
	}
	if !reflect.DeepEqual(got, control) {
		t.Fatalf("restored continuation=%v, serial=%v", got, control)
	}
}

// fak-test:runtime fast est=2s lane=short
func TestProducerConsumerRestoresStateOverLoopbackTCP(t *testing.T) {
	producerModel := pdbenchDenseFixtureModel()
	consumerModel := pdbenchDenseFixtureModel()
	be := pdbenchCPUBackend(t)
	prompt := []int{3, 7, 11, 5, 17, 19}
	const steps = 5
	cfg := pdbenchRunConfig(t, producerModel, prompt, steps)
	producerConn, consumerConn := pdbenchLoopbackPair(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	setDeadline(producerConn, ctx)
	setDeadline(consumerConn, ctx)
	type result struct {
		receipt receipt
		err     error
	}
	consumerDone := make(chan result, 1)
	go func() {
		r, err := runConsumer(ctx, cfg, consumerModel, be, consumerConn)
		consumerDone <- result{receipt: r, err: err}
	}()
	producerReceipt, err := runProducer(ctx, cfg, producerModel, be, producerConn, prompt, steps)
	if err != nil {
		t.Fatalf("producer: %v", err)
	}
	consumerResult := <-consumerDone
	if consumerResult.err != nil {
		t.Fatalf("consumer: %v", consumerResult.err)
	}
	want := pdbenchGreedyContinuation(pdbenchDenseFixtureModel(), prompt, steps)
	if !producerReceipt.Parity || !consumerResult.receipt.Parity ||
		!reflect.DeepEqual(producerReceipt.ReferenceTokens, want) ||
		!reflect.DeepEqual(producerReceipt.ConsumerTokens, want) {
		t.Fatalf("TCP handoff producer=%+v consumer=%+v independent=%v", producerReceipt, consumerResult.receipt, want)
	}
	for label, got := range map[string]receipt{"producer": producerReceipt, "consumer": consumerResult.receipt} {
		if got.EvidenceScope != "software-only" || got.PerformanceClaim {
			t.Fatalf("%s mislabeled CPU receipt: scope=%q performance=%t", label, got.EvidenceScope, got.PerformanceClaim)
		}
		if got.SnapshotBytes == 0 || got.LogitsBytes == 0 || got.WireBytes == 0 {
			t.Fatalf("%s omitted transferred state: %+v", label, got)
		}
	}
}

// fak-test:runtime fast est=1s lane=short
func TestCLIFlagsLoadCPUFixtureAsF32AndReachPrefill(t *testing.T) {
	path := pdbenchTinyDenseGGUF(t)
	cfg, err := parseFlags([]string{
		"-mode", "producer", "-gguf", path, "-backend", "cpu-ref", "-software-only",
		"-address", "127.0.0.1:19191", "-prompt-ids", "1,2,3", "-steps", "3",
	})
	if err != nil {
		t.Fatalf("parse CLI flags: %v", err)
	}
	be, ok := compute.Lookup(cfg.backend)
	if !ok {
		t.Fatalf("CLI backend %q is not registered", cfg.backend)
	}
	if err := validateBackend(be, cfg.softwareOnly); err != nil {
		t.Fatalf("validate CLI backend: %v", err)
	}
	m, quantFormat, err := loadModelForBackend(cfg.gguf, be)
	if err != nil {
		t.Fatalf("backend-aware CLI load: %v", err)
	}
	defer m.CloseWeights()
	if quantFormat != "f32-dequant-on-load" {
		t.Fatalf("CPU load format=%q, want f32-dequant-on-load", quantFormat)
	}
	identity := artifactIdentity{QuantFormat: quantFormat}
	header, snapshot, logits, err := buildHandoff(context.Background(), m, be, cfg.prompt, cfg.steps, identity)
	if err != nil {
		t.Fatalf("real loaded model did not reach prefill/snapshot: %v", err)
	}
	if header.Artifact.QuantFormat != quantFormat || len(snapshot) == 0 || len(logits) == 0 {
		t.Fatalf("loaded handoff omitted format/state: header=%+v snapshot=%d logits=%d", header.Artifact, len(snapshot), len(logits))
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestBuildHandoffDoesNotMislabelCPUAsHybridDeviceProof(t *testing.T) {
	m := pdbenchHybridFixtureModel()
	be := pdbenchCPUBackend(t)
	_, _, _, err := buildHandoff(context.Background(), m, be, []int{3, 7, 11, 5}, 3, artifactIdentity{})
	if err == nil || !strings.Contains(err.Error(), "cannot execute forward \"qwen35-gdn\"") {
		t.Fatalf("CPU hybrid build error=%v, want fail-closed backend refusal", err)
	}
}

// fak-test:runtime fast est=2s lane=short
func TestConsumerRefusesSameGeometryWrongIdentityConfigAndPrompt(t *testing.T) {
	prompt := []int{3, 7, 11, 5}
	const steps = 3
	be := pdbenchCPUBackend(t)
	for _, tc := range []struct {
		name   string
		mutate func(*runConfig)
	}{
		{"artifact", func(cfg *runConfig) { cfg.artifact.RawSHA256 = strings.Repeat("b", 64) }},
		{"quant-config", func(cfg *runConfig) { cfg.configID = strings.Repeat("c", 64) }},
		{"prompt", func(cfg *runConfig) { cfg.expectedPrompt = []int{3, 7, 11, 6} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			producerModel := pdbenchDenseFixtureModel()
			consumerModel := pdbenchDenseFixtureModel()
			producerCfg := pdbenchRunConfig(t, producerModel, prompt, steps)
			consumerCfg := producerCfg
			tc.mutate(&consumerCfg)
			producerConn, consumerConn := pdbenchLoopbackPair(t)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			setDeadline(producerConn, ctx)
			setDeadline(consumerConn, ctx)
			consumerErr := make(chan error, 1)
			go func() {
				_, err := runConsumer(ctx, consumerCfg, consumerModel, be, consumerConn)
				consumerErr <- err
			}()
			if _, err := runProducer(ctx, producerCfg, producerModel, be, producerConn, prompt, steps); err == nil {
				t.Fatal("producer accepted consumer identity refusal")
			}
			if err := <-consumerErr; !errors.Is(err, ErrIdentity) {
				t.Fatalf("consumer error=%v, want ErrIdentity", err)
			}
		})
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestConsumeHandoffRefusesNonFiniteLogits(t *testing.T) {
	m := pdbenchDenseFixtureModel()
	be := pdbenchCPUBackend(t)
	prompt := []int{3, 7, 11, 5}
	header, snapshot, logits, err := buildHandoff(context.Background(), m, be, prompt, 3, artifactIdentity{})
	if err != nil {
		t.Fatalf("build handoff: %v", err)
	}
	copy(logits[:4], encodeFloat32([]float32{float32(math.NaN())}))
	if _, err := consumeHandoff(context.Background(), m, be, header, snapshot, logits); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("consume non-finite logits error=%v", err)
	}
}

// fak-test:runtime fast est=1s lane=short
func TestCanceledConsumerUnblocksBlockedTCPRead(t *testing.T) {
	m := pdbenchDenseFixtureModel()
	be := pdbenchCPUBackend(t)
	cfg := pdbenchRunConfig(t, m, []int{3, 7, 11, 5}, 3)
	client, server := pdbenchLoopbackPair(t)
	if err := server.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatalf("set safety deadline: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := runConsumer(ctx, cfg, m, be, server)
		done <- err
	}()
	time.Sleep(20 * time.Millisecond)
	started := time.Now()
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("blocked consumer error=%v, want context.Canceled", err)
		}
		if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
			t.Fatalf("cancellation took %v, want <=500ms", elapsed)
		}
	case <-time.After(time.Second):
		t.Fatal("cancellation did not unblock blocked TCP read")
	}
	_ = client.Close()
}

type pdbenchNamedBackend struct {
	compute.Backend
	name string
	caps compute.Caps
}

func (b pdbenchNamedBackend) Name() string       { return b.name }
func (b pdbenchNamedBackend) Caps() compute.Caps { return b.caps }

// fak-test:runtime fast est=100ms lane=default
func TestPhysicalDeviceIdentityIsDistinctAndNeverHardwareQualified(t *testing.T) {
	const uuid = "GPU-12345678-1234-1234-1234-123456789abc"
	identity := backendIdentity{
		Name: "cuda", Tier: "sm90", Class: "reference", DeviceMemory: true,
		QuantFormat: "q8_0-on-load", CUDAVisibleDevices: uuid,
	}
	if err := validatePeerBackend(identity, identity); !errors.Is(err, ErrIdentity) {
		t.Fatalf("same physical UUID error=%v, want ErrIdentity", err)
	}
	caseVariant := identity
	caseVariant.CUDAVisibleDevices = "GPU-12345678-1234-1234-1234-123456789ABC"
	if err := validatePeerBackend(identity, caseVariant); !errors.Is(err, ErrIdentity) {
		t.Fatalf("same physical UUID with different hex case error=%v, want ErrIdentity", err)
	}
	if validGPUUUID(uuid + ",GPU-abcdefab-cdef-cdef-cdef-abcdefabcdef") {
		t.Fatal("comma-separated CUDA devices accepted as one physical identity")
	}
	be := pdbenchNamedBackend{Backend: pdbenchCPUBackend(t), name: "cuda", caps: compute.Caps{DeviceMemory: true}}
	if scope := evidenceScope(be); scope != "declared-distinct-device-process-evidence-no-hardware-qualification" {
		t.Fatalf("declared-device evidence scope=%q, want explicit no-hardware-qualification", scope)
	}
}
