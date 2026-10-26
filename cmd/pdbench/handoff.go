package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	protocolVersion  = "fak-pdbench/v1"
	maxHeaderBytes   = uint64(1 << 20)
	maxSnapshotBytes = uint64(512 << 20)
	maxLogitsBytes   = uint64(16 << 20)
	maxReplyBytes    = uint64(1 << 20)
	maxPromptTokens  = 8192
	maxDecodeSteps   = 256
)

var (
	ErrProtocol = errors.New("pdbench: protocol violation")
	ErrIdentity = errors.New("pdbench: identity mismatch")
	ErrParity   = errors.New("pdbench: generated-token parity mismatch")
)

type artifactIdentity struct {
	Bytes              int64  `json:"bytes"`
	RawSHA256          string `json:"raw_sha256"`
	QuantFormat        string `json:"quant_format"`
	ManifestSHA256     string `json:"manifest_sha256,omitempty"`
	ShardSetSHA256     string `json:"shard_set_sha256,omitempty"`
	ChatTemplateSHA256 string `json:"chat_template_sha256,omitempty"`
}

type backendIdentity struct {
	Name               string `json:"name"`
	Tier               string `json:"tier"`
	Class              string `json:"class"`
	DeviceMemory       bool   `json:"device_memory"`
	QuantFormat        string `json:"quant_format"`
	CUDAVisibleDevices string `json:"cuda_visible_devices,omitempty"`
}

type handoffHeader struct {
	Protocol        string           `json:"protocol"`
	Artifact        artifactIdentity `json:"artifact"`
	ConfigSHA256    string           `json:"config_sha256"`
	PromptIDs       []int            `json:"prompt_ids"`
	PromptSHA256    string           `json:"prompt_sha256"`
	Steps           int              `json:"steps"`
	ExpectedTokens  []int            `json:"expected_tokens"`
	ProducerBackend backendIdentity  `json:"producer_backend"`
	SnapshotBytes   int              `json:"snapshot_bytes"`
	LogitsBytes     int              `json:"logits_bytes"`
	PrefillNanos    int64            `json:"prefill_nanos"`
	SnapshotNanos   int64            `json:"snapshot_nanos"`
	ReferenceNanos  int64            `json:"reference_decode_nanos"`
}

type handoffReply struct {
	Protocol        string          `json:"protocol"`
	ConsumerBackend backendIdentity `json:"consumer_backend"`
	Tokens          []int           `json:"tokens"`
	Parity          bool            `json:"parity"`
	RestoreNanos    int64           `json:"restore_nanos"`
	DecodeNanos     int64           `json:"decode_nanos"`
	ErrorCode       string          `json:"error_code,omitempty"`
	Error           string          `json:"error,omitempty"`
}

type wireAck struct {
	OK    bool   `json:"ok"`
	Code  string `json:"code,omitempty"`
	Error string `json:"error,omitempty"`
}

type receipt struct {
	Protocol             string           `json:"protocol"`
	Role                 string           `json:"role"`
	Artifact             artifactIdentity `json:"artifact"`
	ConfigSHA256         string           `json:"config_sha256"`
	PromptIDs            []int            `json:"prompt_ids,omitempty"`
	Steps                int              `json:"steps"`
	ProducerBackend      backendIdentity  `json:"producer_backend,omitempty"`
	ConsumerBackend      backendIdentity  `json:"consumer_backend,omitempty"`
	SnapshotBytes        int              `json:"snapshot_bytes"`
	LogitsBytes          int              `json:"logits_bytes"`
	WireBytes            int64            `json:"wire_bytes"`
	PrefillNanos         int64            `json:"prefill_nanos,omitempty"`
	SnapshotNanos        int64            `json:"snapshot_nanos,omitempty"`
	ReferenceDecodeNanos int64            `json:"reference_decode_nanos,omitempty"`
	RestoreNanos         int64            `json:"restore_nanos,omitempty"`
	DecodeNanos          int64            `json:"decode_nanos,omitempty"`
	EndToEndHandoffNanos int64            `json:"end_to_end_handoff_nanos,omitempty"`
	ReferenceTokens      []int            `json:"reference_tokens,omitempty"`
	ConsumerTokens       []int            `json:"consumer_tokens,omitempty"`
	Parity               bool             `json:"parity"`
	EvidenceScope        string           `json:"evidence_scope"`
	PerformanceClaim     bool             `json:"performance_claim"`
}

type runConfig struct {
	artifact       artifactIdentity
	backend        backendIdentity
	configID       string
	expectedPrompt []int
	expectedSteps  int
}

type countingConn struct {
	net.Conn
	read, written int64
}

func (c *countingConn) Read(p []byte) (int, error) {
	n, err := c.Conn.Read(p)
	c.read += int64(n)
	return n, err
}

func (c *countingConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	c.written += int64(n)
	return n, err
}

func writeFrame(w io.Writer, payload []byte, limit uint64) error {
	if uint64(len(payload)) > limit {
		return fmt.Errorf("pdbench: frame is %d bytes, limit %d", len(payload), limit)
	}
	var header [8]byte
	binary.BigEndian.PutUint64(header[:], uint64(len(payload)))
	if err := writeAll(w, header[:]); err != nil {
		return err
	}
	return writeAll(w, payload)
}

func writeAll(w io.Writer, p []byte) error {
	for len(p) != 0 {
		n, err := w.Write(p)
		if n < 0 || n > len(p) {
			return io.ErrShortWrite
		}
		if n > 0 {
			p = p[n:]
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func readFrame(r io.Reader, limit uint64) ([]byte, error) {
	var header [8]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint64(header[:])
	if n > limit {
		return nil, fmt.Errorf("pdbench: frame declares %d bytes, limit %d", n, limit)
	}
	buf := make([]byte, int(n))
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func buildHandoff(ctx context.Context, m *model.Model, be compute.Backend, prompt []int, steps int, identity artifactIdentity) (handoffHeader, []byte, []byte, error) {
	if len(prompt) == 0 || len(prompt) > maxPromptTokens {
		return handoffHeader{}, nil, nil, fmt.Errorf("pdbench: prompt token count must be 1..%d", maxPromptTokens)
	}
	for _, id := range prompt {
		if id < 0 || id >= m.Cfg.VocabSize {
			return handoffHeader{}, nil, nil, fmt.Errorf("pdbench: token ID %d outside model vocabulary [0,%d)", id, m.Cfg.VocabSize)
		}
	}
	if steps < 2 || steps > maxDecodeSteps {
		return handoffHeader{}, nil, nil, fmt.Errorf("pdbench: steps must be 2..%d", maxDecodeSteps)
	}
	if err := ctx.Err(); err != nil {
		return handoffHeader{}, nil, nil, err
	}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		return handoffHeader{}, nil, nil, err
	}
	defer s.Close()
	s.Quant = be.Caps().UploadDtype
	prefillStart := time.Now()
	logits := s.Prefill(prompt)
	prefillNanos := time.Since(prefillStart).Nanoseconds()
	if err := validateLogits(logits); err != nil {
		return handoffHeader{}, nil, nil, err
	}
	initialLogits := append([]float32(nil), logits...)

	// Capture before reference generation: the wire image is exactly the prefill boundary.
	snapshotStart := time.Now()
	prefix, err := s.PrefixSnapshot()
	if err != nil {
		return handoffHeader{}, nil, nil, err
	}
	host, err := prefix.CloneToHost()
	prefix.Close()
	if err != nil {
		return handoffHeader{}, nil, nil, err
	}
	snapshot, err := host.MarshalBinary()
	host.Close()
	if err != nil {
		return handoffHeader{}, nil, nil, err
	}
	snapshotNanos := time.Since(snapshotStart).Nanoseconds()
	if uint64(len(snapshot)) > maxSnapshotBytes {
		return handoffHeader{}, nil, nil, fmt.Errorf("pdbench: snapshot is %d bytes, limit %d", len(snapshot), maxSnapshotBytes)
	}

	referenceStart := time.Now()
	tokens := make([]int, steps)
	tokens[0] = mathx.ArgmaxF32(logits)
	for i := 1; i < steps; i++ {
		if err := ctx.Err(); err != nil {
			return handoffHeader{}, nil, nil, err
		}
		tokens[i], logits, err = greedyStep(s, tokens[i-1])
		if err != nil {
			return handoffHeader{}, nil, nil, err
		}
	}
	referenceNanos := time.Since(referenceStart).Nanoseconds()
	logitsWire := encodeFloat32(initialLogits)
	if uint64(len(logitsWire)) > maxLogitsBytes {
		return handoffHeader{}, nil, nil, fmt.Errorf("pdbench: logits are %d bytes, limit %d", len(logitsWire), maxLogitsBytes)
	}
	header := handoffHeader{
		Protocol: protocolVersion, Artifact: identity, PromptIDs: append([]int(nil), prompt...),
		PromptSHA256: digestInts(prompt), Steps: steps, ExpectedTokens: tokens,
		SnapshotBytes: len(snapshot), LogitsBytes: len(logitsWire), PrefillNanos: prefillNanos,
		SnapshotNanos: snapshotNanos, ReferenceNanos: referenceNanos,
	}
	return header, snapshot, logitsWire, nil
}

func consumeHandoff(ctx context.Context, m *model.Model, be compute.Backend, header handoffHeader, snapshot, logitsWire []byte) (handoffReply, error) {
	reply := handoffReply{Protocol: protocolVersion, ConsumerBackend: identifyBackend(be)}
	if err := validateHeader(header, snapshot, logitsWire); err != nil {
		return reply, err
	}
	logits, err := decodeFloat32(logitsWire)
	if err != nil {
		return reply, err
	}
	if err := validateLogits(logits); err != nil {
		return reply, err
	}
	if err := ctx.Err(); err != nil {
		return reply, err
	}
	restoreStart := time.Now()
	host, err := model.DecodeHostPrefixSnapshot(snapshot, be, m.Cfg)
	if err != nil {
		return reply, err
	}
	defer host.Close()
	prefix, err := host.Restore()
	if err != nil {
		return reply, err
	}
	defer prefix.Close()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		return reply, err
	}
	defer s.Close()
	s.Quant = be.Caps().UploadDtype
	if err := prefix.Restore(s); err != nil {
		return reply, err
	}
	reply.RestoreNanos = time.Since(restoreStart).Nanoseconds()

	decodeStart := time.Now()
	reply.Tokens = make([]int, header.Steps)
	reply.Tokens[0] = mathx.ArgmaxF32(logits)
	for i := 1; i < header.Steps; i++ {
		if err := ctx.Err(); err != nil {
			return reply, err
		}
		reply.Tokens[i], logits, err = greedyStep(s, reply.Tokens[i-1])
		if err != nil {
			return reply, err
		}
	}
	reply.DecodeNanos = time.Since(decodeStart).Nanoseconds()
	reply.Parity = equalInts(reply.Tokens, header.ExpectedTokens)
	return reply, nil
}

func runProducer(ctx context.Context, cfg runConfig, m *model.Model, be compute.Backend, conn net.Conn, prompt []int, steps int) (receipt, error) {
	stopCancel := interruptConnOnCancel(ctx, conn)
	defer stopCancel()
	if err := validateRunConfig(cfg); err != nil {
		return receipt{}, err
	}
	meter := &countingConn{Conn: conn}
	header, snapshot, logits, err := buildHandoff(ctx, m, be, prompt, steps, cfg.artifact)
	if err != nil {
		return receipt{}, err
	}
	header.ConfigSHA256 = cfg.configID
	header.ProducerBackend = cfg.backend
	headerBytes, err := json.Marshal(header)
	if err != nil {
		return receipt{}, err
	}
	started := time.Now()
	if err := writeFrame(meter, headerBytes, maxHeaderBytes); err != nil {
		return receipt{}, preferContextError(ctx, err)
	}
	ackBytes, err := readFrame(meter, maxReplyBytes)
	if err != nil {
		return receipt{}, preferContextError(ctx, err)
	}
	var ack wireAck
	if err := json.Unmarshal(ackBytes, &ack); err != nil {
		return receipt{}, fmt.Errorf("%w: malformed receiver acknowledgement", ErrProtocol)
	}
	if !ack.OK {
		return receipt{}, decodeWireError(ack.Code, ack.Error)
	}
	for _, frame := range []struct {
		data  []byte
		limit uint64
	}{{snapshot, maxSnapshotBytes}, {logits, maxLogitsBytes}} {
		if err := writeFrame(meter, frame.data, frame.limit); err != nil {
			return receipt{}, preferContextError(ctx, err)
		}
	}
	replyBytes, err := readFrame(meter, maxReplyBytes)
	if err != nil {
		return receipt{}, preferContextError(ctx, err)
	}
	var reply handoffReply
	if err := json.Unmarshal(replyBytes, &reply); err != nil {
		return receipt{}, err
	}
	if reply.Error != "" {
		return receipt{}, decodeWireError(reply.ErrorCode, reply.Error)
	}
	if err := validatePeerBackend(cfg.backend, reply.ConsumerBackend); err != nil {
		return receipt{}, err
	}
	parity := reply.Parity && equalInts(reply.Tokens, header.ExpectedTokens)
	r := receipt{
		Protocol: protocolVersion, Role: "producer", Artifact: cfg.artifact, ConfigSHA256: cfg.configID,
		PromptIDs: prompt, Steps: steps, ProducerBackend: cfg.backend, ConsumerBackend: reply.ConsumerBackend,
		SnapshotBytes: len(snapshot), LogitsBytes: len(logits), WireBytes: meter.read + meter.written,
		PrefillNanos: header.PrefillNanos, SnapshotNanos: header.SnapshotNanos, ReferenceDecodeNanos: header.ReferenceNanos,
		RestoreNanos: reply.RestoreNanos, DecodeNanos: reply.DecodeNanos, EndToEndHandoffNanos: time.Since(started).Nanoseconds(),
		ReferenceTokens: header.ExpectedTokens, ConsumerTokens: reply.Tokens, Parity: parity,
		EvidenceScope: evidenceScope(be), PerformanceClaim: false,
	}
	if !parity {
		return r, ErrParity
	}
	return r, nil
}

func runConsumer(ctx context.Context, cfg runConfig, m *model.Model, be compute.Backend, conn net.Conn) (receipt, error) {
	stopCancel := interruptConnOnCancel(ctx, conn)
	defer stopCancel()
	if err := validateRunConfig(cfg); err != nil {
		return receipt{}, err
	}
	meter := &countingConn{Conn: conn}
	headerBytes, err := readFrame(meter, maxHeaderBytes)
	if err != nil {
		return receipt{}, preferContextError(ctx, err)
	}
	var header handoffHeader
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return receipt{}, sendAckError(meter, fmt.Errorf("%w: malformed header JSON", ErrProtocol))
	}
	err = validateHeaderMetadata(header)
	if err == nil {
		if header.Artifact != cfg.artifact {
			err = fmt.Errorf("%w: producer and consumer GGUF", ErrIdentity)
		} else if header.ConfigSHA256 != cfg.configID {
			err = fmt.Errorf("%w: producer and consumer model configurations", ErrIdentity)
		} else if !equalInts(header.PromptIDs, cfg.expectedPrompt) || header.Steps != cfg.expectedSteps {
			err = fmt.Errorf("%w: prompt or steps differ from receiver expectation", ErrIdentity)
		} else {
			err = validatePeerBackend(cfg.backend, header.ProducerBackend)
		}
	}
	if err != nil {
		return receipt{}, sendAckError(meter, err)
	}
	ackBytes, _ := json.Marshal(wireAck{OK: true})
	if err := writeFrame(meter, ackBytes, maxReplyBytes); err != nil {
		return receipt{}, preferContextError(ctx, err)
	}
	snapshot, err := readFrame(meter, maxSnapshotBytes)
	if err != nil {
		return receipt{}, preferContextError(ctx, err)
	}
	logits, err := readFrame(meter, maxLogitsBytes)
	if err != nil {
		return receipt{}, preferContextError(ctx, err)
	}
	var reply handoffReply
	if err == nil {
		reply, err = consumeHandoff(ctx, m, be, header, snapshot, logits)
	}
	if err != nil {
		reply = handoffReply{Protocol: protocolVersion, ConsumerBackend: cfg.backend, ErrorCode: wireErrorCode(err), Error: err.Error()}
	}
	replyBytes, marshalErr := json.Marshal(reply)
	if marshalErr == nil {
		marshalErr = writeFrame(meter, replyBytes, maxReplyBytes)
	}
	if err != nil {
		return receipt{}, err
	}
	if marshalErr != nil {
		return receipt{}, preferContextError(ctx, marshalErr)
	}
	r := receipt{
		Protocol: protocolVersion, Role: "consumer", Artifact: cfg.artifact, ConfigSHA256: cfg.configID,
		Steps: header.Steps, ProducerBackend: header.ProducerBackend, ConsumerBackend: cfg.backend,
		SnapshotBytes: len(snapshot), LogitsBytes: len(logits), WireBytes: meter.read + meter.written,
		RestoreNanos: reply.RestoreNanos, DecodeNanos: reply.DecodeNanos, ConsumerTokens: reply.Tokens,
		Parity: reply.Parity, EvidenceScope: evidenceScope(be), PerformanceClaim: false,
	}
	if !reply.Parity {
		return r, ErrParity
	}
	return r, nil
}

func validateHeader(h handoffHeader, snapshot, logits []byte) error {
	if err := validateHeaderMetadata(h); err != nil {
		return err
	}
	if h.SnapshotBytes != len(snapshot) || h.LogitsBytes != len(logits) {
		return fmt.Errorf("%w: frame lengths do not match metadata", ErrProtocol)
	}
	return nil
}

func validateHeaderMetadata(h handoffHeader) error {
	if h.Protocol != protocolVersion {
		return fmt.Errorf("%w: version %q", ErrProtocol, h.Protocol)
	}
	if len(h.PromptIDs) == 0 || len(h.PromptIDs) > maxPromptTokens || h.Steps < 2 || h.Steps > maxDecodeSteps {
		return fmt.Errorf("%w: invalid prompt or decode bounds", ErrProtocol)
	}
	if len(h.ExpectedTokens) != h.Steps || h.SnapshotBytes < 1 || uint64(h.SnapshotBytes) > maxSnapshotBytes ||
		h.LogitsBytes < 4 || h.LogitsBytes%4 != 0 || uint64(h.LogitsBytes) > maxLogitsBytes {
		return fmt.Errorf("%w: declared frame lengths or expected tokens", ErrProtocol)
	}
	if digestInts(h.PromptIDs) != h.PromptSHA256 {
		return fmt.Errorf("%w: prompt token digest", ErrIdentity)
	}
	return nil
}

func sendAckError(conn io.Writer, err error) error {
	b, marshalErr := json.Marshal(wireAck{Code: wireErrorCode(err), Error: err.Error()})
	if marshalErr == nil {
		marshalErr = writeFrame(conn, b, maxReplyBytes)
	}
	if marshalErr != nil {
		return errors.Join(err, marshalErr)
	}
	return err
}

func wireErrorCode(err error) string {
	switch {
	case errors.Is(err, ErrProtocol):
		return "protocol"
	case errors.Is(err, ErrIdentity):
		return "identity"
	case errors.Is(err, ErrParity):
		return "parity"
	default:
		return "runtime"
	}
}

func decodeWireError(code, detail string) error {
	var sentinel error
	switch code {
	case "protocol":
		sentinel = ErrProtocol
	case "identity":
		sentinel = ErrIdentity
	case "parity":
		sentinel = ErrParity
	default:
		return fmt.Errorf("pdbench: remote runtime error: %s", detail)
	}
	return fmt.Errorf("%w: remote: %s", sentinel, detail)
}

func validateLogits(v []float32) error {
	if len(v) == 0 {
		return errors.New("pdbench: model returned no logits")
	}
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return errors.New("pdbench: model returned non-finite logits")
		}
	}
	return nil
}

func greedyStep(s *model.Session, token int) (int, []float32, error) {
	next, logits := s.StepGreedy(token)
	if logits == nil {
		return next, nil, nil
	}
	if err := validateLogits(logits); err != nil {
		return 0, nil, err
	}
	return mathx.ArgmaxF32(logits), logits, nil
}

func validatePeerBackend(local, peer backendIdentity) error {
	if peer.Name != local.Name || peer.Class != local.Class || peer.DeviceMemory != local.DeviceMemory ||
		peer.QuantFormat != local.QuantFormat {
		return fmt.Errorf("%w: backend or quant execution identity", ErrIdentity)
	}
	if local.Name == "cuda" {
		if peer.Tier != local.Tier {
			return fmt.Errorf("%w: CUDA backend tier", ErrIdentity)
		}
		if !validGPUUUID(local.CUDAVisibleDevices) || !validGPUUUID(peer.CUDAVisibleDevices) ||
			strings.EqualFold(peer.CUDAVisibleDevices, local.CUDAVisibleDevices) {
			return fmt.Errorf("%w: CUDA roles must declare distinct device UUIDs", ErrIdentity)
		}
	}
	return nil
}

func validateRunConfig(cfg runConfig) error {
	if cfg.artifact.QuantFormat == "" || cfg.artifact.QuantFormat != cfg.backend.QuantFormat {
		return fmt.Errorf("%w: artifact and backend quant execution identity", ErrIdentity)
	}
	return nil
}

func interruptConnOnCancel(ctx context.Context, conn net.Conn) func() {
	stop := context.AfterFunc(ctx, func() {
		_ = conn.SetDeadline(time.Now())
	})
	return func() { _ = stop() }
}

func preferContextError(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	return err
}

func encodeFloat32(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func decodeFloat32(b []byte) ([]float32, error) {
	if len(b) == 0 || len(b)%4 != 0 {
		return nil, errors.New("pdbench: malformed logits frame")
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, nil
}

func digestInts(v []int) string {
	h := sha256.New()
	var b [8]byte
	for _, n := range v {
		binary.BigEndian.PutUint64(b[:], uint64(int64(n)))
		_, _ = h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
