package mcpbroker

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

const (
	minStructuredCompressionBytes = 48
	maxStructuredCompressionBytes = 16 << 20 // 16 MiB bound on compressible response size
)

// IsOperatorForcedIdentity reports whether operator configuration forces identity output
// (FAK_COMPRESSOR=noop or none).
func IsOperatorForcedIdentity() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_COMPRESSOR"))) {
	case "noop", "none":
		return true
	}
	return false
}

// CompressionReason represents the closed vocabulary of structured compression decision reasons.
type CompressionReason string

const (
	// ReasonSaved indicates successful compression with measured byte savings.
	ReasonSaved CompressionReason = "saved"
	// ReasonOptOut indicates compression was bypassed due to environment or caller opt-out.
	ReasonOptOut CompressionReason = "opt_out"
	// ReasonNoneligible indicates the content or result is not eligible for structured compression
	// (e.g. content too small, no structured content, error result, multiple blocks, non-text, or mismatch).
	ReasonNoneligible CompressionReason = "noneligible"
	// ReasonPoison indicates the content was flagged by security screening (ctxmmu.ScreenBytes).
	ReasonPoison CompressionReason = "poison"
	// ReasonMalformed indicates invalid JSON in result, content, block, or payload string.
	ReasonMalformed CompressionReason = "malformed"
	// ReasonAmbiguous indicates duplicate or ambiguous cased keys in envelope or block.
	ReasonAmbiguous CompressionReason = "ambiguous"
	// ReasonInsufficientSaving indicates the potential savings did not clear the minimum threshold
	// (< 256 bytes and < 15% saving).
	ReasonInsufficientSaving CompressionReason = "insufficient_saving"
	// ReasonPreserveFailed indicates exact-original preservation failed (e.g. store failure or size cap exceeded),
	// forcing fail-safe identity output.
	ReasonPreserveFailed CompressionReason = "preserve_failed"
)

const (
	// DefaultCompressionCodec identifies the structured JSON minification codec.
	DefaultCompressionCodec = "json-min"

	// CompressionStageIdentity identifies the pipeline stage in receipts and metadata.
	CompressionStageIdentity = "mcpbroker.compact_structured"
)

// CompressionReceipt represents a payload-free decision receipt for structured compression.
// It contains exact byte accounting, decision reason, codec identifier, and latency duration
// without retaining any raw payload bytes.
type CompressionReceipt struct {
	// Reason is the decision reason distinguishing why compression occurred or was skipped.
	Reason CompressionReason `json:"reason"`

	// Codec identifies the compression algorithm/transform used (e.g. "json-min" when saved).
	Codec string `json:"codec,omitempty"`

	// InputBytes is the byte length of the input content considered for compression.
	InputBytes int `json:"input_bytes"`

	// OutputBytes is the byte length of the content after compression (identical to InputBytes when skipped).
	OutputBytes int `json:"output_bytes"`

	// BytesSaved is the number of bytes saved (InputBytes - OutputBytes; 0 when skipped).
	BytesSaved int `json:"bytes_saved"`

	// Duration is the latency duration taken to evaluate and execute compression.
	Duration time.Duration `json:"duration"`

	// Latency is an alias for Duration for compatibility with latency observers.
	Latency time.Duration `json:"latency,omitempty"`

	// Stage is the pipeline stage identity that emitted this receipt.
	Stage string `json:"stage,omitempty"`

	// RestoreHandle carries the content-addressed retrieval handle if exact-original retention was enabled.
	RestoreHandle string `json:"restore_handle,omitempty"`

	// Metadata carries optional payload-free tags and diagnostic attributes.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// SavedRatio returns the fraction of bytes saved in [0, 1].
func (r *CompressionReceipt) SavedRatio() float64 {
	if r.InputBytes <= 0 || r.OutputBytes >= r.InputBytes {
		return 0
	}
	return float64(r.BytesSaved) / float64(r.InputBytes)
}

// CompressionPolicy represents the compression policy for an MCP tool call or session.
type CompressionPolicy string

const (
	// CompressionAuto indicates default automatic structured compression when eligible.
	CompressionAuto CompressionPolicy = "auto"

	// CompressionIdentity indicates identity output (structured compression opt-out).
	CompressionIdentity CompressionPolicy = "identity"

	// CompressionOff is an alias for CompressionIdentity.
	CompressionOff CompressionPolicy = "off"

	// CompressionOptOut is an alias for CompressionIdentity.
	CompressionOptOut CompressionPolicy = "opt_out"
)

type compressionContextKey struct{}
type compressionPolicyContextKey struct{}
type metadataContextKey struct{}

// WithCompressionPolicy returns a derived context carrying the specified compression policy.
func WithCompressionPolicy(ctx context.Context, policy CompressionPolicy) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if IsCompressionOptOut(string(policy)) {
		ctx = context.WithValue(ctx, compressionContextKey{}, true)
	}
	return context.WithValue(ctx, compressionPolicyContextKey{}, policy)
}

// WithCompressionOptOut returns a derived context requesting identity output (opting out of compression).
func WithCompressionOptOut(ctx context.Context) context.Context {
	return WithCompressionPolicy(ctx, CompressionIdentity)
}

// isContextOptOut reports whether the context requests opting out of structured compression.
func isContextOptOut(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	if v, ok := ctx.Value(compressionContextKey{}).(bool); ok && v {
		return true
	}
	if p, ok := CompressionPolicyFromContext(ctx); ok && IsCompressionOptOut(string(p)) {
		return true
	}
	return false
}

// CompressionPolicyFromContext extracts the compression policy stored in ctx, if any.
func CompressionPolicyFromContext(ctx context.Context) (CompressionPolicy, bool) {
	if ctx == nil {
		return "", false
	}
	p, ok := ctx.Value(compressionPolicyContextKey{}).(CompressionPolicy)
	return p, ok
}

// WithCallMetadata returns a derived context carrying call metadata.
func WithCallMetadata(ctx context.Context, md map[string]string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if len(md) == 0 {
		return ctx
	}
	copied := make(map[string]string, len(md))
	for k, v := range md {
		copied[k] = v
	}
	return context.WithValue(ctx, metadataContextKey{}, copied)
}

// CallMetadataFromContext extracts call metadata stored in ctx, if any.
func CallMetadataFromContext(ctx context.Context) map[string]string {
	if ctx == nil {
		return nil
	}
	md, ok := ctx.Value(metadataContextKey{}).(map[string]string)
	if !ok {
		return nil
	}
	return md
}

// IsCompressionOptOut returns true if the given value requests opting out of structured compression
// (i.e. selecting identity output).
func IsCompressionOptOut(val string) bool {
	switch strings.ToLower(strings.TrimSpace(val)) {
	case "identity", "off", "opt_out", "opt-out", "optout", "none", "noop", "disabled", "disable", "false", "0":
		return true
	default:
		return false
	}
}

// IsCompressionMetadataOptOut inspects a metadata map for supported compression opt-out tags.
func IsCompressionMetadataOptOut(md map[string]string) bool {
	if len(md) == 0 {
		return false
	}
	for k, v := range md {
		switch strings.ToLower(strings.TrimSpace(k)) {
		case "compression", "mcp_compression", "mcp-compression", "structured_compression", "structured-compression":
			if IsCompressionOptOut(v) {
				return true
			}
		}
	}
	return false
}

// ResolveEffectiveCompression determines the effective compression policy for a call based on
// operator environment, call-level request fields, request metadata, context, and session defaults.
// Precedence:
//  1. Operator-forced identity (FAK_COMPRESSOR=noop or none) dominates everything. Caller cannot widen.
//  2. CallRequest.Compression typed field.
//  3. CallRequest.Metadata (e.g. "compression", "mcp_compression").
//  4. Context policy (WithCompressionPolicy / WithCompressionOptOut).
//  5. Session-level policy.
//  6. Default: CompressionAuto.
func ResolveEffectiveCompression(ctx context.Context, req CallRequest, sessionPolicy CompressionPolicy) CompressionPolicy {
	if IsOperatorForcedIdentity() {
		return CompressionIdentity
	}

	if req.Compression != "" {
		if IsCompressionOptOut(string(req.Compression)) {
			return CompressionIdentity
		}
		return CompressionAuto
	}

	if len(req.Metadata) > 0 {
		for k, v := range req.Metadata {
			switch strings.ToLower(strings.TrimSpace(k)) {
			case "compression", "mcp_compression", "mcp-compression", "structured_compression", "structured-compression":
				if IsCompressionOptOut(v) {
					return CompressionIdentity
				}
				switch strings.ToLower(strings.TrimSpace(v)) {
				case "auto", "default", "on", "compress", "enable", "enabled", "true":
					return CompressionAuto
				}
			}
		}
	}

	if ctx != nil {
		if p, ok := CompressionPolicyFromContext(ctx); ok && p != "" {
			if IsCompressionOptOut(string(p)) {
				return CompressionIdentity
			}
			return CompressionAuto
		}
		if isContextOptOut(ctx) {
			return CompressionIdentity
		}
		if md := CallMetadataFromContext(ctx); len(md) > 0 {
			if IsCompressionMetadataOptOut(md) {
				return CompressionIdentity
			}
		}
	}

	if sessionPolicy != "" {
		if IsCompressionOptOut(string(sessionPolicy)) {
			return CompressionIdentity
		}
		return CompressionAuto
	}

	return CompressionAuto
}

// CompressionOptions configures the execution of structured compression.
type CompressionOptions struct {
	Context       context.Context
	OptOut        bool
	Codec         string
	ExactOriginal bool
	SessionID     string
	Store         *RestoreStore
}

// CompressionOption is a functional option for CompactStructuredContentWithReceipt.
type CompressionOption func(*CompressionOptions)

// WithCompressionContext sets the context for caller policy resolution.
func WithCompressionContext(ctx context.Context) CompressionOption {
	return func(o *CompressionOptions) {
		o.Context = ctx
	}
}

// WithOptOut sets caller opt-out for structured compression.
func WithOptOut(optOut bool) CompressionOption {
	return func(o *CompressionOptions) {
		o.OptOut = optOut
	}
}

// WithCompressionCodec overrides the default codec identifier.
func WithCompressionCodec(codec string) CompressionOption {
	return func(o *CompressionOptions) {
		o.Codec = codec
	}
}

// CompactStructuredContentWithReceipt evaluates structured content compression and returns
// the transformed (or identity) content along with a payload-free CompressionReceipt.
func CompactStructuredContentWithReceipt(result, content json.RawMessage, opts ...CompressionOption) (json.RawMessage, *CompressionReceipt) {
	start := time.Now()
	inputLen := len(content)

	var options CompressionOptions
	for _, opt := range opts {
		opt(&options)
	}

	makeReceipt := func(reason CompressionReason, outLen, saved int, codec string) *CompressionReceipt {
		elapsed := time.Since(start)
		meta := map[string]string{
			"stage":  CompressionStageIdentity,
			"reason": string(reason),
		}
		if codec != "" {
			meta["codec"] = codec
		}
		if reason == ReasonSaved {
			meta["decision"] = "saved"
			meta["saved_bytes"] = strconv.Itoa(saved)
			if inputLen > 0 {
				meta["saved_ratio"] = strconv.FormatFloat(float64(saved)/float64(inputLen), 'f', 3, 64)
			}
		} else {
			meta["decision"] = "skipped"
		}
		return &CompressionReceipt{
			Reason:      reason,
			Codec:       codec,
			InputBytes:  inputLen,
			OutputBytes: outLen,
			BytesSaved:  saved,
			Duration:    elapsed,
			Latency:     elapsed,
			Stage:       CompressionStageIdentity,
			Metadata:    meta,
		}
	}

	// 1. Check caller and environment opt-out
	if options.OptOut || IsOperatorForcedIdentity() {
		return content, makeReceipt(ReasonOptOut, inputLen, 0, "")
	}
	if options.Context != nil {
		if isContextOptOut(options.Context) {
			return content, makeReceipt(ReasonOptOut, inputLen, 0, "")
		}
		if policy, ok := CompressionPolicyFromContext(options.Context); ok && IsCompressionOptOut(string(policy)) {
			return content, makeReceipt(ReasonOptOut, inputLen, 0, "")
		}
		if md := CallMetadataFromContext(options.Context); IsCompressionMetadataOptOut(md) {
			return content, makeReceipt(ReasonOptOut, inputLen, 0, "")
		}
	}

	if len(result) > maxStructuredCompressionBytes {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}

	// 2. Parse result envelope
	fields, resReason, ok := parseObjectFields(result)
	if !ok {
		return content, makeReceipt(resReason, inputLen, 0, "")
	}

	// 3. Bound size checks (content must be eligible size)
	if inputLen < minStructuredCompressionBytes || inputLen > maxStructuredCompressionBytes {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}

	// 4. Check reserved envelope keys for ambiguous casing
	for key := range fields {
		for _, reserved := range []string{"content", "structuredContent", "isError"} {
			if key != reserved && strings.EqualFold(key, reserved) {
				return content, makeReceipt(ReasonAmbiguous, inputLen, 0, "")
			}
		}
	}

	// 5. Verify content field presence and identity
	contentField, hasContent := fields["content"]
	if !hasContent || !bytes.Equal(contentField.raw, content) {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}

	// 6. Verify not an error result
	if flag, exists := fields["isError"]; exists && string(flag.raw) != "false" {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}

	// 7. Verify structuredContent presence and object format
	structuredField, hasStructured := fields["structuredContent"]
	if !hasStructured {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}
	structured := structuredField.raw
	if len(structured) == 0 || structured[0] != '{' {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}

	// 8. Parse content array of blocks
	var blocks []json.RawMessage
	if err := json.Unmarshal(content, &blocks); err != nil {
		if !json.Valid(content) {
			return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
		}
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}
	if len(blocks) != 1 {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}
	block := blocks[0]

	// 9. Parse single block object
	parts, blockReason, ok := parseObjectFields(block)
	if !ok {
		return content, makeReceipt(blockReason, inputLen, 0, "")
	}

	// 10. Check block type and text fields
	typeField, hasType := parts["type"]
	textField, hasText := parts["text"]
	if !hasType || !hasText || textField.start < 0 {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}
	var kind string
	if err := json.Unmarshal(typeField.raw, &kind); err != nil || kind != "text" {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}
	var original string
	if err := json.Unmarshal(textField.raw, &original); err != nil {
		return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
	}

	// 11. Security screening: ScreenBytes must not be hidden
	if _, held := ctxmmu.ScreenBytes([]byte(original)); held {
		return content, makeReceipt(ReasonPoison, inputLen, 0, "")
	}

	// 12. Compact and compare with declared structured content
	var compact, declared bytes.Buffer
	if err := json.Compact(&compact, []byte(original)); err != nil {
		return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
	}
	if compact.Len() == 0 || compact.Bytes()[0] != '{' {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}
	if err := json.Compact(&declared, structured); err != nil {
		return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
	}
	if !bytes.Equal(compact.Bytes(), declared.Bytes()) {
		return content, makeReceipt(ReasonNoneligible, inputLen, 0, "")
	}

	// 13. Re-encode and compute byte offsets
	encoded, err := json.Marshal(compact.String())
	if err != nil {
		return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
	}
	blockIdx := bytes.Index(content, block)
	if blockIdx < 0 {
		return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
	}
	startOffset := blockIdx + textField.start
	textLen := len(textField.raw)
	endOffset := startOffset + textLen
	if startOffset < 0 || endOffset > len(content) || startOffset > endOffset {
		return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
	}
	if !bytes.Equal(content[startOffset:endOffset], textField.raw) {
		return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
	}

	newLen := len(content) - (endOffset - startOffset) + len(encoded)
	if newLen < 0 || newLen > maxStructuredCompressionBytes {
		return content, makeReceipt(ReasonMalformed, inputLen, 0, "")
	}

	// 14. Check savings threshold (< 256 bytes and < 15% saving)
	saved := len(content) - newLen
	if saved <= 0 || (saved < 256 && float64(saved)/float64(len(content)) < 0.15) {
		return content, makeReceipt(ReasonInsufficientSaving, inputLen, 0, "")
	}

	// 15. Form compressed output and return ReasonSaved receipt
	out := make([]byte, 0, newLen)
	out = append(out, content[:startOffset]...)
	out = append(out, encoded...)
	out = append(out, content[endOffset:]...)

	// Resolve exact-original retention preferences
	exactOriginal := options.ExactOriginal
	sessionID := options.SessionID
	store := options.Store

	if options.Context != nil {
		if !exactOriginal {
			if eo, ok := ExactOriginalFromContext(options.Context); ok {
				exactOriginal = eo
			}
		}
		if sessionID == "" {
			if sID, ok := SessionIDFromContext(options.Context); ok {
				sessionID = sID
			}
		}
		if store == nil {
			if s, ok := RestoreStoreFromContext(options.Context); ok {
				store = s
			}
		}
		if md := CallMetadataFromContext(options.Context); len(md) > 0 {
			if !exactOriginal && IsExactOriginalMetadataEnabled(md) {
				exactOriginal = true
			}
			if sessionID == "" {
				sessionID = ExtractSessionOrTraceID(md)
			}
		}
	}

	if sessionID != "" && !exactOriginal {
		sessionExactOrigMu.RLock()
		if sessionExactOrig[sessionID] {
			exactOriginal = true
		}
		sessionExactOrigMu.RUnlock()
	}

	var restoreHandle string
	if exactOriginal {
		if store == nil {
			store = DefaultRestoreStore()
		}
		h, err := store.Store(sessionID, content)
		if err != nil {
			// If preservation fails (e.g. store failure or size cap exceeded), fail safely:
			// emit identity output (do not compress without being able to retain the required original).
			rcpt := makeReceipt(ReasonPreserveFailed, inputLen, 0, "")
			if rcpt.Metadata != nil {
				rcpt.Metadata["preserve_error"] = err.Error()
			}
			return content, rcpt
		}
		restoreHandle = h
	}

	codec := DefaultCompressionCodec
	if options.Codec != "" {
		codec = options.Codec
	}
	rcpt := makeReceipt(ReasonSaved, len(out), saved, codec)
	if restoreHandle != "" {
		rcpt.RestoreHandle = restoreHandle
		if rcpt.Metadata == nil {
			rcpt.Metadata = make(map[string]string)
		}
		rcpt.Metadata["restore_handle"] = restoreHandle
		rcpt.Metadata["exact_original"] = "true"
	}
	return out, rcpt
}

// CompactStructuredContent applies structured JSON compression and returns the output payload.
func CompactStructuredContent(result, content json.RawMessage) json.RawMessage {
	out, _ := CompactStructuredContentWithReceipt(result, content)
	return out
}

// compactStructuredContent removes only insignificant JSON whitespace from a
// server-declared structured text mirror. It provides semantic JSON fidelity,
// not byte-exact recovery of the original formatting.
func compactStructuredContent(result, content json.RawMessage) json.RawMessage {
	return compactStructuredContentContext(context.Background(), result, content)
}

// compactStructuredContentContext evaluates operator environment and caller policy
// before removing insignificant JSON whitespace from a server-declared structured text mirror.
func compactStructuredContentContext(ctx context.Context, result, content json.RawMessage) json.RawMessage {
	out, _ := CompactStructuredContentWithReceipt(result, content, WithCompressionContext(ctx))
	return out
}

type compressionField struct {
	raw   json.RawMessage
	start int
}

// compressionObjectFields decodes envelope keys and retains raw value spans.
func compressionObjectFields(raw []byte) (map[string]compressionField, bool) {
	fields, _, ok := parseObjectFields(raw)
	return fields, ok
}

// parseObjectFields decodes envelope keys, checks for ambiguity, and retains raw value spans.
func parseObjectFields(raw []byte) (map[string]compressionField, CompressionReason, bool) {
	if len(raw) == 0 {
		return nil, ReasonNoneligible, false
	}
	if !json.Valid(raw) {
		return nil, ReasonMalformed, false
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	first, err := d.Token()
	if err != nil {
		return nil, ReasonMalformed, false
	}
	if first != json.Delim('{') {
		return nil, ReasonNoneligible, false
	}
	fields := make(map[string]compressionField)
	seen := make(map[string]bool)
	for d.More() {
		token, err := d.Token()
		if err != nil {
			return nil, ReasonMalformed, false
		}
		key, ok := token.(string)
		if !ok {
			return nil, ReasonMalformed, false
		}
		lowerKey := strings.ToLower(key)
		if seen[lowerKey] {
			return nil, ReasonAmbiguous, false
		}
		seen[lowerKey] = true
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return nil, ReasonMalformed, false
		}
		offset := int(d.InputOffset()) - len(value)
		if offset < 0 {
			return nil, ReasonMalformed, false
		}
		fields[key] = compressionField{raw: value, start: offset}
	}
	last, err := d.Token()
	if err != nil || last != json.Delim('}') {
		return nil, ReasonMalformed, false
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, ReasonMalformed, false
	}
	return fields, "", true
}
