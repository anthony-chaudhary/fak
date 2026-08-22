package sessionledger

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/promptaudit"
)

const (
	// KindModelRequestReceipt is the bounded ledger row that points to an exact,
	// content-addressed model-boundary manifest.
	KindModelRequestReceipt = "model_request_receipt"
	KindInputClaimReceipt   = "input_claim_receipt"
	modelRequestObjectDir   = "model-request-objects"
	modelRequestSchema      = "fak-model-request/1"
	inputClaimSchema        = "fak-input-claim/1"
)

// ModelRequestIdentity identifies one model call without participating in the
// model-visible digest. RequestID is durable ledger identity; the other fields
// describe request parameters that do cross the logical provider boundary.
type ModelRequestIdentity struct {
	RequestID string `json:"request_id"`
	Model     string `json:"model"`
	Turn      int    `json:"turn"`
	Stream    bool   `json:"stream,omitempty"`
	MaxTokens int    `json:"max_tokens,omitempty"`
}

// ModelRequestSegment carries one ordered canonical message envelope plus its
// audit attribution. Content is the exact JSON passed to the recorder.
type ModelRequestSegment struct {
	Kind    string             `json:"kind"`
	Source  promptaudit.Source `json:"source"`
	Content json.RawMessage    `json:"content"`
}

// ModelRequest is the reconstructable logical request at the Planner boundary.
// Tools is the exact canonical JSON snapshot of the advertised tool catalog.
type ModelRequest struct {
	Identity   ModelRequestIdentity  `json:"identity"`
	Segments   []ModelRequestSegment `json:"segments"`
	Tools      json.RawMessage       `json:"tools"`
	InputClaim *InputClaimBinding    `json:"input_claim,omitempty"`
}

// InputClaimBinding joins one exact admitted-input set to the request built
// from it without changing the model-visible request digest.
type InputClaimBinding struct {
	ClaimID     string `json:"claim_id"`
	InputSHA256 string `json:"input_sha256"`
	InputCount  int    `json:"input_count"`
}

// InputClaim is the exact ordered set removed from the native-loop inbox for
// one turn. Each item is the canonical message envelope spliced into history.
type InputClaim struct {
	Turn   int               `json:"turn"`
	Inputs []json.RawMessage `json:"inputs"`
}

// InputClaimState is the closed ownership state recorded in the ledger.
type InputClaimState string

const (
	InputClaimClaimed  InputClaimState = "CLAIMED"
	InputClaimReleased InputClaimState = "RELEASED"
)

// InputClaimReceipt is a bounded ledger row pointing to exact claim bytes.
type InputClaimReceipt struct {
	Schema      string                 `json:"schema"`
	ClaimID     string                 `json:"claim_id"`
	Turn        int                    `json:"turn"`
	State       InputClaimState        `json:"state"`
	Inputs      ModelRequestContentRef `json:"inputs"`
	InputSHA256 string                 `json:"input_sha256"`
	InputCount  int                    `json:"input_count"`
	Reason      string                 `json:"reason,omitempty"`
	LedgerEntry Hash                   `json:"-"`
}

// ModelRequestContentRef is a bounded pointer to immutable CAS bytes.
type ModelRequestContentRef struct {
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// ModelRequestReceipt is the only model-request payload embedded in ledger.jsonl.
// LedgerEntry is populated on append/reconstruction but omitted from the content
// whose hash it names.
type ModelRequestReceipt struct {
	Schema        string                 `json:"schema"`
	RequestID     string                 `json:"request_id"`
	Turn          int                    `json:"turn"`
	Model         string                 `json:"model"`
	Manifest      ModelRequestContentRef `json:"manifest"`
	RequestSHA256 string                 `json:"request_sha256"`
	PromptSHA256  string                 `json:"prompt_sha256"`
	SegmentCount  int                    `json:"segment_count"`
	ToolBytes     int                    `json:"tool_bytes"`
	InputClaim    *InputClaimBinding     `json:"input_claim,omitempty"`
	LedgerEntry   Hash                   `json:"-"`
}

type modelRequestSegmentRef struct {
	Kind    string                 `json:"kind"`
	Source  promptaudit.Source     `json:"source"`
	Content ModelRequestContentRef `json:"content"`
}

type modelRequestManifest struct {
	Schema        string                   `json:"schema"`
	Identity      ModelRequestIdentity     `json:"identity"`
	Segments      []modelRequestSegmentRef `json:"segments"`
	Tools         ModelRequestContentRef   `json:"tools"`
	RequestSHA256 string                   `json:"request_sha256"`
	PromptSHA256  string                   `json:"prompt_sha256"`
	InputClaim    *InputClaimBinding       `json:"input_claim,omitempty"`
}

// ModelRequestMismatchAxis is the closed verifier classification.
type ModelRequestMismatchAxis string

const (
	AxisWireOnly   ModelRequestMismatchAxis = "WIRE_ONLY"
	AxisLedgerOnly ModelRequestMismatchAxis = "LEDGER_ONLY"
	AxisContent    ModelRequestMismatchAxis = "CONTENT"
	AxisTools      ModelRequestMismatchAxis = "TOOLS"
	AxisIdentity   ModelRequestMismatchAxis = "IDENTITY"
)

// ModelRequestMismatch reports where the reconstructed request diverged from
// the independently captured model boundary.
type ModelRequestMismatch struct {
	Axis   ModelRequestMismatchAxis
	Index  int
	Detail string
}

func (e *ModelRequestMismatch) Error() string {
	if e.Index >= 0 {
		return fmt.Sprintf("model request mismatch: %s at segment %d: %s", e.Axis, e.Index, e.Detail)
	}
	return fmt.Sprintf("model request mismatch: %s: %s", e.Axis, e.Detail)
}

// AppendInputClaim durably owns one exact admitted input set before prompt
// assembly begins.
func (l *Ledger) AppendInputClaim(trace string, claim InputClaim) (InputClaimReceipt, error) {
	if strings.TrimSpace(trace) == "" {
		return InputClaimReceipt{}, errors.New("sessionledger: input claim trace is required")
	}
	if err := validateInputClaim(claim); err != nil {
		return InputClaimReceipt{}, err
	}
	id, err := newReceiptID("ic_")
	if err != nil {
		return InputClaimReceipt{}, fmt.Errorf("sessionledger: generate input claim id: %w", err)
	}
	digest, err := CanonicalInputClaimDigest(claim.Inputs)
	if err != nil {
		return InputClaimReceipt{}, err
	}
	inputs, err := json.Marshal(claim.Inputs)
	if err != nil {
		return InputClaimReceipt{}, fmt.Errorf("sessionledger: marshal input claim: %w", err)
	}
	ref, err := l.putModelRequestObject(inputs)
	if err != nil {
		return InputClaimReceipt{}, err
	}
	receipt := InputClaimReceipt{
		Schema: inputClaimSchema, ClaimID: id, Turn: claim.Turn,
		State: InputClaimClaimed, Inputs: ref, InputSHA256: digest,
		InputCount: len(claim.Inputs),
	}
	return l.appendInputClaimReceipt(trace, receipt)
}

// ReleaseInputClaim records a deliberate no-retry outcome. A repeated release
// returns the existing terminal receipt instead of appending a duplicate.
func (l *Ledger) ReleaseInputClaim(trace, claimID, reason string) (InputClaimReceipt, error) {
	reason = strings.TrimSpace(reason)
	if reason == "" || len(reason) > 256 {
		return InputClaimReceipt{}, errors.New("sessionledger: input claim release reason is required and bounded")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	receipt, err := l.inputClaimReceiptLocked(trace, claimID)
	if err != nil {
		return InputClaimReceipt{}, err
	}
	if receipt.State == InputClaimReleased {
		return receipt, nil
	}
	receipt.State = InputClaimReleased
	receipt.Reason = reason
	receipt.LedgerEntry = ""
	content, err := json.Marshal(receipt)
	if err != nil {
		return InputClaimReceipt{}, fmt.Errorf("sessionledger: marshal input claim receipt: %w", err)
	}
	parent := l.heads[trace]
	entry := Entry{Parent: parent, Kind: KindInputClaimReceipt, Content: content}
	entry.Hash = digest(parent, entry.Kind, content)
	l.putNode(entry)
	l.heads[trace] = entry.Hash
	if err := l.appendRecord(record{
		Trace: trace, Hash: entry.Hash, Parent: entry.Parent, Kind: entry.Kind, Content: entry.Content,
	}); err != nil {
		return InputClaimReceipt{}, err
	}
	receipt.LedgerEntry = entry.Hash
	return receipt, nil
}

func (l *Ledger) inputClaimReceiptLocked(trace, claimID string) (InputClaimReceipt, error) {
	head, ok := l.heads[trace]
	if !ok {
		return InputClaimReceipt{}, fmt.Errorf("sessionledger: trace %q not found", trace)
	}
	entries, err := l.chainFromLocked(head)
	if err != nil {
		return InputClaimReceipt{}, err
	}
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].Kind != KindInputClaimReceipt {
			continue
		}
		var receipt InputClaimReceipt
		if err := json.Unmarshal(entries[i].Content, &receipt); err != nil {
			return InputClaimReceipt{}, fmt.Errorf("sessionledger: decode input claim receipt: %w", err)
		}
		if receipt.ClaimID == claimID {
			receipt.LedgerEntry = entries[i].Hash
			return receipt, nil
		}
	}
	return InputClaimReceipt{}, fmt.Errorf("sessionledger: input claim %q not found on trace %q", claimID, trace)
}

func (l *Ledger) appendInputClaimReceipt(trace string, receipt InputClaimReceipt) (InputClaimReceipt, error) {
	content, err := json.Marshal(receipt)
	if err != nil {
		return InputClaimReceipt{}, fmt.Errorf("sessionledger: marshal input claim receipt: %w", err)
	}
	entry, err := l.Append(trace, KindInputClaimReceipt, content)
	if err != nil {
		return InputClaimReceipt{}, err
	}
	receipt.LedgerEntry = entry.Hash
	return receipt, nil
}

// ReconstructInputClaim materializes and verifies the newest state of a claim
// after process reopen.
func (l *Ledger) ReconstructInputClaim(trace, claimID string) (InputClaim, InputClaimReceipt, error) {
	if strings.TrimSpace(claimID) == "" {
		return InputClaim{}, InputClaimReceipt{}, errors.New("sessionledger: input claim id is required")
	}
	entries, err := l.Chain(trace)
	if err != nil {
		return InputClaim{}, InputClaimReceipt{}, err
	}
	for i := len(entries) - 1; i >= 0; i-- {
		entry := entries[i]
		if entry.Kind != KindInputClaimReceipt {
			continue
		}
		var receipt InputClaimReceipt
		if err := json.Unmarshal(entry.Content, &receipt); err != nil {
			return InputClaim{}, InputClaimReceipt{}, fmt.Errorf("sessionledger: decode input claim receipt: %w", err)
		}
		if receipt.ClaimID != claimID {
			continue
		}
		receipt.LedgerEntry = entry.Hash
		claim, err := l.rebuildInputClaim(receipt)
		return claim, receipt, err
	}
	return InputClaim{}, InputClaimReceipt{}, fmt.Errorf("sessionledger: input claim %q not found on trace %q", claimID, trace)
}

func (l *Ledger) rebuildInputClaim(receipt InputClaimReceipt) (InputClaim, error) {
	if receipt.Schema != inputClaimSchema || receipt.ClaimID == "" || receipt.Turn <= 0 ||
		(receipt.State != InputClaimClaimed && receipt.State != InputClaimReleased) {
		return InputClaim{}, errors.New("sessionledger: invalid input claim receipt")
	}
	content, err := l.resolveModelRequestObject(receipt.Inputs)
	if err != nil {
		return InputClaim{}, fmt.Errorf("sessionledger: resolve input claim: %w", err)
	}
	var inputs []json.RawMessage
	if err := json.Unmarshal(content, &inputs); err != nil {
		return InputClaim{}, fmt.Errorf("sessionledger: decode input claim: %w", err)
	}
	claim := InputClaim{Turn: receipt.Turn, Inputs: inputs}
	if err := validateInputClaim(claim); err != nil {
		return InputClaim{}, err
	}
	digest, err := CanonicalInputClaimDigest(inputs)
	if err != nil {
		return InputClaim{}, err
	}
	if digest != receipt.InputSHA256 || len(inputs) != receipt.InputCount {
		return InputClaim{}, errors.New("sessionledger: reconstructed input claim digest or count mismatch")
	}
	return claim, nil
}

// CanonicalInputClaimDigest hashes the exact ordered admitted message envelopes.
func CanonicalInputClaimDigest(inputs []json.RawMessage) (string, error) {
	if len(inputs) == 0 {
		return "", errors.New("sessionledger: input claim requires at least one input")
	}
	h := sha256.New()
	for i, input := range inputs {
		if !json.Valid(input) {
			return "", fmt.Errorf("sessionledger: input claim item %d is invalid", i)
		}
		writeModelRequestFrame(h, "input", input)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

func validateInputClaim(claim InputClaim) error {
	if claim.Turn <= 0 {
		return errors.New("sessionledger: input claim turn must be positive")
	}
	_, err := CanonicalInputClaimDigest(claim.Inputs)
	return err
}

// AppendModelRequest persists every exact payload in the ledger-local CAS before
// appending one bounded receipt. Failure is returned synchronously so a caller can
// refuse the model call rather than advance without evidence.
func (l *Ledger) AppendModelRequest(trace string, req ModelRequest) (ModelRequestReceipt, error) {
	if trace == "" {
		return ModelRequestReceipt{}, errors.New("sessionledger: model request trace is required")
	}
	if err := validateModelRequest(&req); err != nil {
		return ModelRequestReceipt{}, err
	}
	if req.Identity.RequestID == "" {
		id, err := newModelRequestID()
		if err != nil {
			return ModelRequestReceipt{}, err
		}
		req.Identity.RequestID = id
	}
	if req.InputClaim != nil {
		claim, claimReceipt, err := l.ReconstructInputClaim(trace, req.InputClaim.ClaimID)
		if err != nil {
			return ModelRequestReceipt{}, fmt.Errorf("sessionledger: bind input claim: %w", err)
		}
		if claimReceipt.State != InputClaimClaimed || claimReceipt.InputSHA256 != req.InputClaim.InputSHA256 ||
			claimReceipt.InputCount != req.InputClaim.InputCount {
			return ModelRequestReceipt{}, errors.New("sessionledger: model request input claim binding mismatch")
		}
		if !inputClaimSurvivesRequest(claim.Inputs, req.Segments) {
			return ModelRequestReceipt{}, errors.New("sessionledger: model request does not contain the exact claimed inputs")
		}
	}

	requestDigest, err := CanonicalModelRequestDigest(req)
	if err != nil {
		return ModelRequestReceipt{}, err
	}
	promptDigest, err := modelRequestPromptDigest(req)
	if err != nil {
		return ModelRequestReceipt{}, err
	}
	manifest := modelRequestManifest{
		Schema: modelRequestSchema, Identity: req.Identity,
		Segments:      make([]modelRequestSegmentRef, 0, len(req.Segments)),
		RequestSHA256: requestDigest, PromptSHA256: promptDigest,
		InputClaim: cloneInputClaimBinding(req.InputClaim),
	}
	for _, segment := range req.Segments {
		ref, err := l.putModelRequestObject(segment.Content)
		if err != nil {
			return ModelRequestReceipt{}, err
		}
		manifest.Segments = append(manifest.Segments, modelRequestSegmentRef{
			Kind: segment.Kind, Source: segment.Source, Content: ref,
		})
	}
	manifest.Tools, err = l.putModelRequestObject(req.Tools)
	if err != nil {
		return ModelRequestReceipt{}, err
	}
	manifestBytes, err := json.Marshal(manifest)
	if err != nil {
		return ModelRequestReceipt{}, fmt.Errorf("sessionledger: marshal model request manifest: %w", err)
	}
	manifestRef, err := l.putModelRequestObject(manifestBytes)
	if err != nil {
		return ModelRequestReceipt{}, err
	}
	receipt := ModelRequestReceipt{
		Schema: modelRequestSchema, RequestID: req.Identity.RequestID,
		Turn: req.Identity.Turn, Model: req.Identity.Model, Manifest: manifestRef,
		RequestSHA256: requestDigest, PromptSHA256: promptDigest,
		SegmentCount: len(req.Segments), ToolBytes: len(req.Tools),
		InputClaim: cloneInputClaimBinding(req.InputClaim),
	}
	receiptBytes, err := json.Marshal(receipt)
	if err != nil {
		return ModelRequestReceipt{}, fmt.Errorf("sessionledger: marshal model request receipt: %w", err)
	}
	entry, err := l.Append(trace, KindModelRequestReceipt, receiptBytes)
	if err != nil {
		return ModelRequestReceipt{}, err
	}
	receipt.LedgerEntry = entry.Hash
	return receipt, nil
}

// ReconstructModelRequest materializes and verifies one receipt after process
// reopen. Empty requestID selects the newest receipt on the trace.
func (l *Ledger) ReconstructModelRequest(trace, requestID string) (ModelRequest, ModelRequestReceipt, error) {
	entries, err := l.Chain(trace)
	if err != nil {
		return ModelRequest{}, ModelRequestReceipt{}, err
	}
	for i := len(entries) - 1; i >= 0; i-- {
		entry := entries[i]
		if entry.Kind != KindModelRequestReceipt {
			continue
		}
		var receipt ModelRequestReceipt
		if err := json.Unmarshal(entry.Content, &receipt); err != nil {
			return ModelRequest{}, ModelRequestReceipt{}, fmt.Errorf("sessionledger: decode model request receipt: %w", err)
		}
		if requestID != "" && receipt.RequestID != requestID {
			continue
		}
		receipt.LedgerEntry = entry.Hash
		request, err := l.rebuildModelRequest(receipt)
		return request, receipt, err
	}
	if requestID == "" {
		return ModelRequest{}, ModelRequestReceipt{}, fmt.Errorf("sessionledger: no model request receipt for trace %q", trace)
	}
	return ModelRequest{}, ModelRequestReceipt{}, fmt.Errorf("sessionledger: model request %q not found on trace %q", requestID, trace)
}

func (l *Ledger) rebuildModelRequest(receipt ModelRequestReceipt) (ModelRequest, error) {
	if receipt.Schema != modelRequestSchema {
		return ModelRequest{}, fmt.Errorf("sessionledger: unsupported model request receipt schema %q", receipt.Schema)
	}
	manifestBytes, err := l.resolveModelRequestObject(receipt.Manifest)
	if err != nil {
		return ModelRequest{}, fmt.Errorf("sessionledger: resolve model request manifest: %w", err)
	}
	var manifest modelRequestManifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return ModelRequest{}, fmt.Errorf("sessionledger: decode model request manifest: %w", err)
	}
	if manifest.Schema != modelRequestSchema || manifest.Identity.RequestID != receipt.RequestID ||
		manifest.Identity.Model != receipt.Model || manifest.Identity.Turn != receipt.Turn {
		return ModelRequest{}, errors.New("sessionledger: receipt/manifest identity mismatch")
	}
	request := ModelRequest{
		Identity:   manifest.Identity,
		Segments:   make([]ModelRequestSegment, 0, len(manifest.Segments)),
		InputClaim: cloneInputClaimBinding(manifest.InputClaim),
	}
	if !equalInputClaimBinding(manifest.InputClaim, receipt.InputClaim) {
		return ModelRequest{}, errors.New("sessionledger: receipt/manifest input claim mismatch")
	}
	for _, segment := range manifest.Segments {
		content, err := l.resolveModelRequestObject(segment.Content)
		if err != nil {
			return ModelRequest{}, fmt.Errorf("sessionledger: resolve model request segment: %w", err)
		}
		request.Segments = append(request.Segments, ModelRequestSegment{
			Kind: segment.Kind, Source: segment.Source, Content: content,
		})
	}
	request.Tools, err = l.resolveModelRequestObject(manifest.Tools)
	if err != nil {
		return ModelRequest{}, fmt.Errorf("sessionledger: resolve model request tools: %w", err)
	}
	requestDigest, err := CanonicalModelRequestDigest(request)
	if err != nil {
		return ModelRequest{}, err
	}
	promptDigest, err := modelRequestPromptDigest(request)
	if err != nil {
		return ModelRequest{}, err
	}
	if requestDigest != receipt.RequestSHA256 || requestDigest != manifest.RequestSHA256 {
		return ModelRequest{}, errors.New("sessionledger: reconstructed model request digest mismatch")
	}
	if promptDigest != receipt.PromptSHA256 || promptDigest != manifest.PromptSHA256 {
		return ModelRequest{}, errors.New("sessionledger: reconstructed prompt digest mismatch")
	}
	if len(request.Segments) != receipt.SegmentCount || len(request.Tools) != receipt.ToolBytes {
		return ModelRequest{}, errors.New("sessionledger: reconstructed model request size mismatch")
	}
	return request, nil
}

// CanonicalModelRequestDigest hashes the exact model-visible logical request.
// Durable request identity and provenance labels are excluded; model/sampling,
// ordered message envelopes, and the tool snapshot are length-framed exactly.
func CanonicalModelRequestDigest(req ModelRequest) (string, error) {
	if err := validateModelRequest(&req); err != nil {
		return "", err
	}
	h := sha256.New()
	writeModelRequestFrame(h, "model", []byte(req.Identity.Model))
	writeModelRequestFrame(h, "stream", []byte(strconv.FormatBool(req.Identity.Stream)))
	writeModelRequestFrame(h, "max_tokens", []byte(strconv.Itoa(req.Identity.MaxTokens)))
	for _, segment := range req.Segments {
		writeModelRequestFrame(h, "message", segment.Content)
	}
	writeModelRequestFrame(h, "tools", req.Tools)
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

func writeModelRequestFrame(h hash.Hash, field string, value []byte) {
	fmt.Fprintf(h, "%d:%s:%d:", len(field), field, len(value))
	_, _ = h.Write(value)
}

func modelRequestPromptDigest(req ModelRequest) (string, error) {
	segments := make([]promptaudit.Segment, 0, len(req.Segments))
	for i, segment := range req.Segments {
		var message struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal(segment.Content, &message); err != nil {
			return "", fmt.Errorf("sessionledger: decode model request segment %d for prompt audit: %w", i, err)
		}
		segments = append(segments, promptaudit.Segment{
			Field: segment.Kind, Source: segment.Source, Raw: message.Content,
		})
	}
	return promptaudit.Audit(segments).Digest(), nil
}

func validateModelRequest(req *ModelRequest) error {
	if req == nil || strings.TrimSpace(req.Identity.Model) == "" {
		return errors.New("sessionledger: model request model is required")
	}
	if len(req.Identity.Model) > 1024 || len(req.Identity.RequestID) > 256 {
		return errors.New("sessionledger: model request identity is too large")
	}
	if req.Identity.Turn <= 0 {
		return errors.New("sessionledger: model request turn must be positive")
	}
	if len(req.Segments) == 0 {
		return errors.New("sessionledger: model request requires at least one segment")
	}
	for i := range req.Segments {
		segment := &req.Segments[i]
		if segment.Kind == "" || !json.Valid(segment.Content) {
			return fmt.Errorf("sessionledger: model request segment %d is invalid", i)
		}
		if !segment.Source.Valid() {
			segment.Source = promptaudit.SourceUnknown
		}
	}
	if !json.Valid(req.Tools) {
		return errors.New("sessionledger: model request tools must be JSON")
	}
	if req.InputClaim != nil && (strings.TrimSpace(req.InputClaim.ClaimID) == "" ||
		strings.TrimSpace(req.InputClaim.InputSHA256) == "" || req.InputClaim.InputCount <= 0) {
		return errors.New("sessionledger: invalid model request input claim binding")
	}
	return nil
}

func newModelRequestID() (string, error) {
	return newReceiptID("mr_")
}

func newReceiptID(prefix string) (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(b[:]), nil
}

func cloneInputClaimBinding(binding *InputClaimBinding) *InputClaimBinding {
	if binding == nil {
		return nil
	}
	copy := *binding
	return &copy
}

func equalInputClaimBinding(a, b *InputClaimBinding) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func inputClaimSurvivesRequest(inputs []json.RawMessage, segments []ModelRequestSegment) bool {
	matched := 0
	for _, segment := range segments {
		if matched < len(inputs) && bytes.Equal(inputs[matched], segment.Content) {
			matched++
		}
	}
	return matched == len(inputs)
}

func (l *Ledger) putModelRequestObject(content []byte) (ModelRequestContentRef, error) {
	sum := sha256.Sum256(content)
	digest := hex.EncodeToString(sum[:])
	if l.objectStore == nil {
		l.mu.Lock()
		if l.objects == nil {
			l.objects = map[string][]byte{}
		}
		if _, ok := l.objects[digest]; !ok {
			l.objects[digest] = bytes.Clone(content)
		}
		l.mu.Unlock()
		return ModelRequestContentRef{SHA256: "sha256:" + digest, Bytes: int64(len(content))}, nil
	}
	ref, err := l.objectStore.Put(context.Background(), content)
	if err != nil {
		return ModelRequestContentRef{}, fmt.Errorf("sessionledger: store model request object: %w", err)
	}
	if ref.Kind == abi.RefInline {
		ref, err = l.objectStore.PageOut(context.Background(), ref)
		if err != nil {
			return ModelRequestContentRef{}, fmt.Errorf("sessionledger: page out model request object: %w", err)
		}
	}
	return ModelRequestContentRef{SHA256: "sha256:" + ref.Digest, Bytes: ref.Len}, nil
}

func (l *Ledger) resolveModelRequestObject(ref ModelRequestContentRef) ([]byte, error) {
	digest := strings.TrimPrefix(ref.SHA256, "sha256:")
	if len(digest) != 64 || ref.Bytes < 0 {
		return nil, errors.New("invalid content reference")
	}
	var content []byte
	var err error
	if l.objectStore == nil {
		l.mu.RLock()
		content = bytes.Clone(l.objects[digest])
		l.mu.RUnlock()
		if content == nil {
			return nil, fmt.Errorf("unknown content digest %s", ref.SHA256)
		}
	} else {
		content, err = l.objectStore.Resolve(context.Background(), abi.Ref{
			Kind: abi.RefBlob, Digest: digest, Len: ref.Bytes,
		})
		if err != nil {
			return nil, err
		}
	}
	if int64(len(content)) != ref.Bytes {
		return nil, fmt.Errorf("content length mismatch for %s", ref.SHA256)
	}
	sum := sha256.Sum256(content)
	if hex.EncodeToString(sum[:]) != digest {
		return nil, fmt.Errorf("content digest mismatch for %s", ref.SHA256)
	}
	return content, nil
}

// VerifyModelRequest compares an independently captured wire request with a
// reconstructed ledger request and classifies missing/invented segments.
func VerifyModelRequest(wire, ledger ModelRequest) error {
	if wire.Identity.Model != ledger.Identity.Model || wire.Identity.Turn != ledger.Identity.Turn ||
		wire.Identity.Stream != ledger.Identity.Stream || wire.Identity.MaxTokens != ledger.Identity.MaxTokens {
		return &ModelRequestMismatch{Axis: AxisIdentity, Index: -1, Detail: "model, turn, stream, or max_tokens differs"}
	}
	if !bytes.Equal(wire.Tools, ledger.Tools) {
		return &ModelRequestMismatch{Axis: AxisTools, Index: -1, Detail: "tool snapshot differs"}
	}
	wireSegments := modelRequestSegmentBytes(wire.Segments)
	ledgerSegments := modelRequestSegmentBytes(ledger.Segments)
	if equalModelRequestSegments(wireSegments, ledgerSegments) {
		return nil
	}
	if modelRequestSubsequence(ledgerSegments, wireSegments) {
		return &ModelRequestMismatch{Axis: AxisWireOnly, Index: firstModelRequestDifference(wireSegments, ledgerSegments), Detail: "segment reached the model but not the ledger"}
	}
	if modelRequestSubsequence(wireSegments, ledgerSegments) {
		return &ModelRequestMismatch{Axis: AxisLedgerOnly, Index: firstModelRequestDifference(wireSegments, ledgerSegments), Detail: "segment exists in the ledger but not at the model boundary"}
	}
	return &ModelRequestMismatch{Axis: AxisContent, Index: firstModelRequestDifference(wireSegments, ledgerSegments), Detail: "ordered segment content differs"}
}

func modelRequestSegmentBytes(segments []ModelRequestSegment) [][]byte {
	out := make([][]byte, len(segments))
	for i := range segments {
		out[i] = segments[i].Content
	}
	return out
}

func equalModelRequestSegments(a, b [][]byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !bytes.Equal(a[i], b[i]) {
			return false
		}
	}
	return true
}

func modelRequestSubsequence(needle, haystack [][]byte) bool {
	i := 0
	for _, segment := range haystack {
		if i < len(needle) && bytes.Equal(needle[i], segment) {
			i++
		}
	}
	return i == len(needle)
}

func firstModelRequestDifference(a, b [][]byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if !bytes.Equal(a[i], b[i]) {
			return i
		}
	}
	return n
}
