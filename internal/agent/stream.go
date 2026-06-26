package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
)

// StreamSink receives incremental assistant CONTENT fragments as they arrive from
// the upstream model. It is the live half of a streamed turn: each call carries the
// next chunk of natural-language output the moment the provider emits it, so a
// downstream client sees a real time-to-first-token instead of waiting for the whole
// turn to finish.
//
// Tool-call deltas are deliberately NOT delivered here — they are buffered inside
// CompleteStream and returned in the final Completion, so the caller (the gateway)
// can route every proposed call through the kernel's adjudication BEFORE the client
// ever sees it. Streaming a tool call live would bypass that gate; streaming content
// does not, because content is the model's own prose, which the buffered path
// forwards verbatim too. A non-nil error returned by the sink aborts the stream
// (e.g. the client disconnected) and surfaces from CompleteStream.
type StreamSink func(contentDelta string) error

// StreamingPlanner is the optional capability a Planner advertises when it can
// stream the upstream completion token-by-token. It is a strict superset of Planner:
// CompleteStream behaves exactly like Complete (same sampling, same quarantine, same
// adjudication-relevant return shape) but invokes sink for each content fragment as
// it arrives, then returns the fully-accumulated Completion (content + buffered tool
// calls + usage + finish reason). A Planner that cannot stream simply does not
// implement this interface, and the gateway falls back to its buffered path.
type StreamingPlanner interface {
	Planner
	// StreamingSupported reports whether a live token stream is available for the
	// planner's CURRENT configuration (e.g. the OpenAI-compatible chat wire), so the
	// gateway can decide to take the streaming path WITHOUT committing to a request
	// it would have to unwind. False means callers must use Complete.
	StreamingSupported() bool
	// CompleteStream is Complete with a live content sink. On a planner whose wire
	// does not support streaming it returns ErrStreamingUnsupported without touching
	// the network, so the caller can fall back having written nothing.
	CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error)
}

// ErrStreamingUnsupported is returned by CompleteStream when the planner's wire
// cannot stream (every non-OpenAI-compatible provider, for now). It is a sentinel so
// the gateway can distinguish "this wire can't stream, fall back cleanly" from a real
// upstream failure.
var ErrStreamingUnsupported = errors.New("agent: streaming not supported for this provider wire")

// upstreamCall is the fully-resolved input to one upstream round-trip — the shared
// product of the buffered Complete and the streaming CompleteStream. Extracting it
// guarantees both paths apply the SAME pre-send quarantine, coherence shaping,
// sampling resolution, request-model + credential pass-through, and (Anthropic)
// raw-body passthrough, so a streamed request differs from a buffered one ONLY by the
// `stream` flag in the body.
type upstreamCall struct {
	adapter      TranscriptAdapter
	url          string
	body         []byte
	apiKey       string
	upstreamBeta string
	quarantined  int
	redacted     int                   // rung 5 (#572): messages whose content was span-redacted pre-send
	redactions   []TranscriptRedaction // the full reversible records (CAS Original) behind that count (#882)
}

// headers builds the per-request header set, applying the Anthropic-wire beta union
// (the inbound client's negotiated betas merged with any the auth scheme required).
// It mirrors the header logic Complete ran inline before the extraction.
func (c *upstreamCall) headers() map[string]string {
	h := c.adapter.Headers(c.apiKey)
	if c.upstreamBeta != "" && c.adapter.Provider() == ProviderAnthropic {
		h["anthropic-beta"] = mergeBeta(h["anthropic-beta"], c.upstreamBeta)
	}
	return h
}

// prepareUpstream resolves messages+tools+opts into a single upstreamCall. stream
// selects whether the marshaled body asks the provider to deliver an SSE token
// stream (honored only by the OpenAI-compatible chat wire; other adapters ignore the
// flag, so an unsupported-wire body is byte-for-byte the non-stream one).
func (p *HTTPPlanner) prepareUpstream(messages []Message, tools []ToolDef, stream bool, opts ...SampleOpt) (*upstreamCall, error) {
	adapter, err := p.transcriptAdapter()
	if err != nil {
		return nil, err
	}
	safeMessages := messages
	var quarantines []TranscriptQuarantine
	if p.QuarantineTranscript {
		safeMessages, quarantines = QuarantineOutboundMessages(messages)
	}
	// §A4 coherence shaping: after the safety quarantine, give the coherence layer a
	// chance to break the provider prefix when a world witness has been refuted. nil
	// hook = unchanged path; applied copy-on-shape so the caller's slice is untouched.
	if p.CoherenceShaper != nil {
		safeMessages = p.CoherenceShaper(safeMessages)
	}
	sp := applySampleOpts(opts...)
	// Request-model pass-through (#82): a client-supplied model wins for THIS turn.
	modelID := p.ModelID
	if sp.Model != "" {
		modelID = sp.Model
	}
	maxTokens := p.MaxTokens
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxTokens = *sp.MaxTokens
	}
	temperature := p.Temperature
	if sp.Temperature != nil {
		temperature = *sp.Temperature
	}
	// Passthrough: forward the client's ORIGINAL bytes so its prompt-cache prefix
	// survives (a real upstream cache hit). The provider gate is load-bearing:
	// RawRequestBody is Anthropic-shaped, so it must never reach an OpenAI/Gemini/xAI
	// endpoint — only the anthropic→anthropic proxy forwards it. The upstream hop is
	// non-streaming (the gateway re-synthesizes the SSE), so stream:true in the raw
	// body is forced off or the buffered ParseResponse chokes on the SSE reply — the
	// same fix Complete applied inline before the extraction.
	var reqBody []byte
	redactedN := 0
	var redactions []TranscriptRedaction
	if len(sp.RawRequestBody) > 0 && adapter.Provider() == ProviderAnthropic {
		reqBody = forceAnthropicNonStreaming(sp.RawRequestBody)
	} else {
		// Rung 5 (#572): span-level PII/secret redaction on the non-passthrough
		// re-marshal path. The Anthropic passthrough above forwards req.Raw verbatim
		// and never serializes these messages, so redaction runs ONLY here, where the
		// re-marshal can carry it to the wire. Default-inert: with FAK_WIRE_REDACT
		// unset, RedactOutboundMessages returns safeMessages unchanged at zero cost.
		safeMessages, redactions = RedactOutboundMessages(safeMessages)
		redactedN = len(redactions)
		reqBody, err = adapter.MarshalRequest(adapterRequest{
			Model:          modelID,
			Messages:       safeMessages,
			Tools:          tools,
			Temperature:    temperature,
			MaxTokens:      maxTokens,
			TopP:           sp.TopP,
			TopK:           sp.TopK,
			Stop:           sp.Stop,
			ResponseFormat: sp.ResponseFormat,
			LogitBias:      sp.LogitBias,
			ExtraBody:      p.ExtraBody,
			Stream:         stream,
		})
		if err != nil {
			return nil, err
		}
	}
	// Transparent hop: when the inbound client supplied its own upstream credential
	// (passthrough), authenticate with THAT key rather than the planner's.
	apiKey := p.APIKey
	if sp.UpstreamAPIKey != "" {
		apiKey = sp.UpstreamAPIKey
	}
	return &upstreamCall{
		adapter:      adapter,
		url:          adapter.Endpoint(p.BaseURL, modelID),
		body:         reqBody,
		apiKey:       apiKey,
		upstreamBeta: sp.UpstreamBeta,
		quarantined:  len(quarantines),
		redacted:     redactedN,
		redactions:   redactions,
	}, nil
}

// StreamingSupported reports whether the planner's configured wire can stream. Only
// the OpenAI-compatible chat wire (OpenAI and the xAI/vLLM/SGLang-compatible servers
// that share its SSE delta format) is wired today; every other provider returns false
// so the gateway keeps its buffered path for them.
func (p *HTTPPlanner) StreamingSupported() bool {
	switch p.Provider {
	case ProviderOpenAI, ProviderXAI, "":
		return true
	default:
		return false
	}
}

// openAIStreamChunk is one OpenAI-compatible `chat.completion.chunk` SSE event. Each
// `data:` line carries one of these; the terminal `data: [DONE]` carries none. The
// delta fields are all optional and additive across chunks: content fragments
// concatenate into the final text, and tool-call fragments accumulate by index (id +
// name on first sight, arguments concatenated). Usage rides the final chunk only when
// the request asked for it (stream_options.include_usage).
type openAIStreamChunk struct {
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *Usage `json:"usage"`
}

// CompleteStream performs one streamed chat-completions round-trip on the
// OpenAI-compatible wire: it forwards each content fragment to sink as it arrives and
// returns the fully-assembled Completion (content + buffered tool calls + usage +
// finish) once the upstream closes the stream. Tool calls are accumulated, NEVER
// streamed, so the caller can adjudicate them before exposing them.
//
// Unlike Complete it does not retry mid-stream: a connection/status failure surfaces
// before any sink call (so the caller can still choose an HTTP status), but once
// bytes have flowed a read error is returned as-is. A non-OpenAI wire returns
// ErrStreamingUnsupported without a network call.
func (p *HTTPPlanner) CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	if !p.StreamingSupported() {
		return nil, ErrStreamingUnsupported
	}
	call, err := p.prepareUpstream(messages, tools, true, opts...)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", call.url, bytes.NewReader(call.body))
	if err != nil {
		return nil, err
	}
	for k, v := range call.headers() {
		if v != "" {
			req.Header.Set(k, v)
		}
	}
	req.Header.Set("Accept", "text/event-stream")
	resp, err := p.Client.Do(req)
	if err != nil {
		if deterministicTransportError(err) {
			return nil, &UpstreamUnreachableError{Err: err}
		}
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, &UpstreamStatusError{Status: resp.StatusCode, Body: truncate(raw, 400)}
	}

	// Some "OpenAI-compatible" servers ignore stream:true and answer with a single
	// buffered JSON body. Detect that by content-type and fall back to the buffered
	// parser — deliver the whole content as one fragment — so the client gets the
	// correct (if not incremental) turn instead of an empty stream.
	if ct := resp.Header.Get("Content-Type"); ct != "" && !strings.Contains(ct, "event-stream") {
		raw, rerr := io.ReadAll(resp.Body)
		if rerr != nil {
			return nil, fmt.Errorf("planner: %s: read body: %w", call.adapter.Provider(), rerr)
		}
		comp, perr := call.adapter.ParseResponse(raw)
		if perr != nil {
			return nil, fmt.Errorf("planner: %s: %w", call.adapter.Provider(), perr)
		}
		comp = normalizeCompletionToolCalls(comp)
		if sink != nil && comp.Message.Content != "" {
			if serr := sink(comp.Message.Content); serr != nil {
				return nil, serr
			}
		}
		p.attachProviderCacheTelemetry(comp, call.body, call.adapter.Provider())
		comp.Raw = raw
		comp.PreSendQuarantines = call.quarantined
		comp.PreSendRedactions = call.redacted
		comp.PreSendRedactionRecords = call.redactions
		return comp, nil
	}

	var (
		content strings.Builder
		rawBuf  bytes.Buffer // reconstructs the wire transcript for Completion.Raw
		toolAcc = map[int]*ToolCall{}
		usage   Usage
		model   string
		finish  string
	)
	sc := bufio.NewScanner(resp.Body)
	// A single SSE data line can carry a large tool-call argument fragment; raise the
	// scanner ceiling well past the 64 KiB default so a big chunk is never truncated.
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		// Keep the raw frames so the streamed Completion carries the same Raw transcript
		// witness the buffered path does (the bytes are otherwise consumed line-by-line).
		rawBuf.Write(sc.Bytes())
		rawBuf.WriteByte('\n')
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue // SSE comments / event: lines / blank separators
		}
		data = strings.TrimSpace(data)
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // tolerate a keep-alive or non-JSON heartbeat line
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
		for _, ch := range chunk.Choices {
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
			if ch.Delta.Content != "" {
				content.WriteString(ch.Delta.Content)
				if sink != nil {
					if err := sink(ch.Delta.Content); err != nil {
						return nil, err
					}
				}
			}
			for _, tcd := range ch.Delta.ToolCalls {
				acc := toolAcc[tcd.Index]
				if acc == nil {
					acc = &ToolCall{Type: "function"}
					toolAcc[tcd.Index] = acc
				}
				if tcd.ID != "" {
					acc.ID = tcd.ID
				}
				if tcd.Type != "" {
					acc.Type = tcd.Type
				}
				if tcd.Function.Name != "" {
					acc.Function.Name = tcd.Function.Name
				}
				acc.Function.Arguments += tcd.Function.Arguments
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("planner: %s: stream read: %w", call.adapter.Provider(), err)
	}

	calls := make([]ToolCall, 0, len(toolAcc))
	for _, idx := range sortedIndices(toolAcc) {
		calls = append(calls, *toolAcc[idx])
	}
	comp := normalizeCompletionToolCalls(&Completion{
		Message:      Message{Role: RoleAssistant, Content: content.String(), ToolCalls: calls},
		FinishReason: finish,
		Usage:        usage,
		Model:        model,
	})
	p.attachProviderCacheTelemetry(comp, call.body, call.adapter.Provider())
	comp.Raw = rawBuf.Bytes()
	comp.PreSendQuarantines = call.quarantined
	comp.PreSendRedactions = call.redacted
	comp.PreSendRedactionRecords = call.redactions
	return comp, nil
}

func sortedIndices(m map[int]*ToolCall) []int {
	idx := make([]int, 0, len(m))
	for k := range m {
		idx = append(idx, k)
	}
	sort.Ints(idx)
	return idx
}
