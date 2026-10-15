package agent

import (
	"bytes"
	"context"
	"net/http"
	"sync"
)

// PrepareRequest shares inference request preparation without retries or parsing.
// Preparation can acquire resources; release after handling the response.
func (p *HTTPPlanner) PrepareRequest(ctx context.Context, messages []Message, tools []ToolDef, stream bool, opts ...SampleOpt) (*http.Request, func(), error) {
	call, err := p.prepareUpstreamContext(ctx, messages, tools, stream, opts...)
	if err != nil {
		return nil, nil, err
	}
	var once sync.Once
	release := func() { once.Do(call.releaseSlot) }
	if err := call.admitSoftResidency(ctx, p); err != nil {
		release()
		return nil, nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, call.url, bytes.NewReader(call.body))
	if err != nil {
		release()
		return nil, nil, err
	}
	call.applyHeaders(req)
	if stream || call.responsesStreamed {
		req.Header.Set("Accept", "text/event-stream")
	}
	return req, release, nil
}
