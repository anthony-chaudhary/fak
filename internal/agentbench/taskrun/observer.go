package taskrun

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

const maxObservedResponse = 16 << 20

type ModelObservation struct {
	Requests                   int      `json:"requests"`
	UpstreamRequests           int      `json:"upstream_requests"`
	SuccessfulRequests         int      `json:"successful_requests"`
	PeakHTTPInFlight           int      `json:"peak_http_in_flight"`
	InternalGPUConcurrency     *int     `json:"internal_gpu_concurrency,omitempty"`
	LimitExceeded              bool     `json:"limit_exceeded"`
	ExpectedModel              string   `json:"expected_model"`
	ObservedModels             []string `json:"observed_models,omitempty"`
	ModelIdentityStatus        string   `json:"model_identity_status"`
	IdentityKnownRequests      int      `json:"identity_known_requests"`
	IdentityUnknownRequests    int      `json:"identity_unknown_requests"`
	IdentityMismatchedRequests int      `json:"identity_mismatched_requests"`
	UsageKnownRequests         int      `json:"usage_known_requests"`
	UsageUnknownRequests       int      `json:"usage_unknown_requests"`
	UsageInvalidRequests       int      `json:"usage_invalid_requests"`
	EventsPath                 string   `json:"events_path"`
	EventsDigest               string   `json:"events_digest,omitempty"`
}

type modelObserver struct {
	parent                                                                                                                                     context.Context
	ctx                                                                                                                                        context.Context
	cancel                                                                                                                                     context.CancelFunc
	listener                                                                                                                                   net.Listener
	server                                                                                                                                     *http.Server
	client                                                                                                                                     *http.Client
	target                                                                                                                                     *url.URL
	expected, endpoint, eventsPath                                                                                                             string
	max                                                                                                                                        int
	mu                                                                                                                                         sync.Mutex
	events                                                                                                                                     *os.File
	requests, inflight, peak, upstream, successful, identityKnown, identityUnknown, identityMismatched, usageKnown, usageUnknown, usageInvalid int
	exceeded                                                                                                                                   bool
	models                                                                                                                                     map[string]struct{}
	closed                                                                                                                                     bool
	result                                                                                                                                     ModelObservation
	closeErr                                                                                                                                   error
	meter                                                                                                                                      *modelConcurrencyMeter
}

type modelConcurrencyMeter struct {
	mu             sync.Mutex
	inflight, peak int
}

func (m *modelConcurrencyMeter) start() {
	m.mu.Lock()
	m.inflight++
	if m.inflight > m.peak {
		m.peak = m.inflight
	}
	m.mu.Unlock()
}
func (m *modelConcurrencyMeter) done()             { m.mu.Lock(); m.inflight--; m.mu.Unlock() }
func (m *modelConcurrencyMeter) observedPeak() int { m.mu.Lock(); defer m.mu.Unlock(); return m.peak }

type observerEvent struct {
	Request int       `json:"request"`
	Stage   string    `json:"stage"`
	Status  int       `json:"status,omitempty"`
	Error   string    `json:"error,omitempty"`
	At      time.Time `json:"at"`
}

func startModelObserver(parent context.Context, upstream, expectedModel, artifactPath string, maxRequests int) (*modelObserver, error) {
	if parent == nil || strings.TrimSpace(expectedModel) == "" || maxRequests <= 0 {
		return nil, errors.New("model observer: context, expected model, and positive request limit required")
	}
	u, err := url.Parse(upstream)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("model observer: invalid upstream endpoint")
	}
	base := strings.TrimRight(u.Path, "/")
	if strings.HasSuffix(base, "/v1") {
		base += "/chat/completions"
	} else {
		base += "/v1/chat/completions"
	}
	u.Path, u.RawPath = base, ""
	if err := os.MkdirAll(filepathDir(artifactPath), 0700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(artifactPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		f.Close()
		return nil, err
	}
	ctx, cancel := context.WithCancel(parent)
	o := &modelObserver{parent: parent, ctx: ctx, cancel: cancel, listener: ln, client: &http.Client{Timeout: 90 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}, target: u, expected: expectedModel, endpoint: "http://" + ln.Addr().String(), eventsPath: artifactPath, max: maxRequests, events: f, models: map[string]struct{}{}}
	o.server = &http.Server{Handler: http.HandlerFunc(o.serve), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 2 * time.Minute, IdleTimeout: 30 * time.Second}
	go func() { _ = o.server.Serve(ln) }()
	go func() { <-ctx.Done(); _ = o.server.Close() }()
	return o, nil
}

func filepathDir(path string) string {
	i := strings.LastIndex(path, string(os.PathSeparator))
	if i < 0 {
		return "."
	}
	return path[:i]
}
func (o *modelObserver) Endpoint() string { return o.endpoint }

func (o *modelObserver) serve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost || (r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/chat/completions") || r.URL.RawQuery != "" {
		http.Error(w, "forbidden observer route", http.StatusNotFound)
		return
	}
	o.mu.Lock()
	o.requests++
	id := o.requests
	denied := id > o.max
	if denied {
		o.exceeded = true
	}
	o.writeEventLocked(observerEvent{Request: id, Stage: "start", At: time.Now().UTC()})
	o.mu.Unlock()
	status := http.StatusBadGateway
	terminalErr := ""
	defer func() {
		o.mu.Lock()
		o.writeEventLocked(observerEvent{Request: id, Stage: "terminal", Status: status, Error: terminalErr, At: time.Now().UTC()})
		o.mu.Unlock()
	}()
	if denied {
		status = http.StatusTooManyRequests
		terminalErr = "request_limit_exceeded"
		http.Error(w, "model request limit exceeded", status)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxObservedResponse+1))
	if err != nil || len(body) > maxObservedResponse {
		status = http.StatusRequestEntityTooLarge
		terminalErr = "request_body_unavailable"
		http.Error(w, terminalErr, status)
		return
	}
	req, err := http.NewRequestWithContext(o.ctx, http.MethodPost, o.target.String(), bytes.NewReader(body))
	if err != nil {
		terminalErr = "upstream_request_failed"
		http.Error(w, terminalErr, status)
		return
	}
	req.Header = r.Header.Clone()
	o.mu.Lock()
	o.upstream++
	o.inflight++
	if o.inflight > o.peak {
		o.peak = o.inflight
	}
	meter := o.meter
	o.mu.Unlock()
	if meter != nil {
		meter.start()
		defer meter.done()
	}
	defer func() { o.mu.Lock(); o.inflight--; o.mu.Unlock() }()
	resp, err := o.client.Do(req)
	if err != nil {
		terminalErr = "upstream_unavailable"
		http.Error(w, terminalErr, status)
		return
	}
	defer resp.Body.Close()
	response, err := io.ReadAll(io.LimitReader(resp.Body, maxObservedResponse+1))
	if err != nil || len(response) > maxObservedResponse {
		terminalErr = "upstream_response_too_large"
		http.Error(w, terminalErr, status)
		return
	}
	status = resp.StatusCode
	if status >= 200 && status < 300 {
		o.mu.Lock()
		o.successful++
		o.mu.Unlock()
		o.observeResponse(response)
	}
	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(status)
	_, _ = w.Write(response)
}

func (o *modelObserver) observeResponse(body []byte) {
	var v struct {
		Model string          `json:"model"`
		Usage json.RawMessage `json:"usage"`
	}
	decodeErr := json.Unmarshal(body, &v)
	o.mu.Lock()
	defer o.mu.Unlock()
	model := strings.TrimSpace(v.Model)
	switch {
	case decodeErr != nil || model == "":
		o.identityUnknown++
	case model != o.expected:
		o.identityMismatched++
		o.models[model] = struct{}{}
	default:
		o.identityKnown++
		o.models[model] = struct{}{}
	}
	known, invalid := validObservedUsage(v.Usage, decodeErr)
	if known {
		o.usageKnown++
	} else {
		o.usageUnknown++
		if invalid {
			o.usageInvalid++
		}
	}
}

func validObservedUsage(raw json.RawMessage, responseErr error) (known, invalid bool) {
	if responseErr != nil {
		return false, true
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return false, false
	}
	var usage struct {
		Prompt     *int `json:"prompt_tokens"`
		Completion *int `json:"completion_tokens"`
		Total      *int `json:"total_tokens"`
	}
	if err := json.Unmarshal(trimmed, &usage); err != nil || usage.Prompt == nil || usage.Completion == nil || usage.Total == nil {
		return false, true
	}
	if *usage.Prompt < 0 || *usage.Completion < 0 || *usage.Total < 0 || *usage.Prompt+*usage.Completion != *usage.Total {
		return false, true
	}
	return true, false
}

func (o *modelObserver) writeEventLocked(e observerEvent) {
	if o.events == nil {
		return
	}
	b, err := json.Marshal(e)
	if err == nil {
		_, err = o.events.Write(append(b, '\n'))
	}
	if err == nil {
		err = o.events.Sync()
	}
	if err != nil && o.closeErr == nil {
		o.closeErr = err
	}
}

func (o *modelObserver) Close() (ModelObservation, error) {
	o.mu.Lock()
	if o.closed {
		r, e := o.result, o.closeErr
		o.mu.Unlock()
		return r, e
	}
	o.closed = true
	o.mu.Unlock()
	parentErr := o.parent.Err()
	o.cancel()
	_ = o.server.Close()
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.events != nil {
		if err := o.events.Sync(); err != nil && o.closeErr == nil {
			o.closeErr = err
		}
		if err := o.events.Close(); err != nil && o.closeErr == nil {
			o.closeErr = err
		}
		o.events = nil
	}
	models := make([]string, 0, len(o.models))
	for m := range o.models {
		models = append(models, m)
	}
	sort.Strings(models)
	identity := "unknown"
	if o.identityMismatched > 0 {
		identity = "mismatched"
	} else if o.successful > 0 && o.identityUnknown == 0 && o.identityKnown == o.successful {
		identity = "matched"
	}
	o.result = ModelObservation{Requests: o.requests, UpstreamRequests: o.upstream, SuccessfulRequests: o.successful, PeakHTTPInFlight: o.peak, LimitExceeded: o.exceeded, ExpectedModel: o.expected, ObservedModels: models, ModelIdentityStatus: identity, IdentityKnownRequests: o.identityKnown, IdentityUnknownRequests: o.identityUnknown, IdentityMismatchedRequests: o.identityMismatched, UsageKnownRequests: o.usageKnown, UsageUnknownRequests: o.usageUnknown, UsageInvalidRequests: o.usageInvalid, EventsPath: o.eventsPath}
	if b, err := os.ReadFile(o.eventsPath); err == nil {
		h := sha256.Sum256(b)
		o.result.EventsDigest = hex.EncodeToString(h[:])
	} else if o.closeErr == nil {
		o.closeErr = err
	}
	if o.exceeded && o.closeErr == nil {
		o.closeErr = fmt.Errorf("model observer: request limit %d exceeded", o.max)
	}
	if parentErr != nil && o.closeErr == nil {
		o.closeErr = parentErr
	}
	return o.result, o.closeErr
}
