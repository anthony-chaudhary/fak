package modelperfobs

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const proxyResponseHeaderTimeout = 30 * time.Second

var fallbackProxyClient = func() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = proxyResponseHeaderTimeout
	return &http.Client{Transport: transport}
}()

type Proxy struct {
	Backend  *url.URL
	Ledger   string
	Client   *http.Client
	Now      func() time.Time
	mu       sync.Mutex
	active   map[string]struct{}
	overlaps map[string]map[string]struct{}
	seq      atomic.Uint64
}

type requestEnvelope struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
}

type responseUsage struct {
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	started := p.now()
	id := fmt.Sprintf("%d-%d", started.UnixNano(), p.seq.Add(1))
	obs := Observation{Schema: Schema, Timestamp: started.UTC(), RequestID: id, Backend: p.Backend.String()}
	p.beginRequest(id)
	completed := false
	defer func() {
		if !completed {
			p.completeObservation(&obs, started)
		}
	}()

	body, err := io.ReadAll(r.Body)
	if err != nil {
		message := fmt.Sprintf("read inbound request body: %v; retry with a readable request body", err)
		obs.Error = message
		http.Error(w, message, http.StatusBadRequest)
		return
	}
	var envelope requestEnvelope
	_ = json.Unmarshal(body, &envelope)
	obs.Model, obs.Streaming = envelope.Model, envelope.Stream

	target := p.Backend.ResolveReference(r.URL)
	req, err := http.NewRequestWithContext(r.Context(), r.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		message := fmt.Sprintf("construct backend request: %v; retry with a valid HTTP method and request target", err)
		obs.Error = message
		http.Error(w, message, http.StatusBadGateway)
		return
	}
	req.Header = r.Header.Clone()
	req.Header.Set("X-Fak-Observation-ID", id)
	client := p.Client
	if client == nil {
		client = fallbackProxyClient
	}
	resp, err := client.Do(req)
	if err != nil {
		message := fmt.Sprintf("reach backend: %v; verify the backend URL and availability, then retry", err)
		obs.Error = message
		p.completeObservation(&obs, started)
		completed = true
		http.Error(w, message, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	copyHeaders(w.Header(), resp.Header)
	w.Header().Set("X-Fak-Observation-ID", id)
	w.WriteHeader(resp.StatusCode)
	obs.Status = resp.StatusCode

	if envelope.Stream || strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		p.copyStream(w, resp.Body, started, &obs)
	} else {
		data, readErr := io.ReadAll(resp.Body)
		_, _ = w.Write(data)
		var usage responseUsage
		if json.Unmarshal(data, &usage) == nil {
			obs.PromptTokens = usage.Usage.PromptTokens
			obs.CompletionTokens = usage.Usage.CompletionTokens
			obs.TotalTokens = usage.Usage.TotalTokens
		}
		if readErr != nil {
			obs.Error = readErr.Error()
		}
	}
	p.completeObservation(&obs, started)
	completed = true
}

func (p *Proxy) copyStream(w http.ResponseWriter, src io.Reader, started time.Time, obs *Observation) {
	flusher, _ := w.(http.Flusher)
	scanner := bufio.NewScanner(src)
	// Model reasoning/tool chunks can exceed Scanner's small default token.
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	var arrivals []float64
	var previous time.Time
	for scanner.Scan() {
		line := append([]byte(nil), scanner.Bytes()...)
		_, _ = w.Write(append(line, '\n'))
		if flusher != nil {
			flusher.Flush()
		}
		if !bytes.HasPrefix(line, []byte("data:")) || bytes.Equal(bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:"))), []byte("[DONE]")) {
			continue
		}
		payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if isOutputChunk(payload) {
			now := p.now()
			if obs.TTFTMS == 0 {
				obs.TTFTMS = millis(now.Sub(started))
			}
			if !previous.IsZero() {
				arrivals = append(arrivals, millis(now.Sub(previous)))
			}
			previous = now
		}
		var usage responseUsage
		if json.Unmarshal(payload, &usage) == nil && usage.Usage.TotalTokens > 0 {
			obs.PromptTokens = usage.Usage.PromptTokens
			obs.CompletionTokens = usage.Usage.CompletionTokens
			obs.TotalTokens = usage.Usage.TotalTokens
		}
	}
	if err := scanner.Err(); err != nil {
		obs.Error = err.Error()
	}
	obs.InterChunkCount = int64(len(arrivals))
	obs.InterChunkP50MS = percentile(arrivals, .50)
	obs.InterChunkP95MS = percentile(arrivals, .95)
}

func isOutputChunk(payload []byte) bool {
	var chunk struct {
		Choices []struct {
			Delta map[string]json.RawMessage `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(payload, &chunk) != nil {
		return false
	}
	for _, choice := range chunk.Choices {
		for key, value := range choice.Delta {
			if key == "role" || key == "finish_reason" {
				continue
			}
			trimmed := bytes.TrimSpace(value)
			if len(trimmed) > 0 && string(trimmed) != `""` && string(trimmed) != "null" {
				return true
			}
		}
	}
	return false
}
func derive(obs *Observation) {
	if obs.CompletionTokens <= 0 {
		return
	}
	decodeMS := obs.DurationMS - obs.TTFTMS
	if decodeMS <= 0 {
		decodeMS = obs.DurationMS
	}
	obs.OutputTokensPerSec = float64(obs.CompletionTokens) / (decodeMS / 1000)
	if obs.CompletionTokens > 1 {
		obs.TPOTMS = decodeMS / float64(obs.CompletionTokens-1)
	}
}

func (p *Proxy) beginRequest(id string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.active == nil {
		p.active = make(map[string]struct{})
	}
	if p.overlaps == nil {
		p.overlaps = make(map[string]map[string]struct{})
	}
	p.overlaps[id] = make(map[string]struct{}, len(p.active))
	for activeID := range p.active {
		p.overlaps[id][activeID] = struct{}{}
		p.overlaps[activeID][id] = struct{}{}
	}
	p.active[id] = struct{}{}
}

func (p *Proxy) finishRequest(id string) []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	peers := p.overlaps[id]
	ids := make([]string, 0, len(peers))
	for peer := range peers {
		ids = append(ids, peer)
	}
	sort.Strings(ids)
	delete(p.active, id)
	delete(p.overlaps, id)
	return ids
}

func (p *Proxy) completeObservation(obs *Observation, started time.Time) {
	obs.CompletedAt = p.now().UTC()
	obs.DurationMS = millis(obs.CompletedAt.Sub(started))
	obs.OverlappingRequestIDs = p.finishRequest(obs.RequestID)
	obs.OverlappingObservedRequests = len(obs.OverlappingRequestIDs)
	derive(obs)
	_ = p.append(*obs)
}

func (p *Proxy) append(obs Observation) error {
	if p.Ledger == "" {
		return nil
	}
	data, err := json.Marshal(obs)
	if err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(p.Ledger), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(p.Ledger, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(data, '\n'))
	return err
}

func (p *Proxy) now() time.Time {
	if p.Now != nil {
		return p.Now()
	}
	return time.Now()
}
func millis(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
func copyHeaders(dst, src http.Header) {
	for k, vv := range src {
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}
func percentile(values []float64, q float64) float64 {
	if len(values) == 0 {
		return 0
	}
	v := append([]float64(nil), values...)
	sort.Float64s(v)
	i := int(float64(len(v)-1)*q + .5)
	return v[i]
}

func ParseBackend(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, errors.New("backend must be an absolute http(s) URL")
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, errors.New("backend must use http or https")
	}
	return u, nil
}
