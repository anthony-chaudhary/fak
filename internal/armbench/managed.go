package armbench

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
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ManagedArm is one explicitly isolated fak context treatment. These values are
// serialized into receipts; do not rename them without a schema revision.
type ManagedArm string

const (
	ManagedDirect        ManagedArm = "direct"
	ManagedPassthrough   ManagedArm = "fak_passthrough"
	ManagedProviderCache ManagedArm = "shared_prefix_provider_cache_only"
	ManagedCompression   ManagedArm = "tool_result_compression_only"
	ManagedShedding      ManagedArm = "context_shedding_only"
	ManagedBundle        ManagedArm = "compression_shedding_bundle"
)

func ManagedArms() []ManagedArm {
	return []ManagedArm{ManagedDirect, ManagedPassthrough, ManagedProviderCache, ManagedCompression, ManagedShedding, ManagedBundle}
}

// ManagedToggles is the auditable switchboard written beside every run.
type ManagedToggles struct {
	FakPath                   bool `json:"fak_path"`
	SharedPrefixProviderCache bool `json:"shared_prefix_provider_cache"`
	ToolResultCompression     bool `json:"tool_result_compression"`
	ContextShedding           bool `json:"context_shedding"`
	Routing                   bool `json:"routing"`
	Policy                    bool `json:"policy"`
	ResponseReuse             bool `json:"response_reuse"`
}

func TogglesForManagedArm(a ManagedArm) (ManagedToggles, error) {
	t := ManagedToggles{FakPath: a != ManagedDirect}
	switch a {
	case ManagedDirect, ManagedPassthrough:
	case ManagedProviderCache:
		t.SharedPrefixProviderCache = true
	case ManagedCompression:
		t.ToolResultCompression = true
	case ManagedShedding:
		t.ContextShedding = true
	case ManagedBundle:
		t.ToolResultCompression, t.ContextShedding = true, true
	default:
		return t, fmt.Errorf("unknown managed arm %q", a)
	}
	return t, nil
}

type ProxyReceipt struct {
	Schema                  string         `json:"schema"`
	Arm                     ManagedArm     `json:"arm"`
	Toggles                 ManagedToggles `json:"toggles"`
	Requests                int64          `json:"requests"`
	InputBytes              int64          `json:"input_bytes"`
	RetainedContextBytes    int64          `json:"retained_context_bytes"`
	OutputBytes             int64          `json:"output_bytes"`
	CacheControlWrites      int64          `json:"cache_control_writes"`
	CompressedToolResults   int64          `json:"compressed_tool_results"`
	ShedMessages            int64          `json:"shed_messages"`
	TransformCPUNanoseconds int64          `json:"fak_cpu_nanoseconds"`
	TTFTMilliseconds        []int64        `json:"ttft_ms"`
	WallMilliseconds        []int64        `json:"wall_ms"`
	RequestSHA256           []string       `json:"request_sha256"`
}

type managedProxy struct {
	mu      sync.Mutex
	receipt ProxyReceipt
}

// StartManagedProxy starts an Anthropic-compatible pass-through on loopback.
// It never records headers or body content: only hashes and aggregate sizes.
func StartManagedProxy(ctx context.Context, arm ManagedArm, upstream string) (baseURL string, stop func() (ProxyReceipt, error), err error) {
	toggles, err := TogglesForManagedArm(arm)
	if err != nil || arm == ManagedDirect {
		return "", nil, fmt.Errorf("managed proxy arm: %w", err)
	}
	target, err := url.Parse(upstream)
	if err != nil || target.Scheme != "https" || target.Host == "" {
		return "", nil, fmt.Errorf("invalid HTTPS upstream")
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", nil, err
	}
	p := &managedProxy{receipt: ProxyReceipt{Schema: "fak-armbench-managed-proxy/1", Arm: arm, Toggles: toggles}}
	client := &http.Client{Timeout: 30 * time.Minute}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		raw, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, "read request", 400)
			return
		}
		cpuStart := time.Now()
		transformed, stats, transformErr := transformAnthropicRequest(raw, toggles)
		transformCPU := time.Since(cpuStart)
		if transformErr != nil {
			http.Error(w, "invalid Anthropic request", 400)
			return
		}
		dest := *target
		dest.Path = strings.TrimRight(target.Path, "/") + r.URL.Path
		dest.RawQuery = r.URL.RawQuery
		req, reqErr := http.NewRequestWithContext(r.Context(), r.Method, dest.String(), bytes.NewReader(transformed))
		if reqErr != nil {
			http.Error(w, "build upstream request", 500)
			return
		}
		req.Header = r.Header.Clone()
		req.Header.Del("content-length")
		req.Host = target.Host
		resp, doErr := client.Do(req)
		if doErr != nil {
			http.Error(w, "upstream unavailable", 502)
			return
		}
		defer resp.Body.Close()
		for k, vv := range resp.Header {
			for _, v := range vv {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)
		first := true
		var out int64
		buf := make([]byte, 32*1024)
		for {
			n, er := resp.Body.Read(buf)
			if n > 0 {
				if first {
					stats.ttft = time.Since(started)
					first = false
				}
				wn, _ := w.Write(buf[:n])
				out += int64(wn)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
			if er != nil {
				break
			}
		}
		sum := sha256.Sum256(raw)
		p.mu.Lock()
		p.receipt.Requests++
		p.receipt.InputBytes += int64(len(raw))
		p.receipt.RetainedContextBytes += int64(len(transformed))
		p.receipt.OutputBytes += out
		p.receipt.CacheControlWrites += stats.cache
		p.receipt.CompressedToolResults += stats.compressed
		p.receipt.ShedMessages += stats.shed
		p.receipt.TransformCPUNanoseconds += transformCPU.Nanoseconds()
		p.receipt.TTFTMilliseconds = append(p.receipt.TTFTMilliseconds, stats.ttft.Milliseconds())
		p.receipt.WallMilliseconds = append(p.receipt.WallMilliseconds, time.Since(started).Milliseconds())
		p.receipt.RequestSHA256 = append(p.receipt.RequestSHA256, hex.EncodeToString(sum[:]))
		p.mu.Unlock()
	})}
	go func() { <-ctx.Done(); _ = srv.Shutdown(context.Background()) }()
	go func() { _ = srv.Serve(ln) }()
	var once sync.Once
	var stopErr error
	stop = func() (ProxyReceipt, error) {
		once.Do(func() { stopErr = srv.Shutdown(context.Background()) })
		p.mu.Lock()
		defer p.mu.Unlock()
		return p.receipt, stopErr
	}
	return "http://" + ln.Addr().String(), stop, nil
}

type transformStats struct {
	cache, compressed, shed int64
	ttft                    time.Duration
}

func transformAnthropicRequest(raw []byte, t ManagedToggles) ([]byte, transformStats, error) {
	started := time.Now()
	if !t.SharedPrefixProviderCache && !t.ToolResultCompression && !t.ContextShedding {
		return append([]byte(nil), raw...), transformStats{ttft: time.Since(started)}, nil
	}
	var root map[string]any
	var st transformStats
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil, st, err
	}
	if t.SharedPrefixProviderCache {
		st.cache += applyCacheControl(root)
	}
	if msgs, ok := root["messages"].([]any); ok {
		if t.ToolResultCompression {
			st.compressed += compressToolResults(msgs)
		}
		if t.ContextShedding {
			var n int64
			msgs, n = shedOldContext(msgs, 48*1024)
			root["messages"] = msgs
			st.shed = n
		}
	}
	out, err := json.Marshal(root)
	st.ttft = time.Since(started)
	return out, st, err
}

func applyCacheControl(root map[string]any) int64 {
	sys, ok := root["system"].([]any)
	if !ok || len(sys) == 0 {
		return 0
	}
	b, ok := sys[len(sys)-1].(map[string]any)
	if !ok {
		return 0
	}
	if _, exists := b["cache_control"]; exists {
		return 0
	}
	b["cache_control"] = map[string]any{"type": "ephemeral"}
	return 1
}

func compressToolResults(msgs []any) int64 {
	var count int64
	for _, m := range msgs {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		blocks, ok := mm["content"].([]any)
		if !ok {
			continue
		}
		for _, b := range blocks {
			bm, ok := b.(map[string]any)
			if !ok || bm["type"] != "tool_result" {
				continue
			}
			s, ok := bm["content"].(string)
			if !ok || len(s) <= 4096 {
				continue
			}
			bm["content"] = s[:2048] + "\n[fak: tool result compressed; " + fmt.Sprint(len(s)-4096) + " middle bytes omitted]\n" + s[len(s)-2048:]
			count++
		}
	}
	return count
}

func shedOldContext(msgs []any, budget int) ([]any, int64) {
	if len(msgs) <= 2 {
		return msgs, 0
	}
	size := func(v any) int { b, _ := json.Marshal(v); return len(b) }
	total := 0
	for _, m := range msgs {
		total += size(m)
	}
	if total <= budget {
		return msgs, 0
	}
	keep := append([]any(nil), msgs...)
	var shed int64
	for len(keep) > 2 && total > budget {
		total -= size(keep[0])
		keep = keep[1:]
		shed++
	}
	marker := map[string]any{"role": "user", "content": "[fak: older context shed to retain the active working set]"}
	return append([]any{marker}, keep...), shed
}

// WriteProxyReceipt atomically writes a secret-free proxy receipt.
func WriteProxyReceipt(path string, r ProxyReceipt) error {
	if path == "" {
		return errors.New("receipt path is required")
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	tmp := path + ".tmp-" + fmt.Sprint(atomic.AddUint64(&receiptSeq, 1))
	if err = os.WriteFile(tmp, b, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

var receiptSeq uint64
