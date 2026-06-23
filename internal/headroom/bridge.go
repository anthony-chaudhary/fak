package headroom

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// HeadroomName is the external-service plugin's registry key and FAK_COMPRESSOR
// value. It bridges to a running Headroom proxy — the upstream `headroom proxy`
// process (default http://127.0.0.1:8787) — over HTTP, NEVER a subprocess, so the
// request path stays interpreter- and exec-free (the internal/architest contract).
// Headroom is local-first: data stays on the user's machine. Selecting this plugin
// with no reachable service is inert: Compress returns Compressed==false and the
// original bytes are admitted as-is.
const HeadroomName = "headroom"

// DefaultHeadroomURL is the Headroom proxy's documented default bind.
const DefaultHeadroomURL = "http://127.0.0.1:8787"

type headroomBridge struct {
	url    string
	client *http.Client
}

func newHeadroomBridge() headroomBridge {
	return headroomBridge{url: HeadroomURL(), client: &http.Client{Timeout: headroomTimeout()}}
}

// HeadroomURL is the configured Headroom proxy base URL (FAK_HEADROOM_URL, else
// the documented default), trailing slash trimmed.
func HeadroomURL() string {
	url := strings.TrimRight(strings.TrimSpace(os.Getenv("FAK_HEADROOM_URL")), "/")
	if url == "" {
		return DefaultHeadroomURL
	}
	return url
}

// headroomTimeout bounds each bridge call — sensible default + escape hatch
// (FAK_HEADROOM_TIMEOUT_MS), the repo idiom. The result path must not stall on a
// slow sidecar, so the default is tight.
func headroomTimeout() time.Duration {
	if v := strings.TrimSpace(os.Getenv("FAK_HEADROOM_TIMEOUT_MS")); v != "" {
		if ms, err := strconv.Atoi(v); err == nil && ms > 0 {
			return time.Duration(ms) * time.Millisecond
		}
	}
	return 2 * time.Second
}

func (headroomBridge) Name() string { return HeadroomName }

// --- the /v1/compress wire shapes (verbatim field names from the headroom proxy
// handler; see headroom/proxy/handlers/openai.py). ---

type hrMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type hrConfig struct {
	CompressUserMessages bool `json:"compress_user_messages"`
	ProtectRecent        int  `json:"protect_recent"`
}

type hrRequest struct {
	Messages []hrMessage `json:"messages"`
	Model    string      `json:"model"`
	Config   hrConfig    `json:"config"`
}

type hrResponse struct {
	Messages          []hrMessage `json:"messages"`
	TokensBefore      int         `json:"tokens_before"`
	TokensAfter       int         `json:"tokens_after"`
	TokensSaved       int         `json:"tokens_saved"`
	CompressionRatio  float64     `json:"compression_ratio"`
	TransformsApplied []string    `json:"transforms_applied"`
	CCRHashes         []string    `json:"ccr_hashes"`
}

// defaultModel is the token-accounting model id the proxy uses when the caller
// passes none; the value is irrelevant to the returned bytes, only to the proxy's
// token math.
const defaultModel = "claude-sonnet-4-5-20250929"

func (h headroomBridge) Compress(ctx context.Context, in Input) (Output, error) {
	model := in.Model
	if model == "" {
		model = defaultModel
	}
	body, err := json.Marshal(hrRequest{
		Messages: []hrMessage{{Role: "tool", Content: string(in.Bytes)}},
		Model:    model,
		Config:   hrConfig{CompressUserMessages: false, ProtectRecent: 0},
	})
	if err != nil {
		return passthrough(in), err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.url+"/v1/compress", bytes.NewReader(body))
	if err != nil {
		return passthrough(in), err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(req)
	if err != nil {
		return passthrough(in), nil // unreachable service -> inert (admit original)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return passthrough(in), nil
	}
	var out hrResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Messages) == 0 {
		return passthrough(in), nil
	}
	compressed := []byte(out.Messages[0].Content)
	if len(compressed) == 0 || len(compressed) >= len(in.Bytes) {
		return passthrough(in), nil // no saving -> admit original
	}
	codec := HeadroomName
	if len(out.TransformsApplied) > 0 {
		codec = HeadroomName + ":" + strings.Join(out.TransformsApplied, ",")
	}
	return Output{
		Bytes:      compressed,
		Compressed: true,
		Codec:      codec,
		Retrieval:  strings.Join(out.CCRHashes, ","),
		OrigLen:    len(in.Bytes),
		NewLen:     len(compressed),
	}, nil
}

// Reachable best-effort probes the configured Headroom proxy (a GET on /stats).
// Used by `fak headroom status`; never on the result path.
func Reachable(ctx context.Context) bool {
	b := newHeadroomBridge()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.url+"/stats", nil)
	if err != nil {
		return false
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode < 500
}

func init() { Register(newHeadroomBridge()) }
