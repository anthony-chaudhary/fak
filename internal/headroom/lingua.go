package headroom

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const LinguaName = "lingua"

// LinguaCompressor is the first-class fak adapter for an LLMLingua-2 service.
// It is deliberately never selected by default: the model-backed transform is
// lossy and optional, while the caller's Admit path preserves original bytes in
// CAS before exposing the transformed view.
type LinguaCompressor struct {
	URL    string
	Client *http.Client
}

func (LinguaCompressor) Name() string { return LinguaName }

type linguaRequest struct {
	Text        string  `json:"text"`
	TargetRatio float64 `json:"target_ratio,omitempty"`
}
type linguaResponse struct {
	Text             string `json:"text"`
	Model            string `json:"model,omitempty"`
	OriginalTokens   int    `json:"original_tokens,omitempty"`
	CompressedTokens int    `json:"compressed_tokens,omitempty"`
}

func LinguaURL() string {
	return strings.TrimRight(strings.TrimSpace(os.Getenv("FAK_LINGUA_URL")), "/")
}

func (l LinguaCompressor) Compress(ctx context.Context, in Input) (Output, error) {
	url := strings.TrimRight(strings.TrimSpace(l.URL), "/")
	if url == "" {
		url = LinguaURL()
	}
	if url == "" {
		return Output{}, fmt.Errorf("headroom lingua: FAK_LINGUA_URL is not configured")
	}
	ratio := 0.5
	if raw := strings.TrimSpace(os.Getenv("FAK_LINGUA_TARGET_RATIO")); raw != "" {
		if parsed, parseErr := strconv.ParseFloat(raw, 64); parseErr == nil && parsed > 0 && parsed <= 1 {
			ratio = parsed
		}
	}
	body, err := json.Marshal(linguaRequest{Text: string(in.Bytes), TargetRatio: ratio})
	if err != nil {
		return Output{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/v1/compress", bytes.NewReader(body))
	if err != nil {
		return Output{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := l.Client
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	resp, err := client.Do(req)
	if err != nil {
		return Output{}, fmt.Errorf("headroom lingua: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return Output{}, fmt.Errorf("headroom lingua: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var decoded linguaResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&decoded); err != nil {
		return Output{}, fmt.Errorf("headroom lingua: decode: %w", err)
	}
	if decoded.Text == "" && len(in.Bytes) > 0 {
		return Output{}, fmt.Errorf("headroom lingua: empty compressed text")
	}
	compressed := decoded.Text != string(in.Bytes) && len(decoded.Text) < len(in.Bytes)
	status, reason := "no_effect", "LLMLingua returned no smaller rendering"
	if compressed {
		status, reason = "saved", fmt.Sprintf("model=%s tokens=%d->%d", decoded.Model, decoded.OriginalTokens, decoded.CompressedTokens)
	}
	return Output{Bytes: []byte(decoded.Text), Compressed: compressed, Codec: "llmlingua-2", OrigLen: len(in.Bytes), NewLen: len(decoded.Text), Status: status, Reason: reason}, nil
}

func init() { Register(LinguaCompressor{}) }
