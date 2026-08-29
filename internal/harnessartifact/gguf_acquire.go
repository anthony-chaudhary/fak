package harnessartifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const GGUFCacheReceiptSchema = "fak.harness.gguf-cache-receipt.v1"

const defaultGGUFDownloadTimeout = 2 * time.Hour

var (
	ErrGGUFInvalidPlan    = errors.New("invalid GGUF acquisition plan")
	ErrGGUFOfflineMiss    = errors.New("GGUF unavailable in offline cache")
	ErrGGUFSizeMismatch   = errors.New("GGUF byte count mismatch")
	ErrGGUFDigestMismatch = errors.New("GGUF SHA-256 mismatch")
)

var ggufDigestPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

type GGUFRequest struct {
	Source   string `json:"source"`
	License  string `json:"license"`
	Bytes    int64  `json:"bytes"`
	SHA256   string `json:"sha256"`
	CacheDir string `json:"cache_dir"`
}

type GGUFPlan struct {
	Source      string `json:"source"`
	License     string `json:"license"`
	Bytes       int64  `json:"bytes"`
	SHA256      string `json:"sha256"`
	Destination string `json:"destination"`
}

type GGUFCacheReceipt struct {
	Schema   string `json:"schema"`
	Source   string `json:"source"`
	License  string `json:"license"`
	Bytes    int64  `json:"bytes"`
	SHA256   string `json:"sha256"`
	Path     string `json:"path"`
	CacheHit bool   `json:"cache_hit"`
}

type GGUFApplyOptions struct {
	Offline bool
	Client  *http.Client
}

// PlanGGUFAcquisition validates and describes acquisition without filesystem or network effects.
func PlanGGUFAcquisition(req GGUFRequest) (GGUFPlan, error) {
	req.Source = strings.TrimSpace(req.Source)
	req.License = strings.TrimSpace(req.License)
	req.SHA256 = strings.ToLower(strings.TrimSpace(req.SHA256))
	req.CacheDir = strings.TrimSpace(req.CacheDir)
	if req.Source == "" || req.License == "" || req.Bytes <= 0 || !ggufDigestPattern.MatchString(req.SHA256) || req.CacheDir == "" {
		return GGUFPlan{}, ErrGGUFInvalidPlan
	}
	if !filepath.IsAbs(req.CacheDir) {
		return GGUFPlan{}, fmt.Errorf("%w: cache_dir must be absolute", ErrGGUFInvalidPlan)
	}
	return GGUFPlan{Source: req.Source, License: req.License, Bytes: req.Bytes, SHA256: req.SHA256, Destination: filepath.Join(filepath.Clean(req.CacheDir), req.SHA256+".gguf")}, nil
}

// ApplyGGUFAcquisition verifies an existing cache blob or downloads, verifies, and atomically promotes one.
func ApplyGGUFAcquisition(ctx context.Context, plan GGUFPlan, opts GGUFApplyOptions) (GGUFCacheReceipt, error) {
	if _, err := PlanGGUFAcquisition(GGUFRequest{Source: plan.Source, License: plan.License, Bytes: plan.Bytes, SHA256: plan.SHA256, CacheDir: filepath.Dir(plan.Destination)}); err != nil || plan.Destination != filepath.Join(filepath.Dir(plan.Destination), plan.SHA256+".gguf") {
		return GGUFCacheReceipt{}, ErrGGUFInvalidPlan
	}
	if err := verifyGGUF(plan.Destination, plan.Bytes, plan.SHA256); err == nil {
		return ggufReceipt(plan, true), nil
	}
	if opts.Offline {
		return GGUFCacheReceipt{}, ErrGGUFOfflineMiss
	}
	if err := os.MkdirAll(filepath.Dir(plan.Destination), 0o755); err != nil {
		return GGUFCacheReceipt{}, err
	}
	tmp, err := os.CreateTemp(filepath.Dir(plan.Destination), ".gguf-download-*")
	if err != nil {
		return GGUFCacheReceipt{}, err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	client := opts.Client
	if client == nil {
		client = defaultGGUFHTTPClient()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, plan.Source, nil)
	if err != nil {
		tmp.Close()
		return GGUFCacheReceipt{}, err
	}
	resp, err := client.Do(req)
	if err != nil {
		tmp.Close()
		return GGUFCacheReceipt{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		tmp.Close()
		return GGUFCacheReceipt{}, fmt.Errorf("GGUF source status %s", resp.Status)
	}
	h := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(tmp, h), io.LimitReader(resp.Body, plan.Bytes+1))
	closeErr := tmp.Close()
	if copyErr != nil {
		return GGUFCacheReceipt{}, copyErr
	}
	if closeErr != nil {
		return GGUFCacheReceipt{}, closeErr
	}
	if n != plan.Bytes {
		return GGUFCacheReceipt{}, fmt.Errorf("%w: got %d want %d", ErrGGUFSizeMismatch, n, plan.Bytes)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != plan.SHA256 {
		return GGUFCacheReceipt{}, fmt.Errorf("%w: got %s", ErrGGUFDigestMismatch, got)
	}
	if err := os.Rename(tmpName, plan.Destination); err != nil {
		return GGUFCacheReceipt{}, err
	}
	return ggufReceipt(plan, false), nil
}

func defaultGGUFHTTPClient() *http.Client {
	// GGUF downloads can legitimately run much longer than an API request, but
	// the default must still terminate a stalled transfer when callers supply
	// neither their own client nor a tighter context deadline.
	return &http.Client{Timeout: defaultGGUFDownloadTimeout}
}

func verifyGGUF(path string, bytes int64, digest string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return err
	}
	if n != bytes {
		return ErrGGUFSizeMismatch
	}
	if hex.EncodeToString(h.Sum(nil)) != digest {
		return ErrGGUFDigestMismatch
	}
	return nil
}

func ggufReceipt(plan GGUFPlan, hit bool) GGUFCacheReceipt {
	return GGUFCacheReceipt{Schema: GGUFCacheReceiptSchema, Source: plan.Source, License: plan.License, Bytes: plan.Bytes, SHA256: plan.SHA256, Path: plan.Destination, CacheHit: hit}
}

func (r GGUFCacheReceipt) CanonicalJSON() ([]byte, error) { return json.Marshal(r) }
