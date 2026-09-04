package modelsrc

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Opener resolves one URL scheme to a random-access object and its byte size.
// The returned ReaderAt may also implement io.Closer.
type Opener func(rawURL string) (io.ReaderAt, int64, error)

// Registry maps model-source URL schemes to transport implementations.
type Registry struct {
	mu     sync.RWMutex
	opener map[string]Opener
}

// Option configures a Registry.
type Option func(*Registry)

// WithHTTPClient makes HTTPS sources use client. It is primarily useful for
// callers that need custom transport, authentication, or TLS roots.
func WithHTTPClient(client *http.Client) Option {
	return func(r *Registry) {
		if client != nil {
			r.opener["https"] = httpOpener(client)
		}
	}
}

// downloadClient bounds the phases that can hang on a dead peer (connect, TLS,
// response headers) but leaves Client.Timeout at 0: a model blob streams for
// minutes, so a whole-request deadline would cut a healthy multi-GB body off
// mid-stream.
const (
	defaultConnectTimeout        = 30 * time.Second
	defaultTLSHandshakeTimeout   = 30 * time.Second
	defaultResponseHeaderTimeout = 60 * time.Second
)

func downloadClient() *http.Client {
	return downloadClientWithTimeouts(defaultConnectTimeout, defaultTLSHandshakeTimeout, defaultResponseHeaderTimeout)
}

func downloadClientWithTimeouts(connectTimeout, tlsTimeout, headerTimeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: 0,
		Transport: &http.Transport{
			DialContext:           (&net.Dialer{Timeout: connectTimeout}).DialContext,
			TLSHandshakeTimeout:   tlsTimeout,
			ResponseHeaderTimeout: headerTimeout,
		},
	}
}

// New returns a registry with local-file and HTTPS transports installed.
func New(opts ...Option) *Registry {
	r := &Registry{opener: make(map[string]Opener)}
	r.opener["file"] = openFile
	r.opener["https"] = httpOpener(downloadClient())
	for _, opt := range opts {
		if opt != nil {
			opt(r)
		}
	}
	return r
}

// Register installs opener for scheme. Scheme matching is case-insensitive.
func (r *Registry) Register(scheme string, opener Opener) {
	scheme = strings.ToLower(strings.TrimSuffix(strings.TrimSpace(scheme), ":"))
	if scheme == "" || opener == nil {
		panic("modelsrc: Register requires a scheme and opener")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.opener[scheme] = opener
}

// Open resolves rawURL through the registered transport for its scheme.
//
// Invariant: model source resolution is fail-closed and bounded. Schemes without
// a registered transport, empty schemes, or unparseable URLs return an error.
func (r *Registry) Open(rawURL string) (io.ReaderAt, int64, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, 0, fmt.Errorf("modelsrc: parse %q: %w", rawURL, err)
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme == "" {
		return nil, 0, fmt.Errorf("modelsrc: %q has no URL scheme; use file:// for local models", rawURL)
	}
	r.mu.RLock()
	opener := r.opener[scheme]
	r.mu.RUnlock()
	if opener == nil {
		return nil, 0, fmt.Errorf("modelsrc: unsupported scheme %q", scheme)
	}
	reader, size, err := opener(rawURL)
	if err != nil {
		return nil, 0, fmt.Errorf("modelsrc: open %s: %w", scheme, err)
	}
	return reader, size, nil
}

var defaultRegistry = New()

// Register installs a transport in the process-wide registry used by Open.
func Register(scheme string, opener Opener) { defaultRegistry.Register(scheme, opener) }

// Open resolves a model-source URL through the process-wide registry.
func Open(rawURL string) (io.ReaderAt, int64, error) { return defaultRegistry.Open(rawURL) }

func openFile(rawURL string) (io.ReaderAt, int64, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, 0, err
	}
	if u.Host != "" && !strings.EqualFold(u.Host, "localhost") {
		return nil, 0, fmt.Errorf("remote file host %q is not supported", u.Host)
	}
	path, err := url.PathUnescape(u.Path)
	if err != nil {
		return nil, 0, fmt.Errorf("decode path: %w", err)
	}
	if len(path) >= 3 && path[0] == '/' && path[2] == ':' {
		path = path[1:]
	}
	path = filepath.FromSlash(path)
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	info, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, 0, err
	}
	if !info.Mode().IsRegular() {
		_ = file.Close()
		return nil, 0, fmt.Errorf("%q is not a regular file", path)
	}
	return file, info.Size(), nil
}

func httpOpener(client *http.Client) Opener {
	return func(rawURL string) (io.ReaderAt, int64, error) {
		resp, err := client.Get(rawURL)
		if err != nil {
			return nil, 0, err
		}
		defer resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode > 299 {
			return nil, 0, fmt.Errorf("GET returned %s", resp.Status)
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, 0, err
		}
		return bytes.NewReader(data), int64(len(data)), nil
	}
}
