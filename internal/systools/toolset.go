package systools

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Engine ID constants.
const (
	EngineGetTime   = "systools.get_time"
	EngineFetchWeb  = "systools.fetch_web"
	EngineWebSearch = "systools.web_search"
)

// RungName identifies this package's adjudicator rung.
const RungName = "systools"

const (
	DefaultFetchByteCap    = 32 * 1024   // 32KB default
	DefaultMaxFetchByteCap = 1024 * 1024 // 1MB ceiling
	DefaultFetchTimeout    = 15 * time.Second
)

// SearchResult represents one structured entry returned by web_search.
type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

// SearchAdapter is the search backend function signature.
type SearchAdapter func(ctx context.Context, query string, maxResults int) ([]SearchResult, error)

// Policy defines tool admission policy.
type Policy struct {
	Allow map[string]bool
}

// DefaultPolicy admits all three systools.
func DefaultPolicy() Policy {
	return Policy{Allow: map[string]bool{
		ToolGetTime:   true,
		ToolFetchWeb:  true,
		ToolWebSearch: true,
	}}
}

// Config configures a systools Toolset.
type Config struct {
	AllowPrivateIPs      bool          // if true, SSRF protection permits 127.0.0.1 and private IPs
	AllowedDomains       []string      // optional domain allowlist; empty permits all public domains
	DefaultMaxFetchBytes int           // default response byte cap (default 32KB)
	MaxFetchBytes        int           // ceiling response byte cap (default 1MB)
	FetchTimeout         time.Duration // timeout for HTTP requests (default 15s)
	SearchAdapter        SearchAdapter // custom search adapter; nil uses simulated documentation search
	Policy               Policy        // tool allowlist
	HTTPClient           *http.Client  // optional custom HTTP client
}

// Toolset represents a configured instance of systools.
type Toolset struct {
	allowPrivateIPs      bool
	allowedDomains       []string
	defaultMaxFetchBytes int
	maxFetchBytes        int
	fetchTimeout         time.Duration
	searchAdapter        SearchAdapter
	policy               Policy
	httpClient           *http.Client
}

// New creates a new Toolset configured with cfg.
func New(cfg Config) (*Toolset, error) {
	defaultBytes := cfg.DefaultMaxFetchBytes
	if defaultBytes <= 0 {
		defaultBytes = DefaultFetchByteCap
	}
	maxBytes := cfg.MaxFetchBytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxFetchByteCap
	}
	timeout := cfg.FetchTimeout
	if timeout <= 0 {
		timeout = DefaultFetchTimeout
	}
	search := cfg.SearchAdapter
	if search == nil {
		search = defaultDocSearchAdapter
	}
	pol := cfg.Policy
	if pol.Allow == nil {
		pol = DefaultPolicy()
	}

	client := cfg.HTTPClient
	if client == nil {
		allowPrivate := cfg.AllowPrivateIPs
		transport := &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				host, port, err := net.SplitHostPort(addr)
				if err != nil {
					return nil, err
				}
				if !allowPrivate {
					ips, err := net.DefaultResolver.LookupIP(ctx, "ip", host)
					if err != nil {
						return nil, fmt.Errorf("SSRF protection: DNS lookup failed for %s: %w", host, err)
					}
					for _, ip := range ips {
						if isPrivateIP(ip) {
							return nil, fmt.Errorf("SSRF protection: access to private IP %s is blocked", ip.String())
						}
					}
				}
				dialer := &net.Dialer{Timeout: 10 * time.Second}
				return dialer.DialContext(ctx, network, net.JoinHostPort(host, port))
			},
			ResponseHeaderTimeout: 10 * time.Second,
		}
		client = &http.Client{
			Transport: transport,
			Timeout:   timeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= 5 {
					return fmt.Errorf("too many redirects")
				}
				if !allowPrivate {
					host := req.URL.Hostname()
					if isHostPrivate(req.Context(), host) {
						return fmt.Errorf("SSRF protection: redirect to private host %s is blocked", host)
					}
				}
				return nil
			},
		}
	}

	return &Toolset{
		allowPrivateIPs:      cfg.AllowPrivateIPs,
		allowedDomains:       cfg.AllowedDomains,
		defaultMaxFetchBytes: defaultBytes,
		maxFetchBytes:        maxBytes,
		fetchTimeout:         timeout,
		searchAdapter:        search,
		policy:               pol,
		httpClient:           client,
	}, nil
}

// RegisterEngines binds the engines into the abi registry under their engine IDs.
func (t *Toolset) RegisterEngines() {
	abi.RegisterEngine(EngineGetTime, getTimeEngine{t})
	abi.RegisterEngine(EngineFetchWeb, fetchWebEngine{t})
	abi.RegisterEngine(EngineWebSearch, webSearchEngine{t})
}

// Register builds a Toolset, registers its engines, and places its rung in the
// adjudication chain at rank.
func Register(cfg Config, rank int) (*Toolset, error) {
	t, err := New(cfg)
	if err != nil {
		return nil, err
	}
	t.RegisterEngines()
	abi.RegisterAdjudicator(rank, t)
	return t, nil
}

func engineFor(tool string) (string, bool) {
	switch tool {
	case ToolGetTime:
		return EngineGetTime, true
	case ToolFetchWeb:
		return EngineFetchWeb, true
	case ToolWebSearch:
		return EngineWebSearch, true
	}
	return "", false
}

// CallMeta returns the metadata map for a systool call.
func CallMeta(tool, principal string) map[string]string {
	m := map[string]string{
		"readOnlyHint": "true",
	}
	if tool != ToolGetTime {
		m["idempotentHint"] = "true"
	}
	if principal != "" {
		m["principal"] = principal
	}
	return m
}

func (t *Toolset) domainAllowed(host string) bool {
	if len(t.allowedDomains) == 0 {
		return true
	}
	host = strings.ToLower(host)
	for _, pattern := range t.allowedDomains {
		pattern = strings.ToLower(pattern)
		if pattern == host {
			return true
		}
		if strings.HasPrefix(pattern, "*.") {
			suffix := pattern[2:]
			if host == suffix || strings.HasSuffix(host, "."+suffix) {
				return true
			}
		}
	}
	return false
}

func (t *Toolset) checkSSRF(ctx context.Context, host string) *Refusal {
	if strings.EqualFold(host, "localhost") {
		return refuse(CodeSSRFBlock, "fetch_web: SSRF protection blocked access to localhost")
	}
	if ip := net.ParseIP(host); ip != nil {
		if isPrivateIP(ip) {
			return refuse(CodeSSRFBlock, "fetch_web: SSRF protection blocked access to private IP "+host)
		}
		return nil
	}
	if ips, err := net.DefaultResolver.LookupIP(ctx, "ip", host); err == nil {
		for _, ip := range ips {
			if isPrivateIP(ip) {
				return refuse(CodeSSRFBlock, "fetch_web: SSRF protection blocked access to private IP "+ip.String()+" for "+host)
			}
		}
	}
	return nil
}

func isPrivateIP(ip net.IP) bool {
	if ip == nil {
		return true
	}
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsMulticast() || ip.IsUnspecified() {
		return true
	}
	ip4 := ip.To4()
	if ip4 != nil {
		if ip4[0] == 0 {
			return true
		}
		if ip4[0] == 100 && (ip4[1]&0xc0) == 64 {
			return true
		}
		if ip4[0] == 169 && ip4[1] == 254 {
			return true
		}
		if ip4[0] == 192 && ip4[1] == 0 && (ip4[2] == 0 || ip4[2] == 2) {
			return true
		}
		if ip4[0] == 198 && (ip4[1]&0xfe) == 18 {
			return true
		}
		if ip4[0] == 198 && ip4[1] == 51 && ip4[2] == 100 {
			return true
		}
		if ip4[0] == 203 && ip4[1] == 0 && ip4[2] == 113 {
			return true
		}
		if ip4[0] >= 240 {
			return true
		}
	}
	return false
}

func isHostPrivate(ctx context.Context, host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	if ip := net.ParseIP(host); ip != nil {
		return isPrivateIP(ip)
	}
	ips, err := net.DefaultResolver.LookupIP(ctx, "ip", host)
	if err != nil {
		return true
	}
	for _, ip := range ips {
		if isPrivateIP(ip) {
			return true
		}
	}
	return false
}

func bytesOf(ctx context.Context, ref abi.Ref) []byte {
	if ref.Kind == abi.RefInline {
		return ref.Inline
	}
	if r := abi.ActiveResolver(); r != nil {
		if body, err := r.Resolve(ctx, ref); err == nil {
			return body
		}
	}
	return nil
}

func putBytes(ctx context.Context, b []byte) abi.Ref {
	if r := abi.ActiveResolver(); r != nil {
		if ref, err := r.Put(ctx, b); err == nil {
			return ref
		}
	}
	return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}
}

func result(ctx context.Context, c *abi.ToolCall, in, out []byte, isErr bool, engineID string) *abi.Result {
	status := abi.StatusOK
	if isErr {
		status = abi.StatusError
	}
	return &abi.Result{Call: c, Payload: putBytes(ctx, out), Status: status, Meta: map[string]string{
		"engine":        engineID,
		"input_tokens":  strconv.Itoa(len(in) / 4),
		"output_tokens": strconv.Itoa(len(out) / 4),
	}}
}

func okJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return refuse(CodeIO, "cannot encode result").JSON()
	}
	return b
}

var defaultDocs = []SearchResult{
	{
		Title:   "FAK Agent Kernel Overview",
		URL:     "https://github.com/anthony-chaudhary/fak#readme",
		Snippet: "fak is an agent kernel that sits between an AI agent and the tools it calls, providing default-deny security and vDSO performance acceleration.",
	},
	{
		Title:   "FAK System Tools and Web Utilities",
		URL:     "https://github.com/anthony-chaudhary/fak/docs/systools.md",
		Snippet: "Utility tools for native agents: get_time for system timestamps, fetch_web with SSRF protection and byte capping, and web_search for documentation querying.",
	},
	{
		Title:   "FAK Tool Sandboxing and Security Policy",
		URL:     "https://github.com/anthony-chaudhary/fak/POLICY.md",
		Snippet: "Kernel-mediated tool security: strict schema decoding, SSRF prevention, domain allowlists, and execution confinement.",
	},
	{
		Title:   "FAK Native Inference and Architecture",
		URL:     "https://github.com/anthony-chaudhary/fak/ARCHITECTURE.md",
		Snippet: "Layered DAG architecture and native inference runtime beating traditional engines in matched, quality-constrained envelopes.",
	},
	{
		Title:   "Go Standard Library Documentation",
		URL:     "https://pkg.go.dev/std",
		Snippet: "Official documentation for the Go standard library packages including net/http, context, time, and encoding/json.",
	},
}

func defaultDocSearchAdapter(_ context.Context, query string, maxResults int) ([]SearchResult, error) {
	qLower := strings.ToLower(query)
	words := strings.Fields(qLower)
	var matched []SearchResult
	for _, doc := range defaultDocs {
		text := strings.ToLower(doc.Title + " " + doc.Snippet)
		match := false
		for _, w := range words {
			if strings.Contains(text, w) {
				match = true
				break
			}
		}
		if match {
			matched = append(matched, doc)
			if len(matched) >= maxResults {
				break
			}
		}
	}
	if len(matched) == 0 {
		for _, doc := range defaultDocs {
			matched = append(matched, doc)
			if len(matched) >= maxResults {
				break
			}
		}
	}
	return matched, nil
}
