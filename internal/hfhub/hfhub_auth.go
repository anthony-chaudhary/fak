package hfhub

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

const (
	// maxAuthRedirects bounds manual redirect hops in doAuthRequest.
	maxAuthRedirects = 10
	// minCredentialPrefix is the minimum decoded prefix length that fails
	// credential containment when matched at any percent-encoding depth.
	minCredentialPrefix = 4
)

var (
	// ErrCredentialRedirect reports that a request URL or redirect Location
	// reflects the Hub credential (raw or percent-encoded).
	ErrCredentialRedirect = errors.New("hfhub: credential reflected in redirect location")
	// ErrTooManyRedirects reports that a request exceeded maxAuthRedirects hops.
	ErrTooManyRedirects = errors.New("hfhub: stopped after too many redirects")
	// ErrRedirectLocationMissing reports a 3xx response without a Location header.
	ErrRedirectLocationMissing = errors.New("hfhub: redirect response missing Location header")
	// ErrHubDenied reports that the Hub origin returned 401 or 403.
	ErrHubDenied = errors.New("hfhub: access denied by hub origin")
)

func extractSecret(token string) string {
	token = strings.TrimSpace(token)
	token = strings.TrimPrefix(token, "Bearer ")
	return token
}

// ownsURL reports whether u shares scheme, host, and effective port with c.base().
// Porting TensorFold pull_auth.zig Auth.owns (Apache-2.0).
func (c *Client) ownsURL(u *url.URL) bool {
	if u == nil {
		return false
	}
	baseURL, err := url.Parse(c.base())
	if err != nil {
		return false
	}
	return sameOrigin(baseURL, u)
}

func sameOrigin(a, b *url.URL) bool {
	if a == nil || b == nil {
		return false
	}
	ah := a.Hostname()
	bh := b.Hostname()
	if ah == "" || bh == "" {
		return false
	}
	ap := effectivePort(a)
	bp := effectivePort(b)
	if ap == "" || bp == "" {
		return false
	}
	return strings.EqualFold(a.Scheme, b.Scheme) && strings.EqualFold(ah, bh) && ap == bp
}

func effectivePort(u *url.URL) string {
	if p := u.Port(); p != "" {
		return p
	}
	switch strings.ToLower(u.Scheme) {
	case "https":
		return "443"
	case "http":
		return "80"
	default:
		return ""
	}
}

// authorize attaches the Bearer token only when req targets the Hub's own origin,
// and strips any existing Authorization header otherwise.
func (c *Client) authorize(req *http.Request) {
	if req == nil {
		return
	}
	secret := extractSecret(c.Token)
	if secret != "" && c.ownsURL(req.URL) {
		req.Header.Set("Authorization", "Bearer "+secret)
		return
	}
	req.Header.Del("Authorization")
}

// reflectsCredential reports whether text contains the client's secret or a
// decoded prefix of at least minCredentialPrefix bytes at any percent-encoding depth.
func (c *Client) reflectsCredential(text string) bool {
	secret := extractSecret(c.Token)
	if secret == "" || text == "" {
		return false
	}
	return containsCredential(text, secret)
}

// doAuthRequest executes req with manual redirect handling so each hop
// re-evaluates same-origin Authorization scoping, checks URL and Location
// headers for credential reflection, and fails fast on 401/403 from the Hub origin.
func (c *Client) doAuthRequest(ctx context.Context, req *http.Request) (*http.Response, error) {
	baseClient := c.httpClient()
	noRedirect := *baseClient
	noRedirect.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}

	cur := req.Clone(ctx)
	for hop := 0; ; hop++ {
		if c.reflectsCredential(cur.URL.String()) {
			return nil, ErrCredentialRedirect
		}
		c.authorize(cur)
		resp, err := noRedirect.Do(cur)
		if err != nil {
			return nil, err
		}
		if (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden) && c.ownsURL(cur.URL) {
			resp.Body.Close()
			return nil, fmt.Errorf("%w: %s", ErrHubDenied, resp.Status)
		}
		if !isRedirectStatus(resp.StatusCode) {
			return resp, nil
		}
		loc := resp.Header.Get("Location")
		resp.Body.Close()
		if loc == "" {
			return nil, ErrRedirectLocationMissing
		}
		if c.reflectsCredential(loc) {
			return nil, ErrCredentialRedirect
		}
		if hop >= maxAuthRedirects {
			return nil, ErrTooManyRedirects
		}
		nextURL, err := cur.URL.Parse(loc)
		if err != nil {
			return nil, fmt.Errorf("hfhub: invalid redirect location: %w", err)
		}
		nextMethod := cur.Method
		if resp.StatusCode == http.StatusMovedPermanently || resp.StatusCode == http.StatusFound || resp.StatusCode == http.StatusSeeOther {
			if nextMethod != http.MethodHead {
				nextMethod = http.MethodGet
			}
		}
		nextReq, err := http.NewRequestWithContext(ctx, nextMethod, nextURL.String(), nil)
		if err != nil {
			return nil, err
		}
		for k, vals := range cur.Header {
			if strings.EqualFold(k, "Authorization") {
				continue
			}
			for _, v := range vals {
				nextReq.Header.Add(k, v)
			}
		}
		cur = nextReq
	}
}

func isRedirectStatus(code int) bool {
	switch code {
	case http.StatusMovedPermanently, http.StatusFound, http.StatusSeeOther,
		http.StatusTemporaryRedirect, http.StatusPermanentRedirect:
		return true
	default:
		return false
	}
}

// credentialMatch mirrors TensorFold platform/credential.zig Match.
type credentialMatch struct {
	prefixLen int
	consumed  int
	hold      bool
}

type decodeKey struct {
	at  int
	byt byte
}

type decodeResult struct {
	ends    []int
	partial bool
}

// credentialMatcher matches arbitrarily nested percent-encoded representations
// of a secret without call-stack recursion.
type credentialMatcher struct {
	text []byte
	memo map[decodeKey]decodeResult
}

func newCredentialMatcher(text []byte) *credentialMatcher {
	return &credentialMatcher{text: text}
}

func (m *credentialMatcher) matchSecret(start int, secret string) credentialMatch {
	if start > len(m.text) || len(secret) == 0 {
		return credentialMatch{}
	}
	states := []int{start}
	var next []int
	hold := false
	for i := 0; i < len(secret); i++ {
		wanted := secret[i]
		next = next[:0]
		partialByte := false
		for _, at := range states {
			if at == len(m.text) {
				hold = true
				continue
			}
			if m.text[at] != '%' {
				if m.text[at] == wanted {
					next = appendUniqueInt(next, at+1)
				}
				continue
			}
			dec := m.decode(at, wanted)
			if dec.partial {
				partialByte = true
				hold = true
			}
			for _, end := range dec.ends {
				next = appendUniqueInt(next, end)
			}
		}
		if len(next) == 0 {
			pLen := i
			if partialByte {
				pLen++
			}
			return credentialMatch{prefixLen: pLen, hold: hold}
		}
		states, next = next, states
	}
	consumed := 0
	for _, end := range states {
		if end-start > consumed {
			consumed = end - start
		}
	}
	return credentialMatch{prefixLen: len(secret), consumed: consumed, hold: hold}
}

type decodeStateKind uint8

const (
	decodeStatePercent decodeStateKind = iota
	decodeStateEncoded
	decodeStateFinished
)

type decodeFrame struct {
	key        decodeKey
	ends       []int
	partial    bool
	kind       decodeStateKind
	scan       int
	twoIdx     int
	pair       int
	percentIdx int
	highIdx    int
}

func newDecodeFrame(text []byte, key decodeKey) decodeFrame {
	f := decodeFrame{
		key:     key,
		partial: key.at == len(text),
		kind:    decodeStateFinished,
	}
	if key.at < len(text) && text[key.at] == key.byt {
		f.ends = append(f.ends, key.at+1)
	}
	if key.byt == '%' {
		f.kind = decodeStatePercent
	} else if key.at < len(text) && text[key.at] == '%' {
		f.kind = decodeStateEncoded
	}
	return f
}

func (m *credentialMatcher) decode(at int, byt byte) decodeResult {
	if m.memo == nil {
		m.memo = make(map[decodeKey]decodeResult)
	}
	wanted := decodeKey{at: at, byt: byt}
	if cached, ok := m.memo[wanted]; ok {
		return cached
	}
	const hexUpper = "0123456789ABCDEF"
	const hexLower = "0123456789abcdef"

	work := []decodeFrame{newDecodeFrame(m.text, wanted)}
	for len(work) > 0 {
		top := &work[len(work)-1]
		switch top.kind {
		case decodeStatePercent:
			paused := false
			for top.scan < len(top.ends) {
				twoKey := decodeKey{at: top.ends[top.scan], byt: '2'}
				two, ok := m.memo[twoKey]
				if !ok {
					work = append(work, newDecodeFrame(m.text, twoKey))
					paused = true
					break
				}
				top.partial = top.partial || two.partial
				if top.twoIdx == len(two.ends) {
					top.scan++
					top.twoIdx = 0
					continue
				}
				fiveKey := decodeKey{at: two.ends[top.twoIdx], byt: '5'}
				five, ok := m.memo[fiveKey]
				if !ok {
					work = append(work, newDecodeFrame(m.text, fiveKey))
					paused = true
					break
				}
				top.partial = top.partial || five.partial
				for _, end := range five.ends {
					top.ends = appendUniqueInt(top.ends, end)
				}
				top.twoIdx++
			}
			if !paused {
				top.kind = decodeStateFinished
			}
		case decodeStateEncoded:
			percentKey := decodeKey{at: top.key.at, byt: '%'}
			percent, ok := m.memo[percentKey]
			if !ok {
				work = append(work, newDecodeFrame(m.text, percentKey))
				break
			}
			top.partial = top.partial || percent.partial
			hiUp := hexUpper[top.key.byt>>4]
			loUp := hexUpper[top.key.byt&15]
			hiLo := hexLower[top.key.byt>>4]
			loLo := hexLower[top.key.byt&15]
			paused := false
			for top.pair < 4 {
				if top.percentIdx == len(percent.ends) {
					top.pair++
					top.percentIdx = 0
					top.highIdx = 0
					continue
				}
				hi := hiUp
				if top.pair&1 != 0 {
					hi = hiLo
				}
				lo := loUp
				if top.pair&2 != 0 {
					lo = loLo
				}
				highKey := decodeKey{at: percent.ends[top.percentIdx], byt: hi}
				highRes, ok := m.memo[highKey]
				if !ok {
					work = append(work, newDecodeFrame(m.text, highKey))
					paused = true
					break
				}
				top.partial = top.partial || highRes.partial
				if top.highIdx == len(highRes.ends) {
					top.percentIdx++
					top.highIdx = 0
					continue
				}
				lowKey := decodeKey{at: highRes.ends[top.highIdx], byt: lo}
				lowRes, ok := m.memo[lowKey]
				if !ok {
					work = append(work, newDecodeFrame(m.text, lowKey))
					paused = true
					break
				}
				top.partial = top.partial || lowRes.partial
				for _, end := range lowRes.ends {
					top.ends = appendUniqueInt(top.ends, end)
				}
				top.highIdx++
			}
			if !paused {
				top.kind = decodeStateFinished
			}
		case decodeStateFinished:
			m.memo[top.key] = decodeResult{ends: top.ends, partial: top.partial}
			work = work[:len(work)-1]
		}
	}
	return m.memo[wanted]
}

func appendUniqueInt(s []int, v int) []int {
	for _, x := range s {
		if x == v {
			return s
		}
	}
	return append(s, v)
}

func containsCredential(text, secret string) bool {
	if secret == "" || text == "" {
		return false
	}
	raw := []byte(text)
	m := newCredentialMatcher(raw)
	for i := 0; i < len(raw); i++ {
		if raw[i] != '%' && raw[i] != secret[0] {
			continue
		}
		if m.matchSecret(i, secret).consumed > 0 {
			return true
		}
		if sensitivePrefixLen(raw[i:], secret) > 0 {
			return true
		}
	}
	return false
}

func sensitivePrefixLen(text []byte, secret string) int {
	if len(text) == 0 || len(secret) == 0 {
		return 0
	}
	if text[0] != '%' && text[0] != secret[0] {
		return 0
	}
	own := 0
	for end := 1; end <= len(text); end++ {
		m := newCredentialMatcher(text[:end])
		found := m.matchSecret(0, secret)
		if found.hold {
			if found.prefixLen >= minCredentialPrefix && end > own {
				own = end
			}
			continue
		}
		if found.prefixLen >= minCredentialPrefix {
			own = end - 1
			break
		}
		own = 0
		break
	}
	return own
}
