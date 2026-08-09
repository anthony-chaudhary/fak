// Package secretgate mechanically quarantines and reversibly obfuscates secrets.
package secretgate

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base32"
	"errors"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/canon"
)

const (
	MinimumObfuscateLength = 8
	keyBytes               = 32
)

var placeholderRE = regexp.MustCompile(`\$\$([A-Z0-9_]*_)?([A-Z2-7]{12}):([ULCM])\$\$`)

// Obfuscator substitutes provider-visible secrets with keyed opaque handles and
// restores those handles only at the tool-execution boundary. Its key is never
// exposed by this API.
type Obfuscator struct {
	key     []byte
	mu      sync.RWMutex
	restore map[string]string
	warning string
}

// LoadObfuscator is default-inert. When FAK_SECRETGATE is not truthy,
// it returns nil and leaves today's bytes unchanged.
func LoadObfuscator(keyPath string) (*Obfuscator, error) {
	if !envTruthy(os.Getenv("FAK_SECRETGATE")) {
		return nil, nil
	}
	return loadObfuscator(keyPath, rand.Reader)
}

func loadObfuscator(keyPath string, random io.Reader) (*Obfuscator, error) {
	key, err := os.ReadFile(keyPath)
	if err == nil {
		if len(key) != keyBytes {
			return nil, errors.New("secretgate obfuscation key has invalid length")
		}
		return &Obfuscator{key: key, restore: map[string]string{}}, nil
	}
	key = make([]byte, keyBytes)
	if _, rerr := io.ReadFull(random, key); rerr != nil {
		return nil, rerr
	}
	if mkerr := os.MkdirAll(filepath.Dir(keyPath), 0700); mkerr == nil {
		if werr := os.WriteFile(keyPath, key, 0600); werr == nil {
			_ = os.Chmod(keyPath, 0600)
			return &Obfuscator{key: key, restore: map[string]string{}}, nil
		}
	}
	// Never fail open: an unavailable install store still gets a process-local
	// key. The warning contains no path or key material.
	return &Obfuscator{key: key, restore: map[string]string{}, warning: "secretgate obfuscation key is ephemeral; placeholders will not survive restart"}, nil
}

func envTruthy(v string) bool {
	v = strings.ToLower(strings.TrimSpace(v))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// Warning reports the key-persistence degradation without disclosing key/path.
func (o *Obfuscator) Warning() string {
	if o == nil {
		return ""
	}
	return o.warning
}

// Substitute returns the provider-visible form. Detection is deliberately
// sourced only from canon.SecretPatterns; secretgate owns no parallel list.
func (o *Obfuscator) Substitute(payload string, label string) string {
	if o == nil {
		return payload
	}
	type match struct{ start, end int }
	matches := make([]match, 0, 4)
	for _, pattern := range canon.SecretPatterns {
		for _, loc := range pattern.FindAllStringSubmatchIndex(payload, -1) {
			start, end := loc[0], loc[1]
			// The keyword-proximity detector intentionally includes the key name.
			// Obfuscation preserves that provider-visible context and substitutes
			// only its final credential token.
			if pattern.NumSubexp() > 0 && len(loc) >= 4 && loc[2] >= 0 {
				if tokenStart := trailingTokenStart(payload[start:end]); tokenStart >= 0 {
					start += tokenStart
				}
			}
			if end-start >= MinimumObfuscateLength {
				matches = append(matches, match{start, end})
			}
		}
	}
	if len(matches) == 0 {
		return payload
	}
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].start != matches[j].start {
			return matches[i].start < matches[j].start
		}
		return matches[i].end > matches[j].end
	})
	var out strings.Builder
	cursor := 0
	for _, m := range matches {
		if m.start < cursor {
			continue
		}
		out.WriteString(payload[cursor:m.start])
		out.WriteString(o.substituteMatch(payload[m.start:m.end], label))
		cursor = m.end
	}
	out.WriteString(payload[cursor:])
	return out.String()
}

func trailingTokenStart(match string) int {
	for i := len(match) - 1; i >= 0; i-- {
		c := match[i]
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || strings.ContainsRune("+/_-", rune(c))) {
			return i + 1
		}
	}
	return 0
}

func (o *Obfuscator) substituteMatch(secret, label string) string {
	if len(secret) < 8 {
		return secret
	}
	p := o.placeholder(secret, label)
	o.mu.Lock()
	o.restore[p] = secret
	o.mu.Unlock()
	return p
}

func (o *Obfuscator) placeholder(secret, label string) string {
	mac := hmac.New(sha256.New, o.key)
	_, _ = mac.Write([]byte(secret))
	base := base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(mac.Sum(nil))[:12]
	clean := sanitizeLabel(label)
	if clean != "" {
		clean += "_"
	}
	return "$$" + clean + base + ":" + caseHint(secret) + "$$"
}

func sanitizeLabel(s string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	if b.Len() > 24 {
		return b.String()[:24]
	}
	return b.String()
}
func caseHint(s string) string {
	if s == strings.ToUpper(s) {
		return "U"
	}
	if s == strings.ToLower(s) {
		return "L"
	}
	if len(s) > 0 && strings.ToUpper(s[:1])+strings.ToLower(s[1:]) == s {
		return "C"
	}
	return "M"
}

// RestoreArguments deep-walks model-authored JSON-shaped tool arguments.
func (o *Obfuscator) RestoreArguments(v any) any {
	if o == nil {
		return v
	}
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.restoreValue(v)
}

func (o *Obfuscator) restoreValue(v any) any {
	switch x := v.(type) {
	case string:
		return placeholderRE.ReplaceAllStringFunc(x, func(token string) string {
			if secret, ok := o.restore[token]; ok {
				return secret
			}
			return token
		})
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			out[k] = o.restoreValue(e)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = o.restoreValue(e)
		}
		return out
	default:
		return v
	}
}

// ExecuteRestored keeps transcriptArgs untouched while delivering restored args
// to the execution callback.
func (o *Obfuscator) ExecuteRestored(transcriptArgs map[string]any, execute func(map[string]any) error) error {
	restored, _ := o.RestoreArguments(transcriptArgs).(map[string]any)
	return execute(restored)
}
