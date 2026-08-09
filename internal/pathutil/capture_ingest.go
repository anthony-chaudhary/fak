package pathutil

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"unicode"
)

// CaptureRefusalReason is an existing member of the ABI's closed refusal vocabulary.
// It is named here (rather than inventing a capture-only reason) because source
// admission deliberately stays below the ABI import boundary used by the
// stdlib-only session journal. The package test pins membership to abi.ReasonSecretExfil.
const CaptureRefusalReason = "SECRET_EXFIL"

// CaptureSourceClass is the non-sensitive category that matched a source.
type CaptureSourceClass string

const (
	CaptureClassCredential CaptureSourceClass = "credential_source"
	CaptureClassLabAccess  CaptureSourceClass = "lab_access_source"
)

// CaptureSourceMatch is a source-admission result. SourceDigest is the only identity a
// refusal may persist; the original source string is intentionally absent.
type CaptureSourceMatch struct {
	Refused      bool
	Reason       string
	Class        CaptureSourceClass
	SourceDigest string
}

// Conservative whole-source deny tables. Entries are path segments, not loose
// substrings, so ordinary public code such as internal/secretgate or prose using
// os.environ remains capturable while credential homes/files and lab-control
// surfaces are refused as a unit.
var credentialSegments = map[string]bool{
	".aws":              true,
	".azure":            true,
	".claude":           true,
	".codex":            true,
	".credentials.json": true,
	".env":              true,
	".gnupg":            true,
	".kube":             true,
	".netrc":            true,
	".npmrc":            true,
	".oauth-token":      true,
	".ssh":              true,
	"credentials":       true,
	"credentials.json":  true,
	"id_ed25519":        true,
	"id_rsa":            true,
	"secrets":           true,
	"secrets.json":      true,
}

var labAccessSegments = map[string]bool{
	".env.slack.local":         true,
	"dgxbridge":                true,
	"fak-private":              true,
	"private-comms-channel.md": true,
}

// sourceKeys are the structured argument/meta fields that can identify what a
// capture read. Values elsewhere (for example a business object's "data" field)
// are content, not source selectors, and are deliberately not mistaken for one.
var sourceKeys = map[string]bool{
	"cmd":        true,
	"command":    true,
	"cwd":        true,
	"file":       true,
	"file_path":  true,
	"filepath":   true,
	"path":       true,
	"repo":       true,
	"repository": true,
	"root":       true,
	"script":     true,
	"source":     true,
	"uri":        true,
	"url":        true,
	"workdir":    true,
}

// CheckCaptureSource evaluates explicit source strings. The first denial wins so a caller
// can pass its strongest source first (for example cwd before secondary hints).
func CheckCaptureSource(sources ...string) CaptureSourceMatch {
	for _, source := range sources {
		if d := decideOne(source); d.Refused {
			return d
		}
	}
	return CaptureSourceMatch{}
}

// CheckCaptureJSON evaluates structured tool arguments plus source-bearing metadata.
// It never returns the extracted strings. Malformed/non-object args are treated
// conservatively as a single possible source selector.
func CheckCaptureJSON(raw []byte, meta map[string]string) CaptureSourceMatch {
	values := make([]string, 0, len(meta)+4)
	if len(raw) > 0 {
		var value any
		if err := json.Unmarshal(raw, &value); err != nil {
			values = append(values, string(raw))
		} else {
			collectSourceValues(value, false, &values)
		}
	}
	for key, value := range meta {
		if sourceKeys[normalizeKey(key)] {
			values = append(values, value)
		}
	}
	return CheckCaptureSource(values...)
}

func collectSourceValues(value any, selected bool, out *[]string) {
	switch v := value.(type) {
	case map[string]any:
		for key, child := range v {
			collectSourceValues(child, sourceKeys[normalizeKey(key)], out)
		}
	case []any:
		for _, child := range v {
			collectSourceValues(child, selected, out)
		}
	case string:
		if selected {
			*out = append(*out, v)
		}
	}
}

func normalizeKey(key string) string {
	return strings.ToLower(strings.TrimSpace(strings.ReplaceAll(key, "-", "_")))
}

func decideOne(source string) CaptureSourceMatch {
	normalized := strings.ToLower(strings.TrimSpace(strings.ReplaceAll(source, `\`, "/")))
	if normalized == "" {
		return CaptureSourceMatch{}
	}
	segments := sourceSegments(normalized)
	for _, segment := range segments {
		if credentialSegments[segment] || strings.HasPrefix(segment, ".env.") {
			return refusal(CaptureClassCredential, normalized)
		}
	}
	for _, segment := range segments {
		if labAccessSegments[segment] {
			return refusal(CaptureClassLabAccess, normalized)
		}
	}
	return CaptureSourceMatch{}
}

func sourceSegments(source string) []string {
	return strings.FieldsFunc(source, func(r rune) bool {
		if unicode.IsSpace(r) {
			return true
		}
		switch r {
		case '/', ':', '=', ',', ';', '|', '&', '"', '\'', '`', '(', ')', '[', ']', '{', '}', '<', '>':
			return true
		}
		return false
	})
}

func refusal(class CaptureSourceClass, normalized string) CaptureSourceMatch {
	sum := sha256.Sum256([]byte(normalized))
	return CaptureSourceMatch{
		Refused:      true,
		Reason:       CaptureRefusalReason,
		Class:        class,
		SourceDigest: "sha256:" + hex.EncodeToString(sum[:]),
	}
}
