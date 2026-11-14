package gateway

import (
	"crypto/rand"
	"encoding/hex"
	"os"
	"strings"
)

// newTraceIdentity builds the per-process/pod identity embedded in every
// gateway-minted trace id (#14445). Two live replicas must never mint the same
// id: the identity carries the replica's own name (POD_NAME under Kubernetes,
// else the host name) so an operator can attribute a trace to the pod that
// served it, plus a random nonce so even two hosts sharing a misconfigured
// name stay disjoint. Computed once per process in New.
func newTraceIdentity() string {
	id := strings.ToLower(strings.TrimSpace(os.Getenv("POD_NAME")))
	if id == "" {
		if h, err := os.Hostname(); err == nil {
			id = strings.ToLower(h)
		}
	}
	id = sanitizeTraceIdentity(id)
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		return id // degenerate entropy source: the name alone still separates replicas
	}
	nonce := hex.EncodeToString(b[:])
	if id == "" {
		return nonce
	}
	return id + "-" + nonce
}

// sanitizeTraceIdentity keeps only [a-z0-9], folds runs of any other character
// into a single '-', trims dangling separators, and caps the length so minted
// ids stay header- and file-name-friendly.
func sanitizeTraceIdentity(s string) string {
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
			lastDash = false
		default:
			if !lastDash && b.Len() > 0 {
				b.WriteByte('-')
			}
			lastDash = true
		}
	}
	out := strings.TrimRight(b.String(), "-")
	if len(out) > 32 {
		out = out[:32]
	}
	return out
}
