package randhex

import (
	"crypto/rand"
	"encoding/hex"
)

// String returns n cryptographic random bytes encoded as lowercase hex.
func String(n int) (string, bool) {
	if n <= 0 {
		return "", true
	}
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", false
	}
	return hex.EncodeToString(b), true
}
