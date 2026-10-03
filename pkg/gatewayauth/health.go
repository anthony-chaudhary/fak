// Package gatewayauth provides the gateway health key-possession proof wire.
package gatewayauth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
)

// KeyProofPath proves endpoint key possession without evaluating readiness.
const KeyProofPath = "/v1/fak/key-proof"

// WriteHealthProof adds the existing health proof header for a valid 32-byte
// base64 challenge and a configured key. Missing or invalid challenges and empty
// keys leave the response unchanged. It does not write a status or response body.
// The proof establishes endpoint key possession, not authenticated process
// identity: a proof can be relayed.
func WriteHealthProof(w http.ResponseWriter, r *http.Request, key string) {
	if key != "" {
		encoded := r.Header.Get("X-Fak-Auth-Challenge")
		if nonce, err := base64.StdEncoding.DecodeString(encoded); err == nil && len(nonce) == 32 {
			mac := hmac.New(sha256.New, []byte(key))
			_, _ = mac.Write([]byte("fak-health-v1\x00"))
			_, _ = mac.Write(nonce)
			w.Header().Set("X-Fak-Auth-Proof", base64.StdEncoding.EncodeToString(mac.Sum(nil)))
		}
	}
}
