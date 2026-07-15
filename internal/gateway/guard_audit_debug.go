package gateway

import (
	"encoding/json"
	"net/http"
	"sync"
)

// GuardAuditFootprint is the bounded guard decision-journal footprint. OldestUnix
// is zero when no matching journal exists.
type GuardAuditFootprint struct {
	Files      int   `json:"files"`
	Bytes      int64 `json:"bytes"`
	OldestUnix int64 `json:"oldest_unix,omitempty"`
}

var guardAuditDebug = struct {
	sync.RWMutex
	provider func() GuardAuditFootprint
}{}

// SetGuardAuditFootprintProvider installs a process-local pull source for the
// /debug/guard-audit endpoint. Passing nil disables the endpoint's payload.
func SetGuardAuditFootprintProvider(fn func() GuardAuditFootprint) {
	guardAuditDebug.Lock()
	guardAuditDebug.provider = fn
	guardAuditDebug.Unlock()
}

func guardAuditFootprint() (GuardAuditFootprint, bool) {
	guardAuditDebug.RLock()
	fn := guardAuditDebug.provider
	guardAuditDebug.RUnlock()
	if fn == nil {
		return GuardAuditFootprint{}, false
	}
	return fn(), true
}

func handleGuardAuditDebug(w http.ResponseWriter, _ *http.Request) {
	v, ok := guardAuditFootprint()
	if !ok {
		http.NotFound(w, nil)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
