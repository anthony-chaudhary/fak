// Package probe is the one health/readiness/identity route contract every fak
// deployable serves and every updater checks.
//
// It has two halves over the same routes:
//
//   - Server: Mount registers /healthz, /readyz, /version and, only when a Drain
//     hook is supplied, the admin-authenticated /v1/admin/drain.
//   - Client: Client.Wait checks liveness (/healthz), readiness (/readyz), the
//     model catalog (/v1/models) and identity (/version, compared to the expected
//     build stamp through the deploykit/stamp parser) inside one bounded
//     wait-and-sample loop.
//
// The loop generalizes the serverlifecycle readiness wait: it retries only the
// not-listening and not-ready failure kinds and fails fast on every other kind,
// so an identity drift or an empty catalog is reported at once instead of
// being retried until the deadline.
//
// The drain response keeps the wire shape fak-server already speaks:
// {"status":"drained","active_connections":0,"elapsed_ms":<n>} plus the
// optional observation_snapshot. A drain succeeded only when status is
// "drained" and no connection is still active.
package probe

import (
	"encoding/json"
	"net/http"
)

// The route contract. Every path is absolute from the server origin.
const (
	// PathHealth is liveness: the process is up and answering HTTP.
	PathHealth = "/healthz"
	// PathReady is readiness: the deployable can take traffic.
	PathReady = "/readyz"
	// PathModels is the OpenAI-compatible model catalog the client requires to
	// be non-empty. Mount does not serve it; the deployable's API does.
	PathModels = "/v1/models"
	// PathVersion serves the running build identity as a `version --json`
	// document (app_version, commit, dirty, stamped).
	PathVersion = "/version"
	// PathDrain drains traffic before a swap. It is mounted only with a Drain
	// hook and always behind AdminAuth.
	PathDrain = "/v1/admin/drain"
)

// errorBody is the JSON body of every refusal Mount's own handlers write.
type errorBody struct {
	Error string `json:"error"`
}

// writeJSON writes v as one JSON document followed by a newline, with the same
// headers and json.Encoder framing the existing fak-server handlers use.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
