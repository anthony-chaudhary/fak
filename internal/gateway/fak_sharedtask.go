package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// SharedTaskRequest is one shared-task co-editing surface call, framed by the
// HTTP handler and adjudicated by the host-installed provider. Path is the
// subtree remainder after /v1/fak/sharedtask/ ("{task_id}", "{task_id}/patch",
// "{task_id}/events"); Scope is the caller-declared reader scope (?scope=) the
// provider uses as the redaction boundary; Body is the raw JSON payload for
// POST (a task record on create, a patch envelope on /patch).
type SharedTaskRequest struct {
	Method string
	Path   string
	Scope  string
	Body   []byte
}

// sharedTaskProvider, when non-nil, adjudicates the shared-task record
// co-editing subtree served at /v1/fak/sharedtask/. It is nil by default, which
// keeps the endpoint inert (404). The request/response types are decoupled (a
// plain struct in, any out) so this package does not import the mechanism-tier
// sharedtask package; the handler only frames HTTP and marshals what the host
// hands back. A host (cmd/fak) installs a provider via SetSharedTaskProvider.
var sharedTaskProvider func(SharedTaskRequest) (int, any)

// SetSharedTaskProvider installs (or, with nil, clears) the shared-task
// co-editing adjudicator for the /v1/fak/sharedtask/ subtree. The provider
// returns the HTTP status plus the JSON body to serve; a status <= 0 reports
// the surface disabled and keeps the endpoint inert.
func SetSharedTaskProvider(provider func(SharedTaskRequest) (int, any)) {
	sharedTaskProvider = provider
}

// SharedTaskProviderInstalled reports whether a provider has been installed. It
// exists so a host can prove its wiring ran without exporting the provider.
func SharedTaskProviderInstalled() bool { return sharedTaskProvider != nil }

// handleFakSharedTask serves the shared-task record co-editing subtree. It
// answers 404 unless a provider is installed AND reports the surface enabled,
// so the endpoint is inert by default. All adjudication (accept / conflict /
// deny / quarantine, scope redaction) happens in the provider's fold; this
// handler only frames the request and encodes the verdict.
func (s *Server) handleFakSharedTask(w http.ResponseWriter, r *http.Request) {
	provider := sharedTaskProvider
	if provider == nil {
		writeSharedTaskDisabled(w)
		return
	}
	var body []byte
	if r.Body != nil {
		b, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		if err != nil {
			http.Error(w, "read request body", http.StatusBadRequest)
			return
		}
		body = b
	}
	status, resp := provider(SharedTaskRequest{
		Method: r.Method,
		Path:   strings.TrimPrefix(r.URL.Path, "/v1/fak/sharedtask/"),
		Scope:  r.URL.Query().Get("scope"),
		Body:   body,
	})
	if status <= 0 {
		writeSharedTaskDisabled(w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(resp)
}

func writeSharedTaskDisabled(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_, _ = w.Write([]byte(`{"error":"shared-task co-editing disabled; enable with FAK_SHAREDTASK=1"}` + "\n"))
}
