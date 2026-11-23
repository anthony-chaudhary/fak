package gateway

import (
	"net/http"
	"strings"
	"time"
)

const responsesRetrievePrefix = "/v1/responses/"

// responsesNotFoundMessage is shared by every miss so unknown, expired and
// wrong-owner ids stay indistinguishable across isolation principals.
const responsesNotFoundMessage = "response not found"

type responsesDeleted struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

// handleResponsesRetrieve serves GET and DELETE /v1/responses/{id} over the
// same bounded continuation store previous_response_id reads.
func (s *Server) handleResponsesRetrieve(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, responsesRetrievePrefix)
	if id == "" || strings.Contains(id, "/") {
		writeErr(w, http.StatusNotFound, responsesNotFoundMessage)
		return
	}
	owner := strings.TrimSpace(principalFromContext(r.Context()))
	store := s.responsesContinuationState()
	switch r.Method {
	case http.MethodGet:
		snapshot, ok := store.Snapshot(id, owner, time.Now())
		if !ok {
			writeErr(w, http.StatusNotFound, responsesNotFoundMessage)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(snapshot)
	case http.MethodDelete:
		if !store.Delete(id, owner, time.Now()) {
			writeErr(w, http.StatusNotFound, responsesNotFoundMessage)
			return
		}
		writeJSON(w, http.StatusOK, responsesDeleted{ID: id, Object: "response", Deleted: true})
	default:
		w.Header().Set("Allow", "GET, DELETE")
		writeErr(w, http.StatusMethodNotAllowed, "use GET or DELETE")
	}
}
