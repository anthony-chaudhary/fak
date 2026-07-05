package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// account_rehome.go — the operator's "switch seat NOW" button on a live guarded
// session (POST /v1/fak/account/rehome). The account-failover machinery in fak guard
// already heals a session onto a permitted sibling seat when an ACCOUNT-SCOPED 403
// forces it; this route exposes the same swap as an OPERATOR verb, so a human can
// rehome a session onto the next available account on demand (a capped seat, a seat
// wanted for something else) without waiting for the upstream to refuse first.
//
// Same seam pattern as SetSessionEndpointsProvider: the gateway knows nothing about
// config homes or rosters — the host (fak guard) injects the swap function, and the
// route stays inert (404) on a gateway with no account roster in force (`fak serve`,
// a non-subscription guard session). Everything crossing this seam is display
// metadata (seat names, emails, a reason token) — never a credential.

// AccountRehome is the result of one operator-forced seat switch: which seat the
// session moved OFF, which seat now serves its turns, and the recorded reason.
// Emails are advisory display identity, never the credential.
type AccountRehome struct {
	From      string `json:"from"`
	FromEmail string `json:"from_email,omitempty"`
	To        string `json:"to"`
	ToEmail   string `json:"to_email,omitempty"`
	Reason    string `json:"reason"`
}

// accountRehomeRequest is the optional POST body: a free-form reason token recorded
// on the swap (defaulted by the host when empty).
type accountRehomeRequest struct {
	Reason string `json:"reason"`
}

// SetAccountRehomeFunc installs (or, with nil, clears) the operator seat-switch
// function behind POST /v1/fak/account/rehome. fak guard wires it to its
// account-failover state on the pinned Claude-subscription path; every other host
// leaves it unset, keeping the route inert. The function must be safe for concurrent
// use (the guard's failover state already is). Safe on a nil Server.
func (s *Server) SetAccountRehomeFunc(fn func(reason string) (AccountRehome, error)) {
	if s == nil {
		return
	}
	s.accountRehomeMu.Lock()
	s.accountRehomeFn = fn
	s.accountRehomeMu.Unlock()
}

// accountRehomeFunc returns the installed seat-switch function, or nil when the host
// wired none (the inert default).
func (s *Server) accountRehomeFunc() func(reason string) (AccountRehome, error) {
	if s == nil {
		return nil
	}
	s.accountRehomeMu.Lock()
	defer s.accountRehomeMu.Unlock()
	return s.accountRehomeFn
}

// handleFakAccountRehome serves the operator seat switch. POST only; the body is an
// OPTIONAL {"reason": "..."} (an empty or absent body is fine — the host defaults the
// reason). 404 when no host installed a swap function; 409 when the swap function
// found no available sibling seat (the session keeps its current seat — the refusal
// message is fak-authored, safe to surface). A successful swap answers 200 with the
// from→to metadata, and takes effect on the session's NEXT upstream request.
func (s *Server) handleFakAccountRehome(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	fn := s.accountRehomeFunc()
	if fn == nil {
		writeErr(w, http.StatusNotFound, "no account roster is in force on this gateway (account rehome is wired by `fak guard` on a Claude subscription seat)")
		return
	}
	var req accountRehomeRequest
	if r.Body != nil {
		b, err := io.ReadAll(io.LimitReader(r.Body, maxBody))
		if err != nil {
			writeErr(w, http.StatusBadRequest, "unreadable request body")
			return
		}
		if len(bytes.TrimSpace(b)) > 0 {
			if json.Unmarshal(b, &req) != nil {
				writeErr(w, http.StatusBadRequest, "malformed request body: want {\"reason\": \"...\"} or an empty body")
				return
			}
		}
	}
	res, err := fn(strings.TrimSpace(req.Reason))
	if err != nil {
		// The swap function's error is fak-authored operator guidance (never an
		// upstream body), so it crosses to the caller verbatim. 409: the request was
		// well-formed but the roster's current state cannot satisfy it.
		writeErrCode(w, http.StatusConflict, "account_rehome_unavailable", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, res)
}
