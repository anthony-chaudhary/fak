package gateway

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const sessionDiscoverySchema = "fak.session-discovery.v1"

type SessionDiscoveryRecord struct {
	Schema           string    `json:"schema"`
	SessionID        string    `json:"session_id"`
	ExecutionEpoch   string    `json:"execution_epoch"`
	RelayURL         string    `json:"relay_url"`
	CapabilityDigest string    `json:"capability_digest"`
	ExpiresAt        time.Time `json:"expires_at"`
	Generation       uint64    `json:"generation"`
}

type SessionDiscoveryPublication struct {
	Record      SessionDiscoveryRecord `json:"record"`
	AccessToken string                 `json:"access_token"`
}

type SessionDiscoveryState struct {
	Record      SessionDiscoveryRecord
	TokenDigest [32]byte
	Revoked     bool
}

func (s *Server) PublishSessionDiscovery(sessionID, relayURL string, ttl time.Duration) (SessionDiscoveryPublication, error) {
	if sessionID == "" || ttl <= 0 || !validSessionRelayURL(relayURL) {
		return SessionDiscoveryPublication{}, errors.New("session discovery requires identity, bounded ttl, and an https relay URL")
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessions[sessionID]
	if sess == nil {
		return SessionDiscoveryPublication{}, errors.New("session not found")
	}
	tokenBytes := make([]byte, 32)
	if _, err := rand.Read(tokenBytes); err != nil {
		return SessionDiscoveryPublication{}, err
	}
	token := hex.EncodeToString(tokenBytes)
	generation := uint64(1)
	if sess.discovery != nil {
		generation = sess.discovery.Record.Generation + 1
	}
	record := SessionDiscoveryRecord{Schema: sessionDiscoverySchema, SessionID: sessionID, ExecutionEpoch: sess.executionEpoch, RelayURL: strings.TrimRight(relayURL, "/"), CapabilityDigest: capabilityDigest(sessionClientCapabilities), ExpiresAt: time.Now().UTC().Add(ttl), Generation: generation}
	digest := sha256.Sum256([]byte(token))
	sess.discovery = &SessionDiscoveryState{Record: record, TokenDigest: digest}
	return SessionDiscoveryPublication{Record: record, AccessToken: token}, nil
}

func rotateSessionDiscoveryEpoch(sess *sessionClientSession, executionEpoch string) {
	if sess.discovery == nil || sess.discovery.Revoked || !time.Now().UTC().Before(sess.discovery.Record.ExpiresAt) {
		return
	}
	sess.discovery.Record.ExecutionEpoch = executionEpoch
	sess.discovery.Record.Generation++
}

func (s *Server) RevokeSessionDiscovery(sessionID string) bool {
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessions[sessionID]
	if sess == nil || sess.discovery == nil {
		return false
	}
	sess.discovery.Revoked = true
	return true
}

func validSessionRelayURL(raw string) bool {
	u, err := url.Parse(raw)
	return err == nil && u.Scheme == "https" && u.Host != "" && u.User == nil && u.RawQuery == "" && u.Fragment == "" && !strings.ContainsAny(raw, "\r\n")
}

func sessionBearer(r *http.Request) string {
	const prefix = "Bearer "
	value := r.Header.Get("Authorization")
	if !strings.HasPrefix(value, prefix) {
		return ""
	}
	return strings.TrimSpace(strings.TrimPrefix(value, prefix))
}

func (s *Server) sessionDiscoveryRecord(sessionID, token string) (SessionDiscoveryRecord, string) {
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessions[sessionID]
	if sess == nil || sess.discovery == nil {
		return SessionDiscoveryRecord{}, "DISCOVERY_NOT_FOUND"
	}
	state := sess.discovery
	if state.Revoked {
		return SessionDiscoveryRecord{}, "DISCOVERY_REVOKED"
	}
	if !time.Now().UTC().Before(state.Record.ExpiresAt) {
		return SessionDiscoveryRecord{}, "DISCOVERY_EXPIRED"
	}
	digest := sha256.Sum256([]byte(token))
	if token == "" || subtle.ConstantTimeCompare(digest[:], state.TokenDigest[:]) != 1 {
		return SessionDiscoveryRecord{}, "DISCOVERY_UNAUTHORIZED"
	}
	if state.Record.ExecutionEpoch != sess.executionEpoch {
		return SessionDiscoveryRecord{}, "STALE_EPOCH"
	}
	return state.Record, ""
}

func (s *Server) authorizeSessionClientForSession(w http.ResponseWriter, r *http.Request, sessionID string) bool {
	if s.authorizeSessionClientSilent(r) {
		return true
	}
	if _, code := s.sessionDiscoveryRecord(sessionID, sessionBearer(r)); code == "" {
		return true
	}
	writeErrCode(w, http.StatusUnauthorized, "REMOTE_AUTH_REQUIRED", "missing, invalid, expired, or revoked session discovery capability")
	return false
}

func (s *Server) authorizeSessionClientSilent(r *http.Request) bool {
	rt := s.clientRuntime()
	rt.mu.Lock()
	token := rt.authToken
	rt.mu.Unlock()
	return token == "" || r.Header.Get(SessionClientTokenHeader) == token
}

func (s *Server) handleFakSessionDiscovery(w http.ResponseWriter, r *http.Request) {
	if !s.handleSessionDiscovery(w, r, strings.TrimPrefix(r.URL.Path, "/v1/fak/discovery/")) {
		http.NotFound(w, r)
	}
}

func writeSessionDiscoveryJSON(w http.ResponseWriter, status int, value any) bool {
	writeJSON(w, status, value)
	return true
}

func (s *Server) handleSessionDiscovery(w http.ResponseWriter, r *http.Request, path string) bool {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		return false
	}
	sessionID := parts[0]
	if len(parts) == 1 && r.Method == http.MethodGet {
		return s.readSessionDiscovery(w, r, sessionID)
	}
	if len(parts) == 2 && parts[1] == "publish" && r.Method == http.MethodPost {
		return s.publishSessionDiscovery(w, r, sessionID)
	}
	if len(parts) == 2 && parts[1] == "revoke" && r.Method == http.MethodPost {
		return s.revokeSessionDiscovery(w, r, sessionID)
	}
	return false
}

func (s *Server) readSessionDiscovery(w http.ResponseWriter, r *http.Request, sessionID string) bool {
	record, code := s.sessionDiscoveryRecord(sessionID, sessionBearer(r))
	if code == "" {
		return writeSessionDiscoveryJSON(w, http.StatusOK, record)
	}
	status := http.StatusUnauthorized
	if code == "DISCOVERY_NOT_FOUND" {
		status = http.StatusNotFound
	}
	if code == "STALE_EPOCH" {
		status = http.StatusConflict
	}
	writeErrCode(w, status, code, "session discovery refused")
	return true
}

func (s *Server) publishSessionDiscovery(w http.ResponseWriter, r *http.Request, sessionID string) bool {
	if !s.authorizeSessionClient(w, r) {
		return true
	}
	var req struct {
		RelayURL   string `json:"relay_url"`
		TTLSeconds int64  `json:"ttl_seconds"`
	}
	if !decodeSessionClientJSON(w, r, &req) {
		return true
	}
	publication, err := s.PublishSessionDiscovery(sessionID, req.RelayURL, time.Duration(req.TTLSeconds)*time.Second)
	if err != nil {
		writeErrCode(w, http.StatusBadRequest, "DISCOVERY_PUBLISH_REFUSED", err.Error())
		return true
	}
	return writeSessionDiscoveryJSON(w, http.StatusCreated, publication)
}

func (s *Server) revokeSessionDiscovery(w http.ResponseWriter, r *http.Request, sessionID string) bool {
	if !s.authorizeSessionClient(w, r) {
		return true
	}
	if !s.RevokeSessionDiscovery(sessionID) {
		writeErrCode(w, http.StatusNotFound, "DISCOVERY_NOT_FOUND", "session discovery not found")
		return true
	}
	return writeSessionDiscoveryJSON(w, http.StatusOK, map[string]any{
		"schema": sessionDiscoverySchema, "session_id": sessionID, "revoked": true,
	})
}
