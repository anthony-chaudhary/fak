package gateway

// session_client.go is the first shared-client spine: terminal and browser clients attach to
// one logical served session through the same descriptor, replay cursor, input lease, and
// typed action route. Transport coordinates remain descriptor metadata, never identity.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

const sessionClientSchema = "fak-session-client/1"
const SessionClientTokenHeader = "X-Fak-Session-Token"

var sessionClientCapabilities = []string{"approve", "close", "deny", "detach", "effect_recovery", "observe", "replay", "terminal_transcript", "text_input"}

type SessionClientDescriptor struct {
	Schema             string              `json:"schema"`
	SessionID          string              `json:"session_id"`
	ExecutionEpoch     string              `json:"execution_epoch"`
	EventHead          uint64              `json:"event_head"`
	Pending            string              `json:"pending_interaction,omitempty"`
	RecoveryDependency string              `json:"recovery_dependency,omitempty"`
	Capabilities       []string            `json:"capabilities"`
	CapabilityDigest   string              `json:"capability_digest"`
	Endpoint           string              `json:"endpoint"`
	State              SessionState        `json:"state"`
	Terminal           SessionTerminalView `json:"terminal"`
	Effects            []SessionEffect     `json:"effects,omitempty"`
}

type SessionTerminalView struct {
	Transcript string `json:"transcript"`
	ByteLength int    `json:"byte_length"`
	Digest     string `json:"digest"`
}

type SessionEffectVerdict string

const (
	SessionEffectKnownNotRun SessionEffectVerdict = "KNOWN_NOT_RUN"
	SessionEffectConfirmed   SessionEffectVerdict = "CONFIRMED"
	SessionEffectUncertain   SessionEffectVerdict = "UNCERTAIN"
)

type SessionEffect struct {
	ID      string               `json:"id"`
	Command string               `json:"command,omitempty"`
	Verdict SessionEffectVerdict `json:"verdict"`
	Check   string               `json:"check,omitempty"`
}

type SessionClientAttachRequest struct {
	ClientKind  string `json:"client_kind"`
	WorkspaceID string `json:"workspace_id"`
	Since       uint64 `json:"since"`
}

type SessionClientAttachResponse struct {
	Schema       string                  `json:"schema"`
	Descriptor   SessionClientDescriptor `json:"descriptor"`
	AttachmentID string                  `json:"attachment_id"`
	InputLease   bool                    `json:"input_lease"`
	Events       []SessionChangeEvent    `json:"events"`
	Cursor       uint64                  `json:"cursor"`
}

type SessionClientActionRequest struct {
	AttachmentID   string `json:"attachment_id"`
	ExecutionEpoch string `json:"execution_epoch"`
	Text           string `json:"text"`
	Principal      string `json:"principal,omitempty"`
}

type SessionClientDetachRequest struct {
	AttachmentID string `json:"attachment_id"`
}

type SessionClientDecisionRequest struct {
	AttachmentID string `json:"attachment_id"`
	Decision     string `json:"decision"`
}

type sessionClientAttachment struct {
	SessionID   string
	WorkspaceID string
	Kind        string
	TouchedAt   time.Time
}

type sessionClientSession struct {
	transcript         []byte
	effects            map[string]SessionEffect
	recoveryDependency string
	executionEpoch     string
	placement          SessionPlacement
	moveHooks          SessionMoveHooks
	moving             bool
	lastMove           []SessionMoveTransition
	discovery          *SessionDiscoveryState
}

type sessionClientRuntime struct {
	mu          sync.Mutex
	serial      uint64
	epoch       string
	attachments map[string]sessionClientAttachment
	leases      map[string]string
	sessions    map[string]*sessionClientSession
	authToken   string
	workspace   map[string]string
}

var sessionClientRuntimes sync.Map // map[*Server]*sessionClientRuntime; avoids a second persisted store

func (s *Server) clientRuntime() *sessionClientRuntime {
	if v, ok := sessionClientRuntimes.Load(s); ok {
		return v.(*sessionClientRuntime)
	}
	epoch := fmt.Sprintf("epoch-%x", sha256.Sum256([]byte(fmt.Sprintf("%p-%d", s, time.Now().UnixNano()))))[:22]
	rt := &sessionClientRuntime{epoch: epoch, attachments: map[string]sessionClientAttachment{}, leases: map[string]string{}, sessions: map[string]*sessionClientSession{}, authToken: strings.TrimSpace(os.Getenv("FAK_SESSION_CLIENT_TOKEN")), workspace: map[string]string{}}
	actual, _ := sessionClientRuntimes.LoadOrStore(s, rt)
	return actual.(*sessionClientRuntime)
}

func (rt *sessionClientRuntime) sessionLocked(sessionID string) *sessionClientSession {
	sess := rt.sessions[sessionID]
	if sess == nil {
		sess = &sessionClientSession{effects: map[string]SessionEffect{}, executionEpoch: rt.epoch}
		rt.sessions[sessionID] = sess
	}
	return sess
}

// RestoreSessionClientState hydrates a materialized view obtained by replaying
// the canonical session journal. It does not restore attachments or leases: a
// restarted daemon always issues a fresh execution epoch and writer fence.
func (s *Server) RestoreSessionClientState(sessionID string, transcript []byte, effects []SessionEffect) error {
	return s.RestoreSessionClientStateWithDependency(sessionID, transcript, effects, "")
}

func (s *Server) RestoreSessionClientStateWithDependency(sessionID string, transcript []byte, effects []SessionEffect, dependency string) error {
	if s == nil || sessionID == "" {
		return fmt.Errorf("session restore requires session_id")
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessionLocked(sessionID)
	sess.recoveryDependency = dependency
	sess.transcript = append(sess.transcript[:0], transcript...)
	sess.effects = map[string]SessionEffect{}
	for _, effect := range effects {
		if effect.ID == "" {
			return fmt.Errorf("session restore contains empty effect id")
		}
		if effect.Verdict != SessionEffectKnownNotRun && effect.Verdict != SessionEffectConfirmed && effect.Verdict != SessionEffectUncertain {
			return fmt.Errorf("session restore effect %q has invalid verdict %q", effect.ID, effect.Verdict)
		}
		sess.effects[effect.ID] = effect
	}
	return nil
}

// RecordSessionTerminalOutput appends exact PTY bytes to the logical session view.
// Clients may disappear and reattach without becoming transcript owners.
func (s *Server) RecordSessionTerminalOutput(sessionID string, output []byte) {
	if s == nil || sessionID == "" || len(output) == 0 {
		return
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessionLocked(sessionID)
	sess.transcript = append(sess.transcript, output...)
}

// BeginSessionEffect writes the intent before an external effect is attempted.
// An unresolved intent is deliberately UNCERTAIN after recovery and is never
// represented as replayable input.
func (s *Server) BeginSessionEffect(sessionID, effectID, command, check string) error {
	if s == nil || sessionID == "" || effectID == "" {
		return fmt.Errorf("session effect requires session_id and effect_id")
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessionLocked(sessionID)
	if _, exists := sess.effects[effectID]; exists {
		return fmt.Errorf("session effect %q already exists", effectID)
	}
	sess.effects[effectID] = SessionEffect{ID: effectID, Command: command, Verdict: SessionEffectUncertain, Check: check}
	return nil
}

// ResolveSessionEffect records independently checked knowledge about an intent.
// Only the two conclusive verdicts may replace UNCERTAIN.
func (s *Server) ResolveSessionEffect(sessionID, effectID string, verdict SessionEffectVerdict) error {
	if verdict != SessionEffectKnownNotRun && verdict != SessionEffectConfirmed {
		return fmt.Errorf("session effect verdict %q is not conclusive", verdict)
	}
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessions[sessionID]
	if sess == nil {
		return fmt.Errorf("session %q has no effects", sessionID)
	}
	effect, ok := sess.effects[effectID]
	if !ok {
		return fmt.Errorf("session effect %q not found", effectID)
	}
	effect.Verdict = verdict
	sess.effects[effectID] = effect
	return nil
}

func (rt *sessionClientRuntime) sessionView(sessionID string) (SessionTerminalView, []SessionEffect, string) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	sess := rt.sessions[sessionID]
	if sess == nil {
		return terminalView(nil), nil, ""
	}
	effects := make([]SessionEffect, 0, len(sess.effects))
	for _, effect := range sess.effects {
		effects = append(effects, effect)
	}
	sort.Slice(effects, func(i, j int) bool { return effects[i].ID < effects[j].ID })
	return terminalView(sess.transcript), effects, sess.recoveryDependency
}

func terminalView(transcript []byte) SessionTerminalView {
	sum := sha256.Sum256(transcript)
	return SessionTerminalView{Transcript: string(transcript), ByteLength: len(transcript), Digest: "sha256:" + hex.EncodeToString(sum[:])}
}

// ConfigureSessionClientAuth installs the per-user local IPC bearer.
func (s *Server) ConfigureSessionClientAuth(token string) {
	rt := s.clientRuntime()
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.authToken = token
}

func (s *Server) authorizeSessionClient(w http.ResponseWriter, r *http.Request) bool {
	rt := s.clientRuntime()
	rt.mu.Lock()
	token := rt.authToken
	rt.mu.Unlock()
	if token == "" || r.Header.Get(SessionClientTokenHeader) == token {
		return true
	}
	writeErrCode(w, http.StatusUnauthorized, "LOCAL_AUTH_REQUIRED", "missing or invalid local session capability")
	return false
}

func capabilityDigest(caps []string) string {
	ordered := append([]string(nil), caps...)
	sort.Strings(ordered)
	sum := sha256.Sum256([]byte(strings.Join(ordered, "\n")))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func (s *Server) clientDescriptor(r *http.Request, sessionID string) (SessionClientDescriptor, bool) {
	if s.observeSession == nil {
		return SessionClientDescriptor{}, false
	}
	st := s.observeSession(r.Context(), sessionID)
	if st.TraceID == "" {
		return SessionClientDescriptor{}, false
	}
	_, head := s.sessionChangesFor(sessionID, 0)
	caps := append([]string(nil), sessionClientCapabilities...)
	rt := s.clientRuntime()
	terminal, effects, dependency := rt.sessionView(sessionID)
	rt.mu.Lock()
	epoch := rt.sessionLocked(sessionID).executionEpoch
	rt.mu.Unlock()
	return SessionClientDescriptor{
		Schema: sessionClientSchema, SessionID: sessionID, ExecutionEpoch: epoch,
		EventHead: head, Capabilities: caps, CapabilityDigest: capabilityDigest(caps),
		Endpoint: requestBaseURL(r) + "/v1/fak/session/" + sessionID, State: st,
		Terminal: terminal, Effects: effects, RecoveryDependency: dependency,
	}, true
}

func (s *Server) sessionClientDescriptorForContext(ctx context.Context, sessionID string) (SessionClientDescriptor, bool) {
	if s == nil || s.observeSession == nil || s.sessionFeed == nil {
		return SessionClientDescriptor{}, false
	}
	st := s.observeSession(ctx, sessionID)
	if st.TraceID == "" {
		return SessionClientDescriptor{}, false
	}
	_, head := s.sessionChangesFor(sessionID, 0)
	caps := append([]string(nil), sessionClientCapabilities...)
	rt := s.clientRuntime()
	terminal, effects, dependency := rt.sessionView(sessionID)
	rt.mu.Lock()
	epoch := rt.sessionLocked(sessionID).executionEpoch
	rt.mu.Unlock()
	return SessionClientDescriptor{Schema: sessionClientSchema, SessionID: sessionID, ExecutionEpoch: epoch, EventHead: head, Capabilities: caps, CapabilityDigest: capabilityDigest(caps), State: st, Terminal: terminal, Effects: effects, RecoveryDependency: dependency}, true
}

func requestBaseURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	if p := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto")); p != "" {
		scheme = strings.Split(p, ",")[0]
	}
	return scheme + "://" + r.Host
}

func decodeSessionClientJSON(w http.ResponseWriter, r *http.Request, dst any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		writeErrCode(w, http.StatusBadRequest, "INVALID_CLIENT_REQUEST", err.Error())
		return false
	}
	return true
}

func (s *Server) handleFakSessionClient(w http.ResponseWriter, r *http.Request, sessionID, verb string) bool {
	known := verb == "client" || verb == "attach" || verb == "move" || verb == "input" || verb == "decision" || verb == "close" || verb == "detach" || verb == "open"
	if !known {
		return false
	}
	if !s.authorizeSessionClientForSession(w, r, sessionID) {
		return true
	}
	if s.sessionFeed == nil {
		if known {
			writeErrCode(w, http.StatusNotFound, "SESSION_CLIENT_UNAVAILABLE", "session client feed is not configured")
			return true
		}
		return false
	}
	if verb == "open" {
		s.handleSessionClientPage(w, r, sessionID)
		return true
	}
	desc, ok := s.clientDescriptor(r, sessionID)
	if !ok {
		writeErrCode(w, http.StatusNotFound, "SESSION_NOT_FOUND", "logical session not found")
		return true
	}
	if verb == "client" {
		if !requireMethod(w, r, http.MethodGet) {
			return true
		}
		writeJSON(w, http.StatusOK, desc)
		return true
	}
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return true
	}
	rt := s.clientRuntime()
	switch verb {
	case "attach":
		var req SessionClientAttachRequest
		if !decodeSessionClientJSON(w, r, &req) {
			return true
		}
		if strings.TrimSpace(req.WorkspaceID) == "" {
			req.WorkspaceID = "default"
		}
		rt.mu.Lock()
		if bound := rt.workspace[sessionID]; bound != "" && bound != req.WorkspaceID {
			rt.mu.Unlock()
			writeErrCode(w, http.StatusConflict, "WORKSPACE_MISMATCH", "logical session is bound to another workspace")
			return true
		}
		rt.workspace[sessionID] = req.WorkspaceID
		rt.serial++
		id := fmt.Sprintf("att-%d", rt.serial)
		rt.attachments[id] = sessionClientAttachment{SessionID: sessionID, WorkspaceID: req.WorkspaceID, Kind: strings.TrimSpace(req.ClientKind), TouchedAt: time.Now().UTC()}
		lease := false
		if rt.leases[sessionID] == "" {
			rt.leases[sessionID] = id
			lease = true
		}
		rt.mu.Unlock()
		events, cursor, _ := s.sessionFeed.drainTrace(sessionID, req.Since)
		desc.EventHead = cursor
		writeJSON(w, http.StatusOK, SessionClientAttachResponse{Schema: sessionClientSchema, Descriptor: desc, AttachmentID: id, InputLease: lease, Events: events, Cursor: cursor})
	case "move":
		if r.Method == http.MethodPost {
			return s.handleSessionMove(w, r, sessionID)
		}
	case "input":
		var req SessionClientActionRequest
		if !decodeSessionClientJSON(w, r, &req) {
			return true
		}
		if req.ExecutionEpoch != desc.ExecutionEpoch {
			writeErrCode(w, http.StatusConflict, "STALE_EPOCH", "session execution epoch changed; describe and reattach")
			return true
		}
		rt.mu.Lock()
		att, exists := rt.attachments[req.AttachmentID]
		held := rt.leases[sessionID] == req.AttachmentID
		if exists {
			att.TouchedAt = time.Now().UTC()
			rt.attachments[req.AttachmentID] = att
		}
		rt.mu.Unlock()
		if !exists || att.SessionID != sessionID {
			writeErrCode(w, http.StatusUnauthorized, "ATTACHMENT_NOT_FOUND", "attach before acting")
			return true
		}
		if !held {
			writeErrCode(w, http.StatusConflict, "LEASE_NOT_HELD", "another client holds the input lease")
			return true
		}
		if strings.TrimSpace(req.Text) == "" {
			writeErrCode(w, http.StatusBadRequest, "EMPTY_INPUT", "text is required")
			return true
		}
		if s.steerSession == nil {
			writeErrCode(w, http.StatusNotImplemented, "INPUT_UNAVAILABLE", "session runtime does not accept text input")
			return true
		}
		principal := kernelPrincipal(r)
		if !principal.IsHuman() {
			writeErrCode(w, http.StatusForbidden, "principal_not_human", "text input consumes user authority and must arrive on the human control wire")
			return true
		}
		if err := s.steerSession(r.Context(), sessionID, stampedSteerPrincipal(principal, req.Principal), req.Text); err != nil {
			writeErrCode(w, http.StatusUnprocessableEntity, "INPUT_REFUSED", err.Error())
			return true
		}
		st := s.observeSession(r.Context(), sessionID)
		s.PublishSessionRevision(st)
		events, cursor, _ := s.sessionFeed.drainTrace(sessionID, desc.EventHead)
		desc.State, desc.EventHead = st, cursor
		writeJSON(w, http.StatusOK, SessionClientAttachResponse{Schema: sessionClientSchema, Descriptor: desc, AttachmentID: req.AttachmentID, InputLease: true, Events: events, Cursor: cursor})
	case "decision":
		var req SessionClientDecisionRequest
		if !decodeSessionClientJSON(w, r, &req) {
			return true
		}
		decision := strings.ToUpper(strings.TrimSpace(req.Decision))
		if decision != "APPROVE" && decision != "DENY" {
			writeErrCode(w, http.StatusBadRequest, "DECISION_INVALID", "decision must be APPROVE or DENY")
			return true
		}
		rt := s.clientRuntime()
		rt.mu.Lock()
		att, ok := rt.attachments[req.AttachmentID]
		lease := rt.leases[sessionID]
		rt.mu.Unlock()
		if !ok || att.SessionID != sessionID || lease != req.AttachmentID {
			writeErrCode(w, http.StatusConflict, "LEASE_NOT_HELD", "attachment does not hold the input lease")
			return true
		}
		desc, _ := s.clientDescriptor(r, sessionID)
		desc.Pending = ""
		writeJSON(w, http.StatusOK, map[string]any{"schema": sessionClientSchema, "session_id": sessionID, "decision": decision, "descriptor": desc})
	case "close":
		var req SessionClientDetachRequest
		if !decodeSessionClientJSON(w, r, &req) {
			return true
		}
		rt := s.clientRuntime()
		rt.mu.Lock()
		att, ok := rt.attachments[req.AttachmentID]
		if ok && att.SessionID == sessionID {
			delete(rt.attachments, req.AttachmentID)
			delete(rt.leases, sessionID)
		}
		rt.mu.Unlock()
		if !ok || att.SessionID != sessionID {
			writeErrCode(w, http.StatusNotFound, "ATTACHMENT_NOT_FOUND", "attachment not found")
			return true
		}
		writeJSON(w, http.StatusOK, map[string]any{"schema": sessionClientSchema, "session_id": sessionID, "closed": true})
	case "detach":
		var req SessionClientDetachRequest
		if !decodeSessionClientJSON(w, r, &req) {
			return true
		}
		rt.mu.Lock()
		att, exists := rt.attachments[req.AttachmentID]
		if exists && att.SessionID == sessionID {
			delete(rt.attachments, req.AttachmentID)
			if rt.leases[sessionID] == req.AttachmentID {
				delete(rt.leases, sessionID)
			}
		}
		rt.mu.Unlock()
		if !exists || att.SessionID != sessionID {
			writeErrCode(w, http.StatusNotFound, "ATTACHMENT_NOT_FOUND", "attachment not found")
			return true
		}
		writeJSON(w, http.StatusOK, map[string]any{"schema": sessionClientSchema, "session_id": sessionID, "detached": true})
	}
	return true
}

var sessionClientPage = template.Must(template.New("session-client").Parse(`<!doctype html>
<html><head><meta charset="utf-8"><title>fak session {{.}}</title><style>
body{font:15px ui-monospace,monospace;background:#101418;color:#e8edf2;max-width:920px;margin:2rem auto;padding:0 1rem}button,input{font:inherit}code,pre{background:#192128;padding:.5rem;white-space:pre-wrap}.ok{color:#72e09a}.err{color:#ff8080}</style></head>
<body><h1>fak session <code id="sid">{{.}}</code></h1><p id="identity">attaching…</p><h2>Full advertised capabilities</h2><ul id="caps"></ul><h2>Shared event tail</h2><pre id="events"></pre><form id="input"><input id="text" size="70" autocomplete="off"><button>Send through shared input lease</button></form><p id="status"></p>
<script>
const sid={{printf "%q" .}}, base='/v1/fak/session/'+encodeURIComponent(sid); let attachment='', epoch='', cursor=0;
async function call(path,body){const r=await fetch(base+path,{method:body?'POST':'GET',headers:{'content-type':'application/json'},body:body?JSON.stringify(body):undefined});const j=await r.json();if(!r.ok)throw new Error((j.error&&j.error.code)+': '+(j.error&&j.error.message));return j}
function render(d){epoch=d.execution_epoch;cursor=d.event_head;identity.textContent='logical='+d.session_id+' epoch='+epoch+' head='+cursor+' capability='+d.capability_digest;caps.innerHTML=d.capabilities.map(x=>'<li>'+x+'</li>').join('')}
(async()=>{try{const d=await call('/client');render(d);const a=await call('/attach',{client_kind:'browser',since:cursor});attachment=a.attachment_id;render(a.descriptor);events.textContent=JSON.stringify(a.events,null,2);status.textContent=a.input_lease?'input lease held':'observe only: another client holds input lease';status.className='ok'}catch(e){status.textContent=e;status.className='err'}})();
input.onsubmit=async e=>{e.preventDefault();try{const a=await call('/input',{attachment_id:attachment,execution_epoch:epoch,text:text.value,principal:'browser'});events.textContent=JSON.stringify(a.events,null,2);render(a.descriptor);text.value='';status.textContent='addressed once at '+a.cursor;status.className='ok'}catch(e){status.textContent=e;status.className='err'}};
</script></body></html>`))

func (s *Server) handleSessionClientPage(w http.ResponseWriter, r *http.Request, sessionID string) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	if _, ok := s.clientDescriptor(r, sessionID); !ok {
		writeErrCode(w, http.StatusNotFound, "SESSION_NOT_FOUND", "logical session not found")
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = sessionClientPage.Execute(w, sessionID)
}
