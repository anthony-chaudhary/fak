package probe

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deploykit/stamp"
)

// Hooks are what a deployable supplies to Mount. Only Ready is required.
type Hooks struct {
	// Ready serves /readyz. Mount registers it unchanged, so the deployable
	// keeps full control of what "ready" means and how it is reported.
	Ready http.HandlerFunc
	// Version returns the identity /version reports. Nil means SelfVersion, the
	// running binary's own stamp.
	Version func() stamp.Identity
	// Drain drains traffic within timeout and reports the result. When nil, no
	// drain route is mounted at all. Mount never runs two Drain calls at once.
	Drain func(ctx context.Context, timeout time.Duration) DrainResponse
	// AdminAuth admits or refuses a drain request before Drain is called. It is
	// required whenever Drain is set.
	AdminAuth func(r *http.Request) bool
}

var (
	// ErrInvalidHooks reports a Mount call that cannot be honored: a nil mux, a
	// nil Ready hook, or a Drain hook without AdminAuth.
	ErrInvalidHooks = errors.New("probe: invalid mount hooks")
	// ErrRouteTaken reports that the mux already serves /readyz, /version or
	// /v1/admin/drain, which Mount owns.
	ErrRouteTaken = errors.New("probe: route already registered")
)

// Mount registers the probe routes on mux:
//
//   - /healthz, unless mux already serves exactly that path, in which case the
//     deployable's own liveness handler is kept;
//   - /readyz, served by Hooks.Ready;
//   - /version, the identity from Hooks.Version as a `version --json` document;
//   - /v1/admin/drain, only when Hooks.Drain is set, and always behind
//     Hooks.AdminAuth.
//
// Mount validates everything before registering anything, so an error leaves
// mux unchanged.
func Mount(mux *http.ServeMux, h Hooks) error {
	switch {
	case mux == nil:
		return fmt.Errorf("%w: nil mux", ErrInvalidHooks)
	case h.Ready == nil:
		return fmt.Errorf("%w: Ready hook is required", ErrInvalidHooks)
	case h.Drain != nil && h.AdminAuth == nil:
		return fmt.Errorf("%w: Drain hook requires AdminAuth", ErrInvalidHooks)
	}
	version := h.Version
	if version == nil {
		version = SelfVersion
	}
	routes := []struct {
		path    string
		handler http.Handler
	}{
		{PathReady, h.Ready},
		{PathVersion, versionHandler(version)},
	}
	if h.Drain != nil {
		routes = append(routes, struct {
			path    string
			handler http.Handler
		}{PathDrain, drainHandler(h.Drain, h.AdminAuth)})
	}
	for _, rt := range routes {
		if serves(mux, rt.path) {
			return fmt.Errorf("%w: %s", ErrRouteTaken, rt.path)
		}
	}
	if !serves(mux, PathHealth) {
		mux.HandleFunc(PathHealth, handleLiveness)
	}
	for _, rt := range routes {
		mux.Handle(rt.path, rt.handler)
	}
	return nil
}

// probeMethods are the methods serves asks the mux about, so a pattern
// registered for one method only (e.g. "POST /v1/admin/drain") is still found.
var probeMethods = []string{
	http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut,
	http.MethodPatch, http.MethodDelete, http.MethodOptions, http.MethodTrace,
}

// serves reports whether mux already has a pattern for exactly path, under any
// method, with or without a method qualifier.
func serves(mux *http.ServeMux, path string) bool {
	for _, method := range probeMethods {
		_, pattern := mux.Handler(&http.Request{Method: method, URL: &url.URL{Path: path}, Header: http.Header{}})
		if i := strings.IndexByte(pattern, ' '); i >= 0 {
			pattern = pattern[i+1:]
		}
		if i := strings.IndexByte(pattern, '/'); i > 0 {
			pattern = pattern[i:]
		}
		if pattern == path {
			return true
		}
	}
	return false
}

// SelfVersion is the running binary's identity: its embedded commit, dirty bit
// and application version, as `version --json` reports them.
func SelfVersion() stamp.Identity {
	s := stamp.Self()
	commit := strings.ToLower(strings.TrimSpace(s.Revision))
	return stamp.Identity{
		AppVersion: stamp.AppVersion(),
		Commit:     commit,
		Dirty:      s.Dirty,
		Stamped:    commit != "",
	}
}

// BearerAuth returns an AdminAuth that admits a request carrying
// "Authorization: Bearer <key>". Both sides are hashed before a constant-time
// compare, so a refusal leaks neither the key's bytes nor its length. An empty
// key admits nothing.
func BearerAuth(key string) func(*http.Request) bool {
	key = strings.TrimSpace(key)
	want := sha256.Sum256([]byte(key))
	return func(r *http.Request) bool {
		if key == "" {
			return false
		}
		scheme, token, ok := strings.Cut(strings.TrimSpace(r.Header.Get("Authorization")), " ")
		if !ok || !strings.EqualFold(scheme, "Bearer") {
			return false
		}
		got := sha256.Sum256([]byte(strings.TrimSpace(token)))
		return subtle.ConstantTimeCompare(got[:], want[:]) == 1
	}
}

func handleLiveness(w http.ResponseWriter, r *http.Request) {
	if !getOrHead(w, r) {
		return
	}
	writeJSON(w, http.StatusOK, struct {
		Status string `json:"status"`
	}{"ok"})
}

func versionHandler(version func() stamp.Identity) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !getOrHead(w, r) {
			return
		}
		writeJSON(w, http.StatusOK, version())
	}
}

// drainHandler refuses before doing anything else unless auth admits the
// request: 401 when no credential was presented, 403 when one was presented
// and refused. Past auth it behaves like fak-server's drain handler: a non-POST
// gets its plain-text 405, drains run one at a time so only one owns
// readiness, and a request cancelled while queued never drains.
func drainHandler(drain func(context.Context, time.Duration) DrainResponse, auth func(*http.Request) bool) http.HandlerFunc {
	var mu sync.Mutex
	return func(w http.ResponseWriter, r *http.Request) {
		if !auth(r) {
			if r.Header.Get("Authorization") == "" && r.Header.Get("X-Api-Key") == "" {
				w.Header().Set("WWW-Authenticate", `Bearer realm="fak-admin"`)
				writeJSON(w, http.StatusUnauthorized, errorBody{"unauthorized"})
				return
			}
			writeJSON(w, http.StatusForbidden, errorBody{"forbidden"})
			return
		}
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if r.Context().Err() != nil {
			return
		}
		writeJSON(w, http.StatusOK, drain(r.Context(), drainTimeout(r)))
	}
}

func getOrHead(w http.ResponseWriter, r *http.Request) bool {
	if r.Method == http.MethodGet || r.Method == http.MethodHead {
		return true
	}
	w.Header().Set("Allow", "GET, HEAD")
	writeJSON(w, http.StatusMethodNotAllowed, errorBody{"method_not_allowed"})
	return false
}
