package gateway

// key_scope_test.go — per-key scope auth adapted from TensorFold
// zig/src/server/auth.zig + zig/src/server/auth_scope.zig at
// 9e8c368b0b4c2c7892a0c9f0f302d7c164016209 (Apache-2.0): exact method+route
// grants (never suffix), conflicting-scope load refusal, and fail-closed
// reload that closes every file key while in-flight requests keep their
// copied scopes. This file holds both arms of the port:
// TestKeyScopeGrantModelAndFailClosedReload pins the grant model and the
// standalone file arm; TestKeysetScopeGrantsExactAndReloadCloses pins the
// keyset seam — scoped load, exact grants through lookupScoped,
// conflicting-scope load refusal, and fail-closed reload. Assertions are
// closed contract: values or sentinel errors via errors.Is, never prose.

import (
	"errors"
	"net/http"
	"testing"
)

func TestKeyScopeGrantModelAndFailClosedReload(t *testing.T) {
	t.Run("grants are exact: wrong method or resource is denied despite suffix overlap", func(t *testing.T) {
		cases := []struct {
			name   string
			scopes keyScopes
			method string
			path   string
			want   bool
		}{
			{"inference POST chat completions", keyScopeInference, http.MethodPost, "/v1/chat/completions", true},
			{"inference POST messages", keyScopeInference, http.MethodPost, "/v1/messages", true},
			{"inference query-normalized still exact", keyScopeInference, http.MethodPost, "/v1/chat/completions?stream=true", true},
			{"inference trailing slash normalized", keyScopeInference, http.MethodPost, "/v1/messages/", true},
			{"inference suffix resource denied", keyScopeInference, http.MethodPost, "/v1/chat/completions/extra", false},
			{"inference sibling route denied", keyScopeInference, http.MethodPost, "/v1/messages/count_tokens", false},
			{"inference wrong method on granted resource", keyScopeInference, http.MethodGet, "/v1/chat/completions", false},
			{"inference may list models", keyScopeInference, http.MethodGet, "/v1/models", true},
			{"inference cannot read observation plane", keyScopeInference, http.MethodGet, "/v1/fak/observation", false},
			{"read GET observation", keyScopeRead, http.MethodGet, "/v1/fak/observation", true},
			{"read GET observation requests", keyScopeRead, http.MethodGet, "/v1/fak/observation/requests", true},
			{"read GET models", keyScopeRead, http.MethodGet, "/v1/models", true},
			{"read cannot infer", keyScopeRead, http.MethodPost, "/v1/messages", false},
			{"read suffix resource denied", keyScopeRead, http.MethodGet, "/v1/fak/observation/requests/x", false},
			{"admin passes every verb and route", fullKeyScopes, http.MethodDelete, "/v1/admin/drain", true},
			{"zero scopes grant nothing", 0, http.MethodGet, "/v1/models", false},
		}
		for _, tc := range cases {
			if got := tc.scopes.allows(tc.method, tc.path); got != tc.want {
				t.Errorf("%s: allows(%s %s) = %v, want %v", tc.name, tc.method, tc.path, got, tc.want)
			}
		}
	})

	t.Run("zero scopes grant nothing; scopes are independent", func(t *testing.T) {
		// The zero value fails closed on every table row and every verb.
		for _, method := range []string{http.MethodGet, http.MethodPost, http.MethodDelete} {
			for _, path := range []string{"/v1/models", "/v1/chat/completions", "/v1/fak/observation"} {
				if keyScopes(0).allows(method, path) {
					t.Errorf("zero scopes allow %s %s; want refused", method, path)
				}
			}
		}
		// Holding read grants no inference route, and vice versa: one bit never
		// widens into another's table.
		if keyScopeRead.allows(http.MethodPost, "/v1/chat/completions") ||
			keyScopeRead.allows(http.MethodPost, "/v1/messages") {
			t.Error("read scope must not reach any inference route")
		}
		if keyScopeInference.allows(http.MethodGet, "/v1/fak/observation") ||
			keyScopeInference.allows(http.MethodGet, "/v1/fak/observation/requests") {
			t.Error("inference scope must not reach the observation plane")
		}
	})

	t.Run("label grammar: unknown repeated or empty scopes refuse", func(t *testing.T) {
		if label, scopes, err := parseKeyScopeLabel("ops"); err != nil || label != "ops" || scopes != fullKeyScopes {
			t.Fatalf("bare label = (%q,%v,%v), want (ops,full,nil)", label, scopes, err)
		}
		if label, scopes, err := parseKeyScopeLabel("ci[read]"); err != nil || label != "ci" || scopes != keyScopeRead {
			t.Fatalf("ci[read] = (%q,%v,%v), want (ci,read,nil)", label, scopes, err)
		}
		if label, scopes, err := parseKeyScopeLabel("both[inference,read]"); err != nil || label != "both" || scopes != keyScopeInference|keyScopeRead {
			t.Fatalf("both[inference,read] = (%q,%v,%v), want (both,inference|read,nil)", label, scopes, err)
		}
		if _, scopes, err := parseKeyScopeLabel("root[admin]"); err != nil || scopes != fullKeyScopes {
			t.Fatalf("root[admin] = (%v,%v), want (full,nil)", scopes, err)
		}
		refusals := []struct {
			raw      string
			sentinel error
		}{
			{"x[admin,read]", ErrInvalidKeyScopes},       // admin mixed with another scope
			{"x[read,read]", ErrInvalidKeyScopes},        // repeated member
			{"x[]", ErrInvalidKeyScopes},                 // empty member
			{"x[bogus]", ErrInvalidKeyScopes},            // unknown scope
			{"x[read", ErrInvalidKeyScopes},              // unterminated bracket
			{"bad label[read]", ErrInvalidKeyScopeLabel}, // space in the name
			{"", ErrInvalidKeyScopeLabel},                // empty label
		}
		for _, rc := range refusals {
			if _, _, err := parseKeyScopeLabel(rc.raw); !errors.Is(err, rc.sentinel) {
				t.Errorf("parseKeyScopeLabel(%q) error = %v, want %v", rc.raw, err, rc.sentinel)
			}
		}
	})

	t.Run("load refusals surface the closed sentinels", func(t *testing.T) {
		f := &scopedKeyFile{}
		if err := f.reload(""); !errors.Is(err, ErrEmptyKeyFile) {
			t.Errorf("empty file error = %v, want ErrEmptyKeyFile", err)
		}
		if err := f.reload("# comment only\n\n# another\n"); !errors.Is(err, ErrEmptyKeyFile) {
			t.Errorf("comment-only file error = %v, want ErrEmptyKeyFile", err)
		}
		if err := f.reload("x[bogus]: sk-one\n"); !errors.Is(err, ErrInvalidKeyScopes) {
			t.Errorf("unknown scope at load = %v, want ErrInvalidKeyScopes", err)
		}
		if err := f.reload("bad label[read]: sk-one\n"); !errors.Is(err, ErrInvalidKeyScopeLabel) {
			t.Errorf("malformed label at load = %v, want ErrInvalidKeyScopeLabel", err)
		}
		if err := f.reload("ops:\n"); err == nil {
			t.Error("a line with an empty key value must refuse the load")
		}
		if err := f.reload("ops: sk one\n"); err == nil {
			t.Error("a key value with whitespace must refuse the load")
		}
		if _, ok := f.match("sk-one"); ok {
			t.Fatal("refused loads must leave the file arm closed")
		}
	})

	t.Run("conflicting scope load refused wholesale; identical scopes dedupe as rotation", func(t *testing.T) {
		f := &scopedKeyFile{}
		err := f.reload("ops: sk-one\nops[read]: sk-one\n")
		if !errors.Is(err, ErrConflictingKeyScopes) {
			t.Fatalf("conflicting load error = %v, want ErrConflictingKeyScopes", err)
		}
		if len(f.entries) != 0 {
			t.Fatalf("a refused load must publish no partial entries, got %d", len(f.entries))
		}
		if _, ok := f.match("sk-one"); ok {
			t.Fatal("a refused first load must leave the file arm closed")
		}

		// The same digest with IDENTICAL scopes is rotation: deduped, not refused.
		g := &scopedKeyFile{}
		if err := g.reload("ci: sk-two\nci: sk-two\n"); err != nil {
			t.Fatalf("identical-scope rotation error = %v, want nil", err)
		}
		if len(g.entries) != 1 {
			t.Fatalf("identical-scope duplicate must dedupe to one entry, got %d", len(g.entries))
		}
		e, ok := g.match("sk-two")
		if !ok || e.principal != "ci" || e.scopes != fullKeyScopes {
			t.Fatalf("rotated key entry = (%q,%v,%v), want (ci,full,true)", e.principal, e.scopes, ok)
		}
	})

	t.Run("invalid reload closes every file key; in-flight keeps copied scopes", func(t *testing.T) {
		f := &scopedKeyFile{}
		if err := f.reload("ops: sk-one\nci[read]: sk-two\n# comment line\n\n"); err != nil {
			t.Fatalf("valid load error = %v", err)
		}
		full, ok := f.match("sk-one")
		if !ok || full.principal != "ops" || full.scopes != fullKeyScopes {
			t.Fatalf("sk-one entry = (%q,%v,%v), want (ops,full,true)", full.principal, full.scopes, ok)
		}
		read, ok := f.match("sk-two")
		if !ok || read.principal != "ci" || read.scopes != keyScopeRead {
			t.Fatalf("sk-two entry = (%q,%v,%v), want (ci,read,true)", read.principal, read.scopes, ok)
		}
		// In-flight request: holds its own value copy of the accepted entry.
		inflight := read

		// An invalid reload (conflicting scopes) closes EVERY file key.
		if err := f.reload("ops: sk-one\nops[read]: sk-one\n"); !errors.Is(err, ErrConflictingKeyScopes) {
			t.Fatalf("refused reload error = %v, want ErrConflictingKeyScopes", err)
		}
		if _, ok := f.match("sk-one"); ok {
			t.Fatal("refused reload must close the previously valid key")
		}
		if _, ok := f.match("sk-two"); ok {
			t.Fatal("refused reload must close every file key, not just the changed one")
		}
		// The in-flight request keeps the scopes it was accepted with.
		if inflight.principal != "ci" || inflight.scopes != keyScopeRead {
			t.Fatalf("in-flight copy = (%q,%v), want (ci,read)", inflight.principal, inflight.scopes)
		}
		if !inflight.scopes.allows(http.MethodGet, "/v1/fak/observation") {
			t.Fatal("in-flight read copy must still allow its granted route")
		}

		// A valid reload reopens exactly the keys it names.
		if err := f.reload("ops: sk-one\n"); err != nil {
			t.Fatalf("re-open load error = %v", err)
		}
		if _, ok := f.match("sk-one"); !ok {
			t.Fatal("re-opened key must match again")
		}
		if _, ok := f.match("sk-two"); ok {
			t.Fatal("a rotated-out key must stay closed after re-open")
		}
	})
}

// TestKeysetScopeGrantsExactAndReloadCloses pins the keyset seam of the port:
// per-key scopes loaded through newScopedKeyset, exact grants on the scopes a
// digest match carries, conflicting-scope load refusal in both arms, and the
// fail-closed reload whose refused load closes every file key while in-flight
// requests keep their copied scopes.
func TestKeysetScopeGrantsExactAndReloadCloses(t *testing.T) {
	t.Run("loaded keyset keeps grants exact: wrong method or resource denied despite suffix overlap", func(t *testing.T) {
		ks, err := newScopedKeyset([]keysetBinding{
			{Key: "sk-infer", Principal: "org-a", Scopes: keyScopeInference},
			{Key: "sk-read", Principal: "org-b", Scopes: keyScopeRead},
			{Key: "sk-full", Principal: "org-c", Scopes: fullKeyScopes},
		})
		if err != nil {
			t.Fatalf("newScopedKeyset error = %v", err)
		}
		cases := []struct {
			name   string
			key    string
			method string
			path   string
			want   bool
		}{
			{"inference POST chat completions", "sk-infer", http.MethodPost, "/v1/chat/completions", true},
			{"inference suffix resource denied", "sk-infer", http.MethodPost, "/v1/chat/completions/extra", false},
			{"inference sibling route denied", "sk-infer", http.MethodPost, "/v1/messages/count_tokens", false},
			{"inference wrong method on granted resource", "sk-infer", http.MethodGet, "/v1/chat/completions", false},
			{"inference may list models", "sk-infer", http.MethodGet, "/v1/models", true},
			{"read GET observation", "sk-read", http.MethodGet, "/v1/fak/observation", true},
			{"read suffix resource denied", "sk-read", http.MethodGet, "/v1/fak/observation/requests/x", false},
			{"read cannot infer", "sk-read", http.MethodPost, "/v1/messages", false},
			{"full passes every verb and route", "sk-full", http.MethodDelete, "/v1/admin/drain", true},
		}
		for _, tc := range cases {
			_, s, ok := ks.lookupScoped(tc.key)
			if !ok {
				t.Fatalf("%s: key %s did not match", tc.name, tc.key)
			}
			if got := s.allows(tc.method, tc.path); got != tc.want {
				t.Errorf("%s: allows(%s %s) = %v, want %v", tc.name, tc.method, tc.path, got, tc.want)
			}
		}
	})

	t.Run("conflicting scope load is refused wholesale", func(t *testing.T) {
		ks, err := newScopedKeyset([]keysetBinding{
			{Key: "k1", Principal: "org-a", Scopes: keyScopeRead},
			{Key: "k1", Principal: "org-a", Scopes: fullKeyScopes},
		})
		if !errors.Is(err, ErrConflictingKeyScopes) {
			t.Fatalf("conflicting bindings error = %v, want ErrConflictingKeyScopes", err)
		}
		if ks != nil {
			t.Fatal("a refused load must not publish a partial keyset")
		}
		// The same digest with IDENTICAL scopes is rotation: deduped, not refused.
		ks, err = newScopedKeyset([]keysetBinding{
			{Key: "k1", Principal: "org-a", Scopes: keyScopeRead},
			{Key: " k1 ", Principal: "org-a", Scopes: keyScopeRead},
		})
		if err != nil {
			t.Fatalf("identical-scope rotation error = %v, want nil", err)
		}
		if p, s, ok := ks.lookupScoped("k1"); !ok || p != "org-a" || s != keyScopeRead {
			t.Fatalf("rotated key lookup = (%q,%v,%v), want (org-a,read,true)", p, s, ok)
		}

		// The file arm refuses the same shadowing inside one load.
		f := &scopedKeyFile{}
		err = f.reload("ops: sk-one\nops[read]: sk-one\n")
		if !errors.Is(err, ErrConflictingKeyScopes) {
			t.Fatalf("file conflict error = %v, want ErrConflictingKeyScopes", err)
		}
		if _, ok := f.match("sk-one"); ok {
			t.Fatal("a refused first load must leave the file arm closed")
		}
	})

	t.Run("invalid reload closes every file key; in-flight keeps copied scopes", func(t *testing.T) {
		f := &scopedKeyFile{}
		if err := f.reload("ops: sk-one\nci[read]: sk-two\n# comment line\n\n"); err != nil {
			t.Fatalf("valid load error = %v", err)
		}
		full, ok := f.match("sk-one")
		if !ok || full.principal != "ops" || full.scopes != fullKeyScopes {
			t.Fatalf("sk-one entry = (%q,%v,%v), want (ops,full,true)", full.principal, full.scopes, ok)
		}
		read, ok := f.match("sk-two")
		if !ok || read.principal != "ci" || read.scopes != keyScopeRead {
			t.Fatalf("sk-two entry = (%q,%v,%v), want (ci,read,true)", read.principal, read.scopes, ok)
		}
		// In-flight request: holds its own value copy of the accepted entry.
		inflight := read

		// An invalid reload (conflicting scopes) closes EVERY file key.
		if err := f.reload("ops: sk-one\nops[read]: sk-one\n"); !errors.Is(err, ErrConflictingKeyScopes) {
			t.Fatalf("refused reload error = %v, want ErrConflictingKeyScopes", err)
		}
		if _, ok := f.match("sk-one"); ok {
			t.Fatal("refused reload must close the previously valid key")
		}
		if _, ok := f.match("sk-two"); ok {
			t.Fatal("refused reload must close every file key, not just the changed one")
		}
		// The in-flight request keeps the scopes it was accepted with.
		if inflight.principal != "ci" || inflight.scopes != keyScopeRead {
			t.Fatalf("in-flight copy = (%q,%v), want (ci,read)", inflight.principal, inflight.scopes)
		}
		if !inflight.scopes.allows(http.MethodGet, "/v1/fak/observation") {
			t.Fatal("in-flight read copy must still allow its granted route")
		}

		// A valid reload reopens exactly the keys it names.
		if err := f.reload("ops: sk-one\n"); err != nil {
			t.Fatalf("re-open load error = %v", err)
		}
		if _, ok := f.match("sk-one"); !ok {
			t.Fatal("re-opened key must match again")
		}
		if _, ok := f.match("sk-two"); ok {
			t.Fatal("a rotated-out key must stay closed after re-open")
		}
	})

	t.Run("keyset entries carry per-key scopes", func(t *testing.T) {
		ks, err := newScopedKeyset([]keysetBinding{
			{Key: "sk-infer", Principal: "org-a", Scopes: keyScopeInference},
			{Key: "sk-admin", Principal: "org-b", Scopes: fullKeyScopes},
		})
		if err != nil {
			t.Fatalf("newScopedKeyset error = %v", err)
		}
		p, s, ok := ks.lookupScoped("sk-infer")
		if !ok || p != "org-a" || s != keyScopeInference {
			t.Fatalf("sk-infer = (%q,%v,%v), want (org-a,inference,true)", p, s, ok)
		}
		if !s.allows(http.MethodPost, "/v1/chat/completions") {
			t.Fatal("inference-scoped key must allow its inference route")
		}
		if s.allows(http.MethodPost, "/v1/admin/drain") {
			t.Fatal("inference-scoped key must not reach the admin plane")
		}
		p, s, ok = ks.lookupScoped("sk-admin")
		if !ok || p != "org-b" || s != fullKeyScopes {
			t.Fatalf("sk-admin = (%q,%v,%v), want (org-b,full,true)", p, s, ok)
		}
		if !s.allows(http.MethodDelete, "/anything") {
			t.Fatal("full-scoped key must pass every route")
		}
		if _, _, ok := ks.lookupScoped("nope"); ok {
			t.Fatal("an unknown key must not match")
		}
		// The legacy unscoped lookup is unchanged by the new field.
		if p, ok := ks.lookup("sk-infer"); !ok || p != "org-a" {
			t.Fatalf("legacy lookup sk-infer = (%q,%v), want (org-a,true)", p, ok)
		}
		var nilKS *keyset
		if _, _, ok := nilKS.lookupScoped("sk-infer"); ok {
			t.Fatal("nil keyset must never match")
		}
	})
}
