package geminicache

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/boundarylint"
)

func TestCachedContentLifecycleCopiesPinnedProviderContract(t *testing.T) {
	now := time.Date(2026, 8, 21, 16, 0, 0, 0, time.UTC)
	model := "models/gemini-2.5-flash"
	prefix := []byte("stable agent instructions and tool context")
	id := NewIdentity("acct-a", "project-a", "us-central1", model, prefix)
	var mu sync.Mutex
	var calls []string
	expires := now.Add(time.Hour)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls = append(calls, r.Method+" "+r.URL.RequestURI())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1beta/cachedContents":
			body, _ := io.ReadAll(r.Body)
			for _, want := range []string{`"model":"` + model + `"`, `"ttl":"3600s"`, `"displayName":"agent-prefix"`} {
				if !strings.Contains(string(body), want) {
					t.Errorf("create body missing %s: %s", want, body)
				}
			}
			json.NewEncoder(w).Encode(CachedContent{Name: "cachedContents/cache-a", Model: model, ExpireTime: expires, UsageMetadata: &UsageMetadata{TotalTokenCount: 900}})
		case r.Method == http.MethodGet && r.URL.Path == "/v1beta/cachedContents/cache-a":
			json.NewEncoder(w).Encode(CachedContent{Name: "cachedContents/cache-a", Model: model, ExpireTime: expires, UsageMetadata: &UsageMetadata{TotalTokenCount: 900}})
		case r.Method == http.MethodPatch && r.URL.Path == "/v1beta/cachedContents/cache-a":
			body, _ := io.ReadAll(r.Body)
			if !strings.Contains(string(body), `"ttl":"7200s"`) {
				t.Errorf("update body did not copy SDK TTL shape: %s", body)
			}
			expires = now.Add(2 * time.Hour)
			json.NewEncoder(w).Encode(CachedContent{Name: "cachedContents/cache-a", Model: model, ExpireTime: expires})
		case r.Method == http.MethodDelete && r.URL.Path == "/v1beta/cachedContents/cache-a":
			w.Write([]byte(`{}`))
		case r.Method == http.MethodGet && r.URL.Path == "/v1beta/cachedContents":
			if r.URL.Query().Get("pageSize") != "10" || r.URL.Query().Get("pageToken") != "next-a" {
				t.Errorf("list query = %s", r.URL.RawQuery)
			}
			w.Write([]byte(`{"cachedContents":[{"name":"cachedContents/cache-a","model":"models/gemini-2.5-flash"}],"nextPageToken":"next-b","futureProviderField":true}`))
		default:
			http.Error(w, "unexpected route", http.StatusNotFound)
		}
	}))
	defer srv.Close()

	client := Client{
		BaseURL: srv.URL + "/v1beta", HTTPClient: srv.Client(), Now: func() time.Time { return now },
		Capabilities: Capabilities{GenerateContent: true, Models: map[string]bool{model: true}, Locations: map[string]bool{"us-central1": true}},
	}
	created, err := client.Create(context.Background(), RouteGenerateContent, id, Admission{
		PredictedReuseValueUSD: 2, CreationStorageCostUSD: 1, TTL: time.Hour, MaxTTL: 4 * time.Hour, PrivacyAllowed: true,
	}, CreateConfig{DisplayName: "agent-prefix", Contents: []Content{{Role: "user", Parts: []Part{{Text: string(prefix)}}}}})
	if err != nil {
		t.Fatal(err)
	}
	if created.State != StateActive || created.Object.UsageMetadata.TotalTokenCount != 900 || created.Schema != ProvenanceSchema {
		t.Fatalf("create receipt did not expose provider state/accounting: %+v", created)
	}
	if _, err := Reference(RouteGenerateContent, id, created, prefix, now); err != nil {
		t.Fatalf("first equivalent generation reference: %v", err)
	}
	if _, err := Reference(RouteGenerateContent, id, created, prefix, now); err != nil {
		t.Fatalf("second equivalent generation reference: %v", err)
	}
	observed, err := client.Get(context.Background(), id, created.Object.Name)
	if err != nil || observed.Object.UsageMetadata.TotalTokenCount != 900 {
		t.Fatalf("observe: receipt=%+v err=%v", observed, err)
	}
	updated, err := client.Update(context.Background(), id, created.Object.Name, UpdateConfig{TTL: "7200s"})
	if err != nil || !updated.Object.ExpireTime.Equal(now.Add(2*time.Hour)) {
		t.Fatalf("update: receipt=%+v err=%v", updated, err)
	}
	objects, next, raw, err := client.List(context.Background(), 10, "next-a")
	if err != nil || len(objects) != 1 || next != "next-b" || !strings.Contains(string(raw), "futureProviderField") {
		t.Fatalf("list: objects=%+v next=%q raw=%s err=%v", objects, next, raw, err)
	}
	deleted, err := client.Delete(context.Background(), id, created.Object.Name)
	if err != nil || deleted.State != StateDeleted {
		t.Fatalf("delete: receipt=%+v err=%v", deleted, err)
	}
	if _, err := Reference(RouteGenerateContent, id, deleted, prefix, now); err == nil {
		t.Fatal("deleted object remained reusable")
	}
	if len(calls) != 5 {
		t.Fatalf("provider calls=%v", calls)
	}
}

func TestCachedContentCreateRequiresNegotiatedCapability(t *testing.T) {
	id := NewIdentity("account", "project", "us-east1", "models/gemini", []byte("prefix"))
	admission := Admission{PredictedReuseValueUSD: 2, CreationStorageCostUSD: 1, TTL: time.Hour, PrivacyAllowed: true}
	cases := []Client{
		{},
		{Capabilities: Capabilities{GenerateContent: true, Models: map[string]bool{"models/other": true}}},
		{Capabilities: Capabilities{GenerateContent: true, Locations: map[string]bool{"us-west1": true}}},
	}
	for _, client := range cases {
		_, err := client.Create(context.Background(), RouteGenerateContent, id, admission, CreateConfig{})
		var unsupported *UnsupportedError
		if !errors.As(err, &unsupported) {
			t.Fatalf("missing typed capability refusal: %v", err)
		}
	}
}

func TestCachedContentRejectsNonResourceNames(t *testing.T) {
	id := NewIdentity("account", "project", "location", "models/gemini", []byte("prefix"))
	client := Client{}
	for _, name := range []string{"", "other/x", "cachedContents/x/extra", "https://attacker.invalid/x"} {
		if _, err := client.Get(context.Background(), id, name); err == nil {
			t.Fatalf("unsafe name %q accepted", name)
		}
	}
}

func TestCachedContentRefusesUnsupportedAndMutatedReferences(t *testing.T) {
	now := time.Now().UTC()
	prefix := []byte("stable")
	id := NewIdentity("account", "project", "location", "models/gemini", prefix)
	receipt := Receipt{Identity: id, State: StateActive, Object: CachedContent{Name: "cachedContents/x", Model: id.Model, ExpireTime: now.Add(time.Hour)}}

	_, err := Reference(RouteInteractions, id, receipt, prefix, now)
	var unsupported *UnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("Interactions refusal was not typed: %v", err)
	}
	for name, mutate := range map[string]func(Identity, Receipt) (Identity, Receipt, []byte){
		"project": func(i Identity, r Receipt) (Identity, Receipt, []byte) { i.Project = "other"; return i, r, prefix },
		"model":   func(i Identity, r Receipt) (Identity, Receipt, []byte) { i.Model = "models/other"; return i, r, prefix },
		"prefix":  func(i Identity, r Receipt) (Identity, Receipt, []byte) { return i, r, []byte("mutated") },
		"expired": func(i Identity, r Receipt) (Identity, Receipt, []byte) {
			r.Object.ExpireTime = now
			return i, r, prefix
		},
	} {
		t.Run(name, func(t *testing.T) {
			i, r, p := mutate(id, receipt)
			if _, err := Reference(RouteGenerateContent, i, r, p, now); err == nil {
				t.Fatal("expected immutable identity/lifecycle refusal")
			}
		})
	}
}

func TestCachedContentAdmissionIsNetTrueAndBounded(t *testing.T) {
	base := Admission{PredictedReuseValueUSD: 2, CreationStorageCostUSD: 1, TTL: time.Hour, MaxTTL: 2 * time.Hour, PrivacyAllowed: true}
	if err := base.Check(); err != nil {
		t.Fatal(err)
	}
	cases := []Admission{base, base, base}
	cases[0].PredictedReuseValueUSD = 1
	cases[1].TTL = 3 * time.Hour
	cases[2].PrivacyAllowed = false
	for _, admission := range cases {
		if err := admission.Check(); err == nil {
			t.Fatalf("unsafe/unprofitable admission accepted: %+v", admission)
		}
	}
}

func TestCachedContentProvenancePinsGeneratedSDK(t *testing.T) {
	raw, err := os.ReadFile("testdata/upstream_provenance.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"fbdae87185987c7fc1abbe9e0b023c614bc8dc3b", "Apache-2.0", "Caches.Create", "Caches.Delete", "CreateCachedContentConfig"} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("provenance missing %q: %s", want, raw)
		}
	}
}

func TestClientDefaultHTTPClientHasTimeout(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate package source")
	}
	findings, err := boundarylint.Scan([]string{filepath.Dir(file)}, boundarylint.DefaultRules())
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 0 {
		t.Fatalf("geminicache violates boundary policy: %v", findings)
	}
}
