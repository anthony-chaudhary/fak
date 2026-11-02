package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// fak-test:justify why=regression when=changed:cmd/fak/serve_proof.go,cmd/fak/main.go
// fak-test:runtime fast est=250ms lane=default
func TestServeProofCommand(t *testing.T) {
	catalog := gateway.FeatureCatalog{Schema: gateway.FeatureSchema, Features: []gateway.FeatureStatus{{Feature: gateway.FeatureCompactHistory, State: gateway.FeatureConfiguredActive}}}
	collector, err := gateway.NewFeatureProofCollector(2)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_TEST_FEATURE_PROOF_TOKEN", "synthetic-read-token")
	for _, mode := range []string{"json", "table", "auth", "unsupported", "unauthorized", "malformed", "oversize", "malicious", "timeout", "redirect"} {
		t.Run(mode, func(t *testing.T) {
			var redirected atomic.Int32
			other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { redirected.Add(1) }))
			defer other.Close()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				wantAuth := ""
				if mode == "auth" || mode == "redirect" {
					wantAuth = "Bearer synthetic-read-token"
				}
				if r.Header.Get("Authorization") != wantAuth {
					t.Errorf("authorization binding mismatch")
				}
				if r.Method != http.MethodGet {
					t.Errorf("method=%s", r.Method)
					w.WriteHeader(405)
					return
				}
				if r.URL.Path == "/v1/fak/features" {
					_ = json.NewEncoder(w).Encode(catalog)
					return
				}
				if r.URL.Path != "/v1/fak/features/proof" {
					t.Errorf("path=%s", r.URL.Path)
					w.WriteHeader(404)
					return
				}
				switch mode {
				case "unsupported":
					w.WriteHeader(404)
				case "unauthorized":
					http.Error(w, "private response must not leak", 401)
				case "malformed":
					_, _ = w.Write([]byte(`{"schema":"wrong"}`))
				case "oversize":
					_, _ = w.Write([]byte(strings.Repeat(" ", (1<<20)+2)))
				case "malicious":
					proof := collector.Snapshot()
					proof.Receipts = []gateway.FeatureProofReceipt{{Schema: gateway.FeatureProofSchema, Sequence: 1, ObservedAt: "2026-11-01T00:00:00Z", Feature: gateway.FeatureCompactHistory, Outcome: gateway.FeatureProofFailed, Reason: "private response must not leak", Failure: gateway.FeatureProofFailure("synthetic-read-token")}}
					_ = json.NewEncoder(w).Encode(proof)
				case "timeout":
					select {
					case <-r.Context().Done():
					case <-time.After(time.Second):
					}
				case "redirect":
					http.Redirect(w, r, other.URL, http.StatusTemporaryRedirect)
				default:
					_ = json.NewEncoder(w).Encode(collector.Snapshot())
				}
			}))
			defer srv.Close()
			args := []string{"--addr", srv.URL, "--timeout", "100ms"}
			if mode == "timeout" {
				args = []string{"--addr", srv.URL, "--timeout", "10ms"}
			}
			if mode != "table" {
				args = append(args, "--json")
			}
			if mode == "auth" || mode == "redirect" {
				args = append(args, "--token-env", "FAK_TEST_FEATURE_PROOF_TOKEN")
			}
			var out, errOut bytes.Buffer
			rc := runServeProof(&out, &errOut, args)
			wantCode := map[string]string{"unsupported": "FEATURE_PROOF_UNSUPPORTED", "unauthorized": "FEATURE_PROOF_UNAUTHORIZED", "malformed": "FEATURE_PROOF_MALFORMED", "oversize": "FEATURE_PROOF_MALFORMED", "malicious": "FEATURE_PROOF_OBSERVED", "timeout": "FEATURE_PROOF_TIMEOUT", "redirect": "FEATURE_PROOF_HTTP_ERROR"}[mode]
			if wantCode == "" {
				if rc != 0 || !strings.Contains(out.String(), "ENABLED_IDLE") || strings.Contains(out.String(), "VERIFIED_ACTIVE") {
					t.Fatalf("idle must not fabricate proof: rc=%d out=%s err=%s", rc, out.String(), errOut.String())
				}
			} else if rc != 1 || !strings.Contains(out.String(), wantCode) {
				t.Fatalf("want %s refusal: rc=%d out=%s err=%s", wantCode, rc, out.String(), errOut.String())
			}
			if redirected.Load() != 0 || strings.Contains(out.String()+errOut.String(), "private response") || strings.Contains(out.String()+errOut.String(), "synthetic-read-token") {
				t.Fatal("redirect followed or server body leaked")
			}
		})
	}
}

// fak-test:justify why=regression when=changed:cmd/fak/serve_proof.go
// fak-test:runtime fast est=100ms lane=default
func TestServeProofEvidence(t *testing.T) {
	c, err := gateway.NewFeatureProofCollector(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.RecordRewrite(gateway.FeatureCompactHistory, []byte("prefix-cut-suffix"), []byte("prefix-suffix"), gateway.FeatureRewriteWitness{PrefixBytes: 7, SuffixBytes: 6, ShedBytes: 4}); err != nil {
		t.Fatal(err)
	}
	catalog := gateway.FeatureCatalog{Schema: gateway.FeatureSchema, Features: []gateway.FeatureStatus{{Feature: gateway.FeatureCompactHistory, State: gateway.FeatureConfiguredActive}}}
	if p := foldServeProof(catalog, c.Snapshot()); p.Code != "FEATURE_PROOF_OBSERVED" || p.Rows[0].Verdict != "VERIFIED_ACTIVE" {
		t.Fatalf("actual quantified receipt rejected: %+v", p)
	}
	forged := c.Snapshot()
	forged.Receipts[0].Rewrite.ShedBytes++
	if p := foldServeProof(catalog, forged); p.Code != "FEATURE_PROOF_MALFORMED" {
		t.Fatalf("contradictory evidence accepted: %+v", p)
	}
	if err := c.RecordFailure(gateway.FeatureCompactHistory, gateway.FeatureProofInvalidBody); err != nil {
		t.Fatal(err)
	}
	if p := foldServeProof(catalog, c.Snapshot()); p.Rows[0].Verdict != "DEGRADED_OR_BLOCKED" {
		t.Fatalf("latest proof failure hidden: %+v", p)
	}
	catalog.Features[0].State = gateway.FeatureDisabled
	if p := foldServeProof(catalog, c.Snapshot()); p.Rows[0].Verdict != "DISABLED" {
		t.Fatalf("old receipt overrides disabled state: %+v", p)
	}
	catalog.Features[0].State = gateway.FeatureConfiguredActive
	lifetimeOnly := c.Snapshot()
	lifetimeOnly.Receipts = nil
	if p := foldServeProof(catalog, lifetimeOnly); p.Rows[0].Verdict != "ENABLED_IDLE" {
		t.Fatalf("lifetime counters fabricated retained proof: %+v", p)
	}
}
