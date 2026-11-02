package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// This consumer reports the server's retained execution receipts, not an
// independent replay, lifetime performance, or proof of activity right now.
type serveProofRow struct {
	Feature gateway.ServeFeature `json:"feature"`
	State   gateway.FeatureState `json:"state"`
	Verdict string               `json:"verdict"`
	Reason  string               `json:"reason"`
	Receipt *serveProofEvidence  `json:"receipt,omitempty"`
}

// Only validated identities and quantities cross the HTTP-to-output boundary.
// Remote reason/method strings are not a safe diagnostic vocabulary.
type serveProofEvidence struct {
	Sequence   uint64  `json:"sequence"`
	ObservedAt string  `json:"observed_at"`
	SavedBytes *int    `json:"saved_bytes,omitempty"`
	CacheHits  *uint64 `json:"cache_hits,omitempty"`
}

type serveProofReport struct {
	Schema        string          `json:"schema"`
	EvidenceScope string          `json:"evidence_scope"`
	Code          string          `json:"code"`
	Rows          []serveProofRow `json:"rows"`
}

func serveProofReportEmpty() serveProofReport {
	return serveProofReport{Schema: "fak-serve-proof/1", EvidenceScope: "server_retained_receipts_not_current_activity", Code: "FEATURE_PROOF_OBSERVED", Rows: []serveProofRow{}}
}

func runServeProof(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("serve proof", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", "http://127.0.0.1:8080", "running gateway HTTP(S) address")
	timeout := fs.Duration("timeout", 5*time.Second, "total observation deadline (0 < duration <= 1m)")
	tokenEnv := fs.String("token-env", "", "optional environment variable containing the gateway read bearer (never printed)")
	asJSON := fs.Bool("json", false, "emit typed feature verification report")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 || *timeout <= 0 || *timeout > time.Minute {
		fmt.Fprintln(stderr, "fak serve proof: invalid arguments or timeout")
		return 2
	}
	rep := observeServeProof(*addr, *timeout, *tokenEnv)
	if *asJSON {
		if rc := encodeJSONOrFail(stdout, stderr, rep, "fak serve proof"); rc != 0 {
			return rc
		}
	} else {
		writeServeProofHuman(stdout, rep)
	}
	return serveProofExit(rep)
}

func observeServeProof(addr string, timeout time.Duration, tokenEnv string) serveProofReport {
	rep := serveProofReportEmpty()
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	u, err := url.Parse(addr)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Path != "" && u.Path != "/") || timeout <= 0 || timeout > time.Minute {
		rep.Code = "FEATURE_PROOF_INVALID_ARGUMENT"
		return rep
	}
	token := ""
	if tokenEnv != "" {
		token = os.Getenv(tokenEnv)
		if token == "" || strings.ContainsAny(token, "\r\n") {
			rep.Code = "FEATURE_PROOF_AUTH_UNAVAILABLE"
			return rep
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	client := &http.Client{Timeout: timeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	get := func(path string, target any) string {
		endpoint := *u
		endpoint.Path = path
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
		if err != nil {
			return "FEATURE_PROOF_INVALID_ARGUMENT"
		}
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := client.Do(req)
		if err != nil {
			var ne net.Error
			if errors.Is(err, context.DeadlineExceeded) || (errors.As(err, &ne) && ne.Timeout()) {
				return "FEATURE_PROOF_TIMEOUT"
			}
			return "FEATURE_PROOF_UNREACHABLE"
		}
		defer resp.Body.Close()
		switch resp.StatusCode {
		case http.StatusNotFound:
			return "FEATURE_PROOF_UNSUPPORTED"
		case http.StatusUnauthorized, http.StatusForbidden:
			return "FEATURE_PROOF_UNAUTHORIZED"
		case http.StatusOK:
		default:
			return "FEATURE_PROOF_HTTP_ERROR"
		}
		// Never print response bodies or request errors (which may contain URLs
		// and credentials). Reject oversize and trailing data rather than truncate.
		body, err := io.ReadAll(io.LimitReader(resp.Body, (1<<20)+1))
		if err != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return "FEATURE_PROOF_TIMEOUT"
		}
		if err != nil || len(body) > 1<<20 || json.Unmarshal(body, target) != nil {
			return "FEATURE_PROOF_MALFORMED"
		}
		return ""
	}
	var catalog gateway.FeatureCatalog
	var proofs gateway.FeatureProofSummary
	if rep.Code = get("/v1/fak/features", &catalog); rep.Code != "" {
		return rep
	}
	if rep.Code = get("/v1/fak/features/proof", &proofs); rep.Code != "" {
		return rep
	}
	return foldServeProof(catalog, proofs)
}

func foldServeProof(catalog gateway.FeatureCatalog, proofs gateway.FeatureProofSummary) serveProofReport {
	rep := serveProofReportEmpty()
	malformed := func() serveProofReport {
		rep.Code = "FEATURE_PROOF_MALFORMED"
		rep.Rows = []serveProofRow{}
		return rep
	}
	if catalog.Schema != gateway.FeatureSchema || proofs.Schema != gateway.FeatureProofSchema || len(catalog.Features) == 0 || len(catalog.Features) > 256 || len(proofs.Receipts) > gateway.MaxFeatureProofCapacity {
		return malformed()
	}
	latest := map[gateway.ServeFeature]gateway.FeatureProofReceipt{}
	supported := map[gateway.ServeFeature]bool{}
	for _, feature := range proofs.SupportedFeatures {
		supported[feature] = true
	}
	var previous uint64
	for _, receipt := range proofs.Receipts {
		if receipt.Schema != gateway.FeatureProofSchema || receipt.Sequence <= previous || !supported[receipt.Feature] {
			return malformed()
		}
		previous = receipt.Sequence
		if _, err := time.Parse(time.RFC3339Nano, receipt.ObservedAt); err != nil {
			return malformed()
		}
		switch receipt.Outcome {
		case gateway.FeatureProofVerified:
			if !serveReceiptQuantitative(receipt) {
				return malformed()
			}
		case gateway.FeatureProofFailed:
			if receipt.Rewrite != nil || receipt.Cache != nil {
				return malformed()
			}
		default:
			return malformed()
		}
		latest[receipt.Feature] = receipt
	}
	seen := map[gateway.ServeFeature]bool{}
	for _, feature := range catalog.Features {
		if seen[feature.Feature] || !serveProofLabel(string(feature.Feature)) {
			return malformed()
		}
		seen[feature.Feature] = true
		row := serveProofRow{Feature: feature.Feature, State: feature.State}
		switch feature.State {
		case gateway.FeatureDisabled:
			row.Verdict, row.Reason = "DISABLED", "FEATURE_DISABLED"
		case gateway.FeatureRefusedUnavailable:
			row.Verdict, row.Reason = "DEGRADED_OR_BLOCKED", "FEATURE_REFUSED_UNAVAILABLE"
		case gateway.FeatureConfiguredActive, gateway.FeatureConfiguredStandby:
			row.Verdict, row.Reason = "ENABLED_IDLE", "NO_RETAINED_EXECUTION_RECEIPT"
			if receipt, ok := latest[feature.Feature]; ok {
				observed, _ := time.Parse(time.RFC3339Nano, receipt.ObservedAt)
				row.Receipt = &serveProofEvidence{Sequence: receipt.Sequence, ObservedAt: observed.UTC().Format(time.RFC3339Nano)}
				if receipt.Outcome == gateway.FeatureProofVerified {
					if receipt.Rewrite != nil {
						value := receipt.Rewrite.ShedBytes
						row.Receipt.SavedBytes = &value
					}
					if receipt.Cache != nil {
						value := receipt.Cache.HitCount
						row.Receipt.CacheHits = &value
					}
					row.Verdict, row.Reason = "VERIFIED_ACTIVE", "RETAINED_QUANTITATIVE_RECEIPT"
				} else {
					row.Verdict, row.Reason = "DEGRADED_OR_BLOCKED", "PROOF_GENERATION_FAILED"
				}
			}
		default:
			return malformed()
		}
		rep.Rows = append(rep.Rows, row)
	}
	sort.Slice(rep.Rows, func(i, j int) bool { return rep.Rows[i].Feature < rep.Rows[j].Feature })
	return rep
}

func serveProofLabel(s string) bool {
	if len(s) == 0 || len(s) > 64 {
		return false
	}
	for _, c := range s {
		if (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '_' {
			return false
		}
	}
	return true
}

func serveReceiptQuantitative(r gateway.FeatureProofReceipt) bool {
	digest := func(s string) bool { b, err := hex.DecodeString(s); return err == nil && len(b) == 32 }
	if p := r.Rewrite; p != nil && r.Cache == nil {
		if r.Feature != gateway.FeatureCompactHistory && r.Feature != gateway.FeatureElideResults {
			return false
		}
		if p.PostBytes < 0 || p.PreBytes <= p.PostBytes || p.ShedBytes != p.PreBytes-p.PostBytes || p.PrefixBytes < 0 || p.SuffixBytes < 0 || p.PrefixBytes > p.PostBytes || p.SuffixBytes > p.PostBytes-p.PrefixBytes {
			return false
		}
		return digest(p.PreSHA256) && digest(p.PostSHA256) && digest(p.PrePrefixSHA256) && digest(p.PreSuffixSHA256) && p.PrePrefixSHA256 == p.PostPrefixSHA256 && p.PreSuffixSHA256 == p.PostSuffixSHA256
	}
	if p := r.Cache; p != nil && r.Rewrite == nil {
		return r.Feature == gateway.FeatureVDSO && p.HitCount > 0 && p.AdjudicationVerdict.Kind == "ALLOW" && p.AdjudicationVerdict.Reason == "SERVED_INLINE" && p.AdjudicationVerdict.By == "vdso" && digest(p.IdentitySHA256) && digest(p.PayloadSHA256) && digest(p.CacheWitnessHash)
	}
	return false
}

func serveProofExit(rep serveProofReport) int {
	if rep.Code != "FEATURE_PROOF_OBSERVED" {
		return 1
	}
	for _, row := range rep.Rows {
		if row.Verdict == "DEGRADED_OR_BLOCKED" {
			return 1
		}
	}
	return 0
}

func writeServeProofHuman(w io.Writer, rep serveProofReport) {
	fmt.Fprintf(w, "Feature proof: %s (%s)\n", rep.Code, rep.EvidenceScope)
	for _, row := range rep.Rows {
		fmt.Fprintf(w, "%-28s %-20s %s\n", row.Feature, row.Verdict, row.Reason)
	}
}
