package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/geminicache"
)

//fak:ctxplan verb=geminicache enters="Google Gemini CachedContent provider lifecycle records and prefix identity descriptors" pages="cached content admission checks, reference resolution, and inspection reports" warms="explicit Gemini CachedContent lifecycle state"
func cmdGeminiCache(argv []string) {
	os.Exit(runGeminiCache(os.Stdout, os.Stderr, argv))
}

func runGeminiCache(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak geminicache", flag.ContinueOnError)
	fs.SetOutput(stderr)

	model := fs.String("model", "models/gemini-2.5-flash", "Gemini model identifier")
	account := fs.String("account", "default", "account label")
	project := fs.String("project", "", "GCP project identifier")
	location := fs.String("location", "us-central1", "GCP location")
	prefix := fs.String("prefix", "", "prompt prefix string for identity computation")
	checkAdmission := fs.Bool("check-admission", false, "validate admission bounds against default economics")
	reuseValueUSD := fs.Float64("predicted-reuse-usd", 1.0, "predicted reuse value in USD")
	creationCostUSD := fs.Float64("creation-cost-usd", 0.5, "creation/storage cost in USD")
	ttl := fs.Duration("ttl", 1*time.Hour, "requested cache TTL")
	maxTTL := fs.Duration("max-ttl", 4*time.Hour, "maximum allowed TTL")
	privacyAllowed := fs.Bool("privacy-allowed", true, "whether privacy policy allows provider storage residency")
	asJSON := fs.Bool("json", false, "emit result as JSON")

	if !parseFlags(fs, argv) {
		return 2
	}

	id := geminicache.NewIdentity(*account, *project, *location, *model, []byte(*prefix))

	adm := geminicache.Admission{
		PredictedReuseValueUSD: *reuseValueUSD,
		CreationStorageCostUSD: *creationCostUSD,
		TTL:                    *ttl,
		MaxTTL:                 *maxTTL,
		PrivacyAllowed:         *privacyAllowed,
	}

	var admErr string
	if *checkAdmission {
		if err := adm.Check(); err != nil {
			admErr = err.Error()
		}
	}

	res := struct {
		Schema         string                `json:"schema"`
		Identity       geminicache.Identity  `json:"identity"`
		Admission      geminicache.Admission `json:"admission,omitempty"`
		AdmissionValid bool                  `json:"admission_valid"`
		AdmissionError string                `json:"admission_error,omitempty"`
		Status         string                `json:"status"`
	}{
		Schema:         geminicache.ProvenanceSchema,
		Identity:       id,
		Admission:      adm,
		AdmissionValid: admErr == "",
		AdmissionError: admErr,
		Status:         "ready",
	}

	if admErr != "" {
		res.Status = "admission_refused"
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			fmt.Fprintf(stderr, "fak geminicache: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "GEMINI CACHEDCONTENT LIFECYCLE ADAPTER\n")
		fmt.Fprintf(stdout, "  schema:        %s\n", res.Schema)
		fmt.Fprintf(stdout, "  account:       %s\n", id.Account)
		fmt.Fprintf(stdout, "  model:         %s\n", id.Model)
		fmt.Fprintf(stdout, "  prefix_digest: %s\n", id.PrefixDigest)
		if *checkAdmission {
			fmt.Fprintf(stdout, "  admission:     %s\n", res.Status)
			if admErr != "" {
				fmt.Fprintf(stdout, "  reason:        %s\n", admErr)
			}
		}
	}

	if admErr != "" {
		return 1
	}
	return 0
}

func init() {
	// Keep compiler verification happy if needed for direct references
	_ = context.Background
	_ = strings.TrimSpace
}
