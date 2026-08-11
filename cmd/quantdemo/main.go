package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type pins struct {
	Model struct {
		Name, Revision, URL, SHA256, License string
		Bytes                                int64
	}
	Runtime struct {
		ID, License, LicenseSHA256 string
	}
}

type endpointResult struct {
	Mode       string          `json:"mode"`
	URL        string          `json:"url"`
	StatusCode int             `json:"status_code"`
	Body       json.RawMessage `json:"body"`
}

type liveReport struct {
	Contract       string           `json:"contract"`
	Claim          string           `json:"claim"`
	Decision       decision         `json:"decision"`
	ArtifactSHA256 string           `json:"artifact_sha256"`
	Results        []endpointResult `json:"results"`
}

func pinned() pins {
	var p pins
	p.Model.Name, p.Model.Revision, p.Model.URL = ModelName, ModelRevision, ModelURL
	p.Model.SHA256, p.Model.Bytes, p.Model.License = ModelSHA256, ModelBytes, ModelLicense
	p.Runtime.ID, p.Runtime.License, p.Runtime.LicenseSHA256 = RuntimePin, RuntimeLicense, RuntimeLicenseSHA256
	return p
}

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

func run(out, errw io.Writer, args []string) int {
	fs := flag.NewFlagSet("quantdemo", flag.ContinueOnError)
	fs.SetOutput(errw)
	selfcheck := fs.Bool("selfcheck", false, "run deterministic offline contract witness")
	showPins := fs.Bool("pins", false, "print immutable artifact and runtime pins as JSON")
	live := fs.Bool("live", false, "compare the pinned runtime directly and through fak")
	modelPath := fs.String("model", "", "path to the pinned GGUF artifact (required by -live)")
	runtime := fs.String("runtime", RuntimePin, "runtime identity asserted by the operator")
	directURL := fs.String("direct-url", "http://127.0.0.1:8081/v1/chat/completions", "native runtime OpenAI-compatible endpoint")
	fakURL := fs.String("fak-url", "http://127.0.0.1:8080/v1/chat/completions", "same runtime composed through fak")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	switch {
	case *selfcheck:
		if err := selfCheck(out); err != nil {
			fmt.Fprintln(errw, err)
			return 1
		}
		return 0
	case *showPins:
		return writeJSON(out, errw, pinned())
	case *live:
		r, err := compareLive(*modelPath, *runtime, *directURL, *fakURL)
		if err != nil {
			fmt.Fprintln(errw, "quantdemo:", err)
			return 1
		}
		return writeJSON(out, errw, r)
	default:
		fmt.Fprintln(errw, "usage: quantdemo -selfcheck | -pins | -live -model FILE [-runtime PIN] [-direct-url URL] [-fak-url URL]")
		return 2
	}
}

func selfCheck(out io.Writer) error {
	cases := []struct{ name, format, quant, runtime, result, reason string }{
		{"supported", FormatGGUFV3, QuantQ4KM, RuntimePin, ResultCompatible, ReasonKnownCombination},
		{"unknown-format", "mystery@9", QuantQ4KM, RuntimePin, ResultAbstain, ReasonUnknownFormat},
		{"unknown-runtime", FormatGGUFV3, QuantQ4KM, "mystery-runtime@1", ResultRuntimeHandoff, ReasonRuntimeNotPinned},
		{"unsupported-combination", FormatGGUFV3, "iq1_s", RuntimePin, ResultRefuse, ReasonCombinationNotListed},
	}
	for _, tc := range cases {
		got := adjudicate(tc.format, tc.quant, tc.runtime)
		if got.Result != tc.result || got.Reason != tc.reason {
			return fmt.Errorf("%s: got %s/%s, want %s/%s", tc.name, got.Result, got.Reason, tc.result, tc.reason)
		}
	}
	p := pinned()
	if p.Model.SHA256 == "" || p.Model.License == "" || p.Runtime.ID == "" || p.Runtime.LicenseSHA256 == "" {
		return errors.New("pins are incomplete")
	}
	fmt.Fprintf(out, "PASS %s fixtures=%d\n", ContractID, len(cases))
	fmt.Fprintf(out, "PASS pins model=%s sha256=%s license=%s\n", ModelName, ModelSHA256, ModelLicense)
	fmt.Fprintf(out, "PASS runtime=%s license=%s license_sha256=%s\n", RuntimePin, RuntimeLicense, RuntimeLicenseSHA256)
	fmt.Fprintln(out, "PASS typed unknown-format=ABSTAIN unknown-runtime=DELEGATE unsupported-combination=REFUSE")
	fmt.Fprintln(out, "CLAIM composability-only: same pinned artifact/runtime, direct and through fak; no quality or performance winner")
	return nil
}

func compareLive(modelPath, runtime, directURL, fakURL string) (liveReport, error) {
	var report liveReport
	report.Contract = ContractID
	report.Claim = "composability-only: both paths returned an OpenAI-compatible response; no quality or performance winner"
	if modelPath == "" {
		return report, errors.New("-model is required")
	}
	if !validPin(runtime) {
		report.Decision = adjudicate(FormatGGUFV3, QuantQ4KM, runtime)
		return report, fmt.Errorf("%s: got %q, require %q", ReasonRuntimeNotPinned, runtime, RuntimePin)
	}
	format, err := inspectGGUF(modelPath)
	if err != nil {
		return report, err
	}
	hash, size, err := hashFile(modelPath)
	if err != nil {
		return report, err
	}
	if size != ModelBytes || hash != ModelSHA256 {
		return report, fmt.Errorf("artifact pin mismatch: bytes=%d sha256=%s", size, hash)
	}
	report.ArtifactSHA256 = hash
	report.Decision = adjudicate(format, QuantQ4KM, runtime)
	if report.Decision.Result != ResultCompatible {
		return report, fmt.Errorf("%s/%s", report.Decision.Result, report.Decision.Reason)
	}

	payload := []byte(`{"model":"SmolLM2-135M-Instruct","messages":[{"role":"user","content":"Reply with exactly: portable quant demo"}],"temperature":0,"seed":6262,"max_tokens":12,"stream":false}`)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	client := &http.Client{}
	for _, ep := range []struct{ mode, url string }{{"without-fak", directURL}, {"with-fak", fakURL}} {
		res, err := post(ctx, client, ep.mode, ep.url, payload)
		if err != nil {
			return report, err
		}
		report.Results = append(report.Results, res)
	}
	return report, nil
}

func post(ctx context.Context, client *http.Client, mode, url string, payload []byte) (endpointResult, error) {
	r := endpointResult{Mode: mode, URL: url}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return r, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return r, fmt.Errorf("%s request: %w", mode, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return r, err
	}
	r.StatusCode, r.Body = resp.StatusCode, append(json.RawMessage(nil), body...)
	if resp.StatusCode/100 != 2 {
		return r, fmt.Errorf("%s status=%d body=%s", mode, resp.StatusCode, body)
	}
	var valid map[string]any
	if err := json.Unmarshal(body, &valid); err != nil {
		return r, fmt.Errorf("%s invalid JSON: %w", mode, err)
	}
	if _, ok := valid["choices"]; !ok {
		return r, fmt.Errorf("%s response has no choices", mode)
	}
	return r, nil
}

func hashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func writeJSON(out, errw io.Writer, v any) int {
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		fmt.Fprintln(errw, err)
		return 1
	}
	return 0
}
