package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func runMCPFilterProof(out, errw io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak mcp-filter-proof", flag.ContinueOnError)
	fs.SetOutput(errw)
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON")
	live := fs.Bool("live", false, "run model-driven A/B using OPENAI_API_KEY")
	endpoint := fs.String("endpoint", "", "OpenAI-compatible API base (default OPENAI_BASE_URL or api.openai.com/v1)")
	model := fs.String("model", "", "live proof model (default FAK_MCP_FILTER_PROOF_MODEL or gpt-4.1-mini)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	srv, err := gateway.New(gateway.Config{Model: "proof"})
	if err != nil {
		fmt.Fprintf(errw, "mcp-filter-proof: %v\n", err)
		return 1
	}
	defer srv.Close()
	if *live {
		defaultEndpoint, apiKey, defaultModel := liveProofDefaults()
		if *endpoint == "" {
			*endpoint = defaultEndpoint
		}
		if *model == "" {
			*model = defaultModel
		}
		proof, runErr := runLiveMCPFilterProof(context.Background(), srv, *endpoint, apiKey, *model)
		if runErr != nil {
			fmt.Fprintf(errw, "mcp-filter-proof: %v\n", runErr)
			return 1
		}
		if *jsonOut {
			enc := json.NewEncoder(out)
			enc.SetIndent("", "  ")
			if err := enc.Encode(proof); err != nil {
				fmt.Fprintf(errw, "mcp-filter-proof: encode: %v\n", err)
				return 1
			}
		} else {
			fmt.Fprintf(out, "mcp-filter-proof live: %s — active tasks %.0f%% vs control %.0f%% · recall %.0f%% · first-call %.0f%% · saved %d descriptor bytes\n", proof.Verdict, 100*proof.Active.TaskSuccessRate, 100*proof.Control.TaskSuccessRate, 100*proof.Active.SearchRecall, 100*proof.Active.FirstCallSuccess, proof.Active.SavedDescriptorBytes)
		}
		if proof.Verdict != "PASS" {
			return 3
		}
		return 0
	}
	proof := srv.NativeMCPFilterProof()
	if *jsonOut {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if err := enc.Encode(proof); err != nil {
			fmt.Fprintf(errw, "mcp-filter-proof: encode: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(out, "mcp-filter-proof: %s — tasks %.0f%% · recall %.0f%% · first-call routes %.0f%% · saved %d descriptor bytes\n", proof.Verdict, 100*proof.TaskSuccessRate, 100*proof.SearchRecall, 100*proof.FirstCallRouteSuccess, proof.Active.SavedBytes)
		if proof.Verdict != "PASS" {
			fmt.Fprintf(out, "  bailout: %s (%s); rollback FAK_ABLATE_MCP_TOOL_FILTER=1\n", proof.Active.Mode, proof.Active.Reason)
		}
	}
	if proof.Verdict != "PASS" {
		return 3
	}
	return 0
}

// Kept here so command tests can use one stable JSON comparison helper.
func compactJSON(raw []byte) []byte {
	var dst bytes.Buffer
	if json.Compact(&dst, raw) != nil {
		return raw
	}
	return dst.Bytes()
}
