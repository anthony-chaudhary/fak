package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func runMCPFilterProof(out, errw io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak mcp-filter-proof", flag.ContinueOnError)
	fs.SetOutput(errw)
	jsonOut := fs.Bool("json", false, "emit fak-native-mcp-filter-proof/1 JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	srv, err := gateway.New(gateway.Config{Model: "proof"})
	if err != nil {
		fmt.Fprintf(errw, "mcp-filter-proof: %v\n", err)
		return 1
	}
	defer srv.Close()
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
