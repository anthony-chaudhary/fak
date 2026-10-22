package devcmd

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
)

func RunInferenceTopology(stdout, stderr io.Writer, args []string) int {
	usage := func() {
		fmt.Fprintln(stderr, "usage: fak-dev inference-topology <validate|analyze|inventory|render> --manifest PATH [--at YYYY-MM-DD] [--prose PATH] [--out PATH] [--json]")
	}
	if len(args) == 0 {
		usage()
		return 2
	}
	verb := args[0]
	if verb != "validate" && verb != "analyze" && verb != "inventory" && verb != "render" {
		usage()
		return 2
	}
	fs := flag.NewFlagSet("inference-topology "+verb, flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifest := fs.String("manifest", "", "versioned inference topology manifest (required)")
	at := fs.String("at", "", "inventory review date, default manifest as_of")
	prose := fs.String("prose", "", "UTF-8 prose file to include before generated report appendix")
	out := fs.String("out", "", "write output atomically to this path instead of stdout")
	jsonOutput := fs.Bool("json", false, "JSON validation output; analyze and inventory always emit JSON; render does not accept --json")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	if *manifest == "" || fs.NArg() != 0 || (*prose != "" && verb != "render") || (*at != "" && verb != "inventory") || (*jsonOutput && verb == "render") {
		usage()
		return 2
	}
	m, err := inferencetopology.Load(*manifest)
	if err != nil {
		fmt.Fprintf(stderr, "inference-topology: %v\n", err)
		return 1
	}
	var b bytes.Buffer
	var result any
	switch verb {
	case "validate":
		_, err = inferencetopology.Analyze(m)
		if *jsonOutput {
			result = map[string]any{"schema": "fak-inference-topology-validation/1", "manifest_id": m.ID, "valid": err == nil, "evidence": "ANALYTICAL_NOT_MEASURED"}
		} else if err == nil {
			fmt.Fprintf(&b, "valid inference topology manifest %s: sources=%d claims=%d scenarios=%d; ANALYTICAL_NOT_MEASURED\n", m.ID, len(m.Sources), len(m.Claims), len(m.Scenarios))
		}
	case "analyze":
		result, err = inferencetopology.Analyze(m)
	case "inventory":
		if *at == "" {
			*at = m.AsOf
		}
		result, err = inferencetopology.Reinventory(m, *at)
	case "render":
		var p []byte
		if *prose != "" {
			p, err = os.ReadFile(*prose)
		}
		if err == nil {
			err = inferencetopology.Render(&b, m, string(p))
		}
	}
	if err != nil {
		fmt.Fprintf(stderr, "inference-topology: %v\n", err)
		return 1
	}
	if result != nil {
		enc := json.NewEncoder(&b)
		enc.SetIndent("", "  ")
		if err = enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "inference-topology: encode: %v\n", err)
			return 1
		}
	}
	if *out != "" {
		err = writeStudyAdjacencySummary(*out, b.Bytes())
	} else {
		_, err = stdout.Write(b.Bytes())
	}
	if err != nil {
		fmt.Fprintf(stderr, "inference-topology: write: %v\n", err)
		return 1
	}
	return 0
}
