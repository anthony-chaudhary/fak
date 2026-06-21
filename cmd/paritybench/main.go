// Command paritybench assembles the CROSS-MODEL parity artifact: it ingests the
// live fak-agent A/B reports for a ladder of LOCAL models (produced by
// tools/run_local_model.sh via the OpenAI-compatible shim) plus the committed
// FRONTIER reference cards (hosted Claude Haiku/Sonnet, measured + graded on the
// same frozen task), scores every card on the three never-blended axes
// (capability / safety / cost), and emits the parity table.
//
// It is the driver for internal/turnbench/parity.go — the headline "viral local
// vs frontier" comparison. Capability is graded against the fixed oracle; safety
// is the fak-arm (kernel-mediated) guarantee with the unmediated baseline kept
// visible; cost is the kernel-counted fak-arm token spend at each model's price.
//
// Usage:
//
//	paritybench \
//	  --local 'fak/experiments/parity/local-*.json' \
//	  --local-gpu 'fak/experiments/parity/remote-*-7b.json' \
//	  --reference-cards fak/experiments/parity/reference-frontier.json \
//	  --reference claude-sonnet \
//	  --require-phase1 \
//	  --out-json fak/experiments/parity/parity.json \
//	  --out-md   fak/experiments/parity/PARITY.md
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

func main() {
	localGlob := flag.String("local", "fak/experiments/parity/local-*.json", "glob of live fak-agent A/B reports for local models")
	localGPUGlob := flag.String("local-gpu", "", "optional glob of live fak-agent A/B reports served from a local GPU/non-CPU endpoint")
	refPath := flag.String("reference-cards", "fak/experiments/parity/reference-frontier.json", "committed frontier reference cards JSON")
	reference := flag.String("reference", "claude-sonnet", "which card is the frontier reference to score against")
	task := flag.String("task", agent.DefaultTask, "task description for the report header")
	outJSON := flag.String("out-json", "fak/experiments/parity/parity.json", "parity report JSON output")
	outMD := flag.String("out-md", "fak/experiments/parity/PARITY.md", "parity report Markdown output")
	requirePhase1 := flag.Bool("require-phase1", false, "fail unless measured 0.5B + 1.5B local rungs and a live local-gpu 7-9B parity card are present")
	flag.Parse()

	cards, err := loadReferenceCards(*refPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load reference cards: %v\n", err)
		os.Exit(1)
	}

	local, err := loadCards(*localGlob, "local-cpu")
	if err != nil {
		fmt.Fprintf(os.Stderr, "load local reports: %v\n", err)
		os.Exit(1)
	}
	cards = append(cards, local...)
	if *localGPUGlob != "" {
		gpu, err := loadCards(*localGPUGlob, "local-gpu")
		if err != nil {
			fmt.Fprintf(os.Stderr, "load local-gpu reports: %v\n", err)
			os.Exit(1)
		}
		cards = append(cards, gpu...)
	}

	if len(cards) == 0 {
		fmt.Fprintln(os.Stderr, "no cards found (no reference + no local reports)")
		os.Exit(1)
	}

	rep := turnbench.BuildParityReport(*task, *reference, cards)

	must(os.WriteFile(*outJSON, rep.JSON(), 0o644))
	must(os.WriteFile(*outMD, []byte(rep.Markdown()), 0o644))

	// Echo the Markdown to stdout — the operator's at-a-glance view.
	fmt.Println(rep.Markdown())
	fmt.Fprintf(os.Stderr, "\nwrote %s and %s (%d cards, %d verdicts)\n",
		*outJSON, *outMD, len(rep.Cards), len(rep.Verdicts))
	if *requirePhase1 {
		gate := turnbench.CheckPhase1CapabilityGate(rep)
		if !gate.Passed {
			fmt.Fprintln(os.Stderr, "phase1 capability gate: FAIL")
			for _, r := range gate.Reasons {
				fmt.Fprintln(os.Stderr, " - "+r)
			}
			os.Exit(4)
		}
		fmt.Fprintf(os.Stderr, "phase1 capability gate: PASS candidate=%s\n", gate.Candidate)
	}
}

// refFile is the on-disk shape of reference-frontier.json (a _README plus cards).
type refFile struct {
	Cards []turnbench.ModelCard `json:"cards"`
}

func loadReferenceCards(path string) ([]turnbench.ModelCard, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // a local-only run is allowed
		}
		return nil, err
	}
	var rf refFile
	if err := json.Unmarshal(b, &rf); err != nil {
		return nil, err
	}
	return rf.Cards, nil
}

var paramRe = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*([bm])`)

// inferParams pulls a "1.5B" / "135M" style size out of a HF model id for display.
func inferParams(model string) string {
	m := paramRe.FindStringSubmatch(model)
	if m == nil {
		return "?"
	}
	return m[1] + strings.ToUpper(m[2])
}

func loadCards(glob, class string) ([]turnbench.ModelCard, error) {
	paths, err := filepath.Glob(glob)
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	out := make([]turnbench.ModelCard, 0, len(paths))
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		var rr agent.RunResult
		if err := json.Unmarshal(b, &rr); err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		// Local models are priced at $0. The caller supplies the execution class so a
		// remote llama-server/GPU report is not collapsed into local-cpu by accident.
		card := turnbench.CardFromRunResult(&rr, class, inferParams(rr.Model), true, 0, 0)
		out = append(out, card)
	}
	return out, nil
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
