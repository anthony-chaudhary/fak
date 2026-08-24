package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/signals"
)

// cmdSignals handles `fak signals <subcommand>` — plain-English BEHAVIORAL signals
// (an NL prompt + a verdict schema + a sample rate) judged over an agent's turns. The
// behavioral complement to the structural anti-pattern detectors. See internal/signals.
func cmdSignals(args []string) {
	if len(args) == 0 {
		signalsUsage()
		os.Exit(2)
	}
	switch args[0] {
	case "validate":
		cmdSignalsValidate(args[1:])
	case "plan":
		cmdSignalsPlan(args[1:])
	case "-h", "--help", "help":
		signalsUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak signals: unknown subcommand %q\n", args[0])
		signalsUsage()
		os.Exit(2)
	}
}

func signalsUsage() {
	fmt.Fprintln(os.Stderr, "usage: fak signals <validate|plan> --config <signals.json> [--items <items.jsonl>]")
	fmt.Fprintln(os.Stderr, "       fak signals validate --config signals.json          (check signals are well-formed)")
	fmt.Fprintln(os.Stderr, "       fak signals plan --config signals.json --items items.jsonl")
	fmt.Fprintln(os.Stderr, "            (dry run: which items each signal samples + the rendered judge prompt; no model call)")
}

func loadSignalsConfig(path string) signals.Config {
	return loadJSONFileOrExit[signals.Config](path, "fak signals")
}

// loadItems reads a JSONL of signals.Item (one per line). Empty path => no items.
func loadItems(path string) []signals.Item {
	return readJSONLCorpus[signals.Item](path, "fak signals")
}

func cmdSignalsValidate(args []string) {
	fs := flag.NewFlagSet("signals validate", flag.ExitOnError)
	config := fs.String("config", "signals.json", "signals config (JSON)")
	_ = fs.Parse(args)
	cfg := loadSignalsConfig(*config)
	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "invalid: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("ok: %d signal(s) valid\n", len(cfg.Signals))
}

func cmdSignalsPlan(args []string) {
	fs := flag.NewFlagSet("signals plan", flag.ExitOnError)
	config := fs.String("config", "signals.json", "signals config (JSON)")
	itemsPath := fs.String("items", "", "items JSONL to preview sampling over")
	showPrompts := fs.Bool("prompts", false, "include the rendered judge prompt per sampled item")
	_ = fs.Parse(args)
	cfg := loadSignalsConfig(*config)
	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "invalid: %v\n", err)
		os.Exit(1)
	}
	items := loadItems(*itemsPath)

	type sampledItem struct {
		ItemID string `json:"item_id"`
		Prompt string `json:"prompt,omitempty"`
	}
	type signalPlan struct {
		Signal      string        `json:"signal"`
		SampleRate  float64       `json:"sample_rate"`
		TotalItems  int           `json:"total_items"`
		SampledN    int           `json:"sampled_count"`
		SampledList []sampledItem `json:"sampled,omitempty"`
	}
	var plans []signalPlan
	for _, sig := range cfg.Signals {
		p := signalPlan{Signal: sig.Name, SampleRate: sig.SampleRate, TotalItems: len(items)}
		for _, it := range items {
			if !sig.Sampled(it.ID) {
				continue
			}
			p.SampledN++
			si := sampledItem{ItemID: it.ID}
			if *showPrompts {
				si.Prompt = signals.RenderPrompt(sig, it)
			}
			p.SampledList = append(p.SampledList, si)
		}
		plans = append(plans, p)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(plans)
}
