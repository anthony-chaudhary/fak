package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/fastintent"
	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func runUltracodeFastProfile(stdout, stderr io.Writer, pairPath string) int {
	if pairPath == "" {
		fmt.Fprintln(stderr, "fak ultracode bench: --scenario fast-profile requires --pair PATH")
		return 2
	}
	b, err := os.ReadFile(pairPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak ultracode bench: read pair: %v\n", err)
		return 1
	}
	var replay fastintent.ReplayBundle
	if err := json.Unmarshal(b, &replay); err == nil && replay.Schema == fastintent.Schema {
		return writeUltracodeBenchJSON(stdout, stderr, replay.Evaluation)
	}
	var bundle ultracodebench.FastProfileBundle
	if err := json.Unmarshal(b, &bundle); err != nil {
		fmt.Fprintf(stderr, "fak ultracode bench: decode fast-profile: %v\n", err)
		return 1
	}
	return writeUltracodeBenchJSON(stdout, stderr, ultracodebench.EvaluateFastProfile(bundle))
}

func runUltracodeAccessFrontier(stdout, stderr io.Writer, scenario, inputPath, widthsText string) int {
	if scenario != "access-frontier" {
		fmt.Fprintf(stderr, "fak ultracode bench: unknown scenario %q\n", scenario)
		return 2
	}
	widths, err := parseUltracodeWidths(widthsText)
	if err != nil {
		fmt.Fprintf(stderr, "fak ultracode bench: %v\n", err)
		return 2
	}
	if inputPath == "" {
		report, err := ultracodebench.EvaluateAccessModeFrontier(ultracodebench.AccessModeFrontierFixture(), widths)
		if err != nil {
			fmt.Fprintf(stderr, "fak ultracode bench: evaluate access-frontier: %v\n", err)
			return 1
		}
		return writeUltracodeBenchJSON(stdout, stderr, report)
	}
	data, err := os.ReadFile(inputPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak ultracode bench: read scenario input: %v\n", err)
		return 1
	}
	var header struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		fmt.Fprintf(stderr, "fak ultracode bench: decode scenario input: %v\n", err)
		return 1
	}
	var report any
	switch header.Schema {
	case ultracodebench.AccessModeFrontierSchema:
		var frontier ultracodebench.AccessModeFrontier
		if err := json.Unmarshal(data, &frontier); err != nil {
			fmt.Fprintf(stderr, "fak ultracode bench: decode scenario input: %v\n", err)
			return 1
		}
		report, err = ultracodebench.EvaluateAccessModeFrontier(frontier, widths)
	case ultracodebench.AccessFrontierSchema:
		var frontier ultracodebench.AccessFrontier
		if err := json.Unmarshal(data, &frontier); err != nil {
			fmt.Fprintf(stderr, "fak ultracode bench: decode scenario input: %v\n", err)
			return 1
		}
		report, err = ultracodebench.EvaluateAccessFrontier(frontier, widths)
	default:
		fmt.Fprintf(stderr, "fak ultracode bench: unsupported access-frontier schema %q\n", header.Schema)
		return 1
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak ultracode bench: evaluate access-frontier: %v\n", err)
		return 1
	}
	return writeUltracodeBenchJSON(stdout, stderr, report)
}

func runUltracodeFactorial(stdout, stderr io.Writer, path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "read factorial campaign: %v\n", err)
		return 1
	}
	var campaign ultracodebench.FactorialCampaign
	if err := json.Unmarshal(data, &campaign); err != nil {
		fmt.Fprintf(stderr, "parse factorial campaign: %v\n", err)
		return 1
	}
	report, err := ultracodebench.EvaluateFactorialCampaign(campaign, ultracodebench.FactorialWidths(campaign))
	if err != nil {
		fmt.Fprintf(stderr, "evaluate factorial campaign: %v\n", err)
		return 1
	}
	return writeUltracodeBenchJSON(stdout, stderr, report)
}

func runUltracodeNetWork(stdout, stderr io.Writer, path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "read net-work campaign: %v\n", err)
		return 1
	}
	var campaign ultracodebench.NetWorkCampaign
	if err := json.Unmarshal(data, &campaign); err != nil {
		fmt.Fprintf(stderr, "parse net-work campaign: %v\n", err)
		return 1
	}
	report, err := ultracodebench.EvaluateNetWorkCampaign(campaign)
	if err != nil {
		fmt.Fprintf(stderr, "evaluate net-work campaign: %v\n", err)
		return 1
	}
	return writeUltracodeBenchJSON(stdout, stderr, report)
}
