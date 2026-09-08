// Command qwen38campaign runs the frozen Qwen3.8 evidence campaign.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

func main() { os.Exit(run(os.Stdout, os.Stderr, os.Args[1:])) }

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("qwen38campaign", flag.ContinueOnError)
	fs.SetOutput(stderr)
	config := fs.String("config", "", "campaign adapter JSON")
	corpus := fs.String("corpus", "docs/benchmarks/qwen38-quant/corpus.json", "frozen corpus JSON")
	report := fs.String("report", "", "validator-clean report output")
	archive := fs.String("archive", "", "secret-scrubbed raw archive output")
	verifyPromptPacket := fs.String("verify-prompt-packet", "", "read and verify a v2 prompt-token packet without executing a trial")
	soak := fs.Bool("soak", false, "run the three-finalist production soak")
	oracle := fs.Bool("oracle", false, "compare pinned llama.cpp and fak evidence")
	scoreboard := fs.Bool("amd-scoreboard", false, "build a matched AMD fak-native versus llama.cpp scoreboard")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	verifyPromptPacketSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "verify-prompt-packet" {
			verifyPromptPacketSet = true
		}
	})
	if verifyPromptPacketSet {
		if *verifyPromptPacket == "" || *config != "" || *report != "" || *archive != "" || *soak || *oracle || *scoreboard || fs.NArg() != 0 {
			fmt.Fprintln(stderr, "usage: qwen38campaign --verify-prompt-packet PACKET.json")
			return 2
		}
		packet, err := qwen38quantrun.ReadPromptPacketFile(*verifyPromptPacket)
		if err != nil {
			fmt.Fprintf(stderr, "qwen38campaign: prompt packet: %v\n", err)
			return 1
		}
		if err := qwen38quantrun.VerifyPromptPacket(packet); err != nil {
			fmt.Fprintf(stderr, "qwen38campaign: prompt packet: %v\n", err)
			return 1
		}
		if packet.Schema != qwen38quantrun.PromptTokenPacketSchema {
			fmt.Fprintf(stderr, "qwen38campaign: prompt packet schema %q is readable but not eligible; require %q\n", packet.Schema, qwen38quantrun.PromptTokenPacketSchema)
			return 1
		}
		fmt.Fprintf(stdout, "PASS prompt_packet=%s digest=%s\n", *verifyPromptPacket, packet.PacketDigest)
		return 0
	}
	if *config == "" || *report == "" || fs.NArg() != 0 || boolCount(*soak, *oracle, *scoreboard) > 1 || (!*scoreboard && *archive == "") {
		fmt.Fprintln(stderr, "usage: qwen38campaign [--soak | --oracle | --amd-scoreboard] --config CONFIG.json --report REPORT.json [--archive ARCHIVE.json] [--corpus CORPUS.json] | --verify-prompt-packet PACKET.json")
		return 2
	}
	var err error
	verdict := "PASS"
	if *scoreboard {
		var scoreboardReport qwen38quantrun.AMDScoreboardReport
		scoreboardReport, err = qwen38quantrun.BuildAMDScoreboardFile(*config, *report)
		verdict = scoreboardReport.Verdict
	} else if *oracle {
		var oracleReport qwen38quantrun.OracleReport
		oracleReport, err = qwen38quantrun.RunOracle(context.Background(), *config, *corpus, *report, *archive)
		verdict = oracleReport.Verdict
	} else if *soak {
		err = qwen38quantrun.RunSoakAdapter(context.Background(), *config, *corpus, *report, *archive)
	} else {
		err = qwen38quantrun.RunAdapter(context.Background(), *config, *corpus, *report, *archive)
	}
	if err != nil {
		fmt.Fprintf(stderr, "qwen38campaign: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "%s report=%s archive=%s\n", verdict, *report, *archive)
	return 0
}

func boolCount(values ...bool) int {
	count := 0
	for _, value := range values {
		if value {
			count++
		}
	}
	return count
}
