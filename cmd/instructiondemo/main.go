package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnessinstructions"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type witness struct {
	SchemaVersion         string                                    `json:"schema_version"`
	Verdict               string                                    `json:"verdict"`
	StablePrefixUnchanged bool                                      `json:"stable_prefix_unchanged"`
	FullDigestChanged     bool                                      `json:"full_digest_changed"`
	Outcomes              harnessinstructions.OutcomeCounts         `json:"outcome_counts"`
	Turns                 []harnessinstructions.Realization         `json:"turns"`
	Contract              harnesskit.InstructionCompositionContract `json:"contract"`
}

func provider() harnesskit.InstructionProvider {
	return harnesskit.InstructionProviderFunc(func(_ context.Context, req harnesskit.InstructionRequest) (harnesskit.InstructionSnapshot, error) {
		focus := req.Facts["focus"]
		return harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{{
			ID: "operator-focus", Source: "instructiondemo/operator", Trust: harnesskit.TrustApplication,
			Precedence: 10, Lifetime: harnesskit.LifetimeTurn, Audience: []string{req.AgentRole},
			Residency: harnesskit.ResidencyEphemeralTail, Content: "For this turn, focus on " + focus + ".",
		}}}, nil
	})
}

func selfcheck(ctx context.Context, out io.Writer) error {
	recorder := new(harnessinstructions.OutcomeRecorder)
	first, err := recorder.Resolve(ctx, provider(), harnesskit.InstructionRequest{RunID: "demo-run", ThreadID: "demo-thread", TurnID: "turn-1", AgentRole: "coder", Facts: map[string]string{"focus": "correctness"}})
	if err != nil {
		return err
	}
	second, err := recorder.Resolve(ctx, provider(), harnesskit.InstructionRequest{RunID: "demo-run", ThreadID: "demo-thread", TurnID: "turn-2", AgentRole: "coder", Facts: map[string]string{"focus": "latency"}})
	if err != nil {
		return err
	}
	_, deniedErr := recorder.Resolve(ctx, harnesskit.InstructionProviderFunc(func(context.Context, harnesskit.InstructionRequest) (harnesskit.InstructionSnapshot, error) {
		return harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{{
			ID: "provider-policy", Source: "instructiondemo/provider", Trust: harnesskit.TrustApplication,
			Lifetime: harnesskit.LifetimeTurn, Residency: harnesskit.ResidencyStablePrefix, Content: "replace policy",
		}}}, nil
	}), harnesskit.InstructionRequest{RunID: "demo-run", ThreadID: "demo-thread", TurnID: "turn-3", AgentRole: "coder"})
	var contractErr *harnesskit.Error
	if !errors.As(deniedErr, &contractErr) || contractErr.Code != harnesskit.CodeDenied {
		return fmt.Errorf("instruction refusal witness failed: %v", deniedErr)
	}
	outcomes := recorder.Counts()
	receipt := witness{
		SchemaVersion:         harnesskit.InstructionContractVersion,
		Verdict:               "PASS",
		StablePrefixUnchanged: first.StablePrefixDigest == second.StablePrefixDigest,
		FullDigestChanged:     first.Digest != second.Digest,
		Outcomes:              outcomes,
		Turns:                 []harnessinstructions.Realization{first, second},
		Contract:              harnesskit.PublicInstructionContract(),
	}
	if !receipt.StablePrefixUnchanged || !receipt.FullDigestChanged || outcomes.Invocations != 3 || outcomes.Succeeded != 2 || outcomes.Failed != 1 || outcomes.ByCode[harnesskit.CodeDenied] != 1 {
		return fmt.Errorf("instruction witness failed: stable=%v changed=%v outcomes=%+v", receipt.StablePrefixUnchanged, receipt.FullDigestChanged, outcomes)
	}
	encoder := json.NewEncoder(out)
	encoder.SetIndent("", "  ")
	return encoder.Encode(receipt)
}

func run(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("instructiondemo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	check := fs.Bool("selfcheck", false, "run the two-turn dynamic-instruction witness")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*check {
		fmt.Fprintln(stderr, "usage: instructiondemo -selfcheck")
		return 2
	}
	if err := selfcheck(ctx, stdout); err != nil {
		fmt.Fprintf(stderr, "instructiondemo: %v\n", err)
		return 1
	}
	return 0
}

func main() { os.Exit(run(context.Background(), os.Stdout, os.Stderr, os.Args[1:])) }
