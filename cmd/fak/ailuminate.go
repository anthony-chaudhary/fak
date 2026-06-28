package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ailuminate"
	"github.com/anthony-chaudhary/fak/internal/dispatchpost"
)

// cmdAILuminate dispatches the AILuminate benchmark-entry scoping commands.
// AILuminate is a Tier-3, adapter-gated lane in the #1063 benchmark-entry
// portfolio: it scores a model+guardrail "AI system" SUT, but its content-harm
// grade rides on the fronted model, not on fak. The only subcommand today emits
// the fenced go/no-go scoping contract (#1070).
func cmdAILuminate(argv []string) {
	if len(argv) == 0 {
		ailuminateUsage()
		return
	}
	switch argv[0] {
	case "contract":
		cmdAILuminateContract(argv[1:])
	case "-h", "--help", "help":
		ailuminateUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak ailuminate: unknown subcommand %q\n\n", argv[0])
		ailuminateUsage()
		os.Exit(2)
	}
}

func ailuminateUsage() {
	fmt.Fprint(os.Stderr, `fak ailuminate — MLCommons AILuminate benchmark-entry scoping (#1070)

USAGE:
  fak ailuminate contract [--content-filter] [--fak-commit SHA] [--out FILE] [--md FILE]

        Emit the AILuminate model+guardrail SUT scoping + go/no-go contract.
        It maps fak's gateway path against AILuminate v1.1's 12 hazard categories,
        resolves the prerequisite (does fak inspect the completion/content path?),
        and fences the artifact: the AILuminate grade is the AI-system's content
        behavior graded by MLCommons' ensemble (OBSERVED), never a fak score.

        --content-filter   assert a content-moderation filter IS wired into fak's
                           completion path (flips the gate to OK / GO). Default off:
                           today fak adjudicates tool calls, not completions, so the
                           default verdict is NO-GO / scoped-out.
        --fak-commit SHA   lineage fak commit (default: detected git HEAD).
        --out FILE         write the contract JSON here (default stdout).
        --md FILE          write the contract markdown here.
`)
}

func cmdAILuminateContract(argv []string) {
	fs := flag.NewFlagSet("ailuminate contract", flag.ExitOnError)
	contentFilter := fs.Bool("content-filter", false, "assert fak's gateway inspects the model completion/content path for content-harm (default: false — tool calls only)")
	fakCommit := fs.String("fak-commit", "", "lineage fak commit SHA (default: detected git HEAD)")
	frontedModel := fs.String("fronted-model", "", "fronted chat model id for lineage (placeholder until a real run)")
	modelProvider := fs.String("model-provider", "", "fronted model provider for lineage")
	modelDate := fs.String("model-date", "", "fronted model release/snapshot date for lineage")
	harnessCommit := fs.String("harness-commit", "", "ModelGauge/ModelBench harness commit for lineage")
	out := fs.String("out", "", "write the contract JSON here (default stdout)")
	md := fs.String("md", "", "write the contract markdown here")
	_ = fs.Parse(argv)

	commit := *fakCommit
	if commit == "" {
		commit = dispatchpost.HeadSHA(ctx(), "")
	}

	contract := ailuminate.Build(ailuminate.ContractInput{
		GeneratedAt:                 time.Now().UTC().Format(time.RFC3339),
		FakCommit:                   commit,
		CompletionPathContentFilter: *contentFilter,
		FrontedModelID:              *frontedModel,
		ModelProvider:               *modelProvider,
		ModelDate:                   *modelDate,
		HarnessCommit:               *harnessCommit,
	})

	if *out != "" {
		must(os.WriteFile(*out, jsonIndent(contract), 0o644))
	} else {
		fmt.Println(string(jsonIndent(contract)))
	}
	if *md != "" {
		must(os.WriteFile(*md, []byte(ailuminate.RenderMarkdown(contract)), 0o644))
	}

	fmt.Fprintf(os.Stderr, "\n== fak ailuminate contract ==\n")
	fmt.Fprintf(os.Stderr, "status         : %s\n", contract.Status)
	fmt.Fprintf(os.Stderr, "recommendation : %s\n", contract.Recommendation)
	fmt.Fprintf(os.Stderr, "prerequisite   : %s\n", contract.Prerequisite.Verdict)
	fmt.Fprintf(os.Stderr, "result claim   : %t (the AILuminate grade is OBSERVED, never a fak score)\n", contract.ResultClaimAllowed)
	if *out != "" {
		fmt.Fprintf(os.Stderr, "json           : %s\n", *out)
	}
	if *md != "" {
		fmt.Fprintf(os.Stderr, "markdown       : %s\n", *md)
	}
}
