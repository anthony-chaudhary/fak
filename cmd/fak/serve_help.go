package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

type serveHelpCategory struct {
	name    string
	summary string
	flags   []string
}

// serveHelpCategories is deliberately exhaustive: a new serve flag must be placed
// where an operator will look before the help-contract test permits it to land.
var serveHelpCategories = []serveHelpCategory{
	{name: "start", summary: "Listener, deployment manifest, and provider connection.", flags: strings.Fields("addr stdio config print-effective-config provider base-url model api-key-env require-key-env stream-progress-timeout expose key-principal unsafe-allow-unauthenticated-bind")},
	{name: "model", summary: "Local engines, weights, parallelism, and hardware.", flags: strings.Fields("engine backend gguf tokenizer qwen38-runtime llama-server llama-startup-timeout cuda-graph cpu-offload-experts n-cpu-moe metal expert-parallel tensor-parallel replica-base-url vulkan-q4k-profile vulkan-stage-q4k")},
	{name: "cache", summary: "Shared engine-cache attachment and cache correctness.", flags: strings.Fields("engine-cache-engine engine-cache-base-url engine-cache-admin-key-env engine-cache-idle-timeout engine-cache-require-exact-span")},
	{name: "context", summary: "Context budgets, compaction, elision, and reuse.", flags: strings.Fields("ctx-view-budget compact-history-budget positive-residual-substitution compact-anchor-head assume-session-turns elide-result-bytes elide-stale-reads vcache-anchor defer-cold-tools context-budget-tokens reset-on-budget")},
	{name: "policy", summary: "Policy, routing, plans, and invalidation.", flags: strings.Fields("policy policy-canary-turns policy-check plan-json vdso invalidation route-manifest route-accounts")},
	{name: "session", summary: "Session identity, persistence, and spend controls.", flags: strings.Fields("session-id session-state session-registry budget-webhook budget-warn-fraction spend-cap spend-scope-trace")},
	{name: "native", summary: "Owned agent loop, coding tools, and speculation.", flags: strings.Fields("native native-qwen-q4k-prefill-chunk-tokens native-qwen35-metal-gdn-sequence native-q4k-gateup-slab native-prefix-profile native-max-turns native-admission-token-budget native-code-tools native-code-workspace native-speculate vdso-proxy-fill")},
	{name: "observe", summary: "Notifications, metrics, debug stats, and dojo mode.", flags: strings.Fields("notify-native notify-webhook notify-slack otlp-traces-endpoint debug-stats metrics-snapshot dojo")},
	{name: "fleet", summary: "Fleet control-bus membership and identity.", flags: strings.Fields("fleet-bus fleet-bus-dir fleet-bus-id fleet-bus-interval")},
}

func configureServeHelp(fs *flag.FlagSet) {
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { printServeHelp(os.Stderr, fs, "") }
}

// serveHelpRequested handles the discoverable help-only subcommands while leaving
// the long-standing `fak serve [flags]` start path unchanged.
func serveHelpRequested(fs *flag.FlagSet, argv []string) bool {
	category, ok := serveHelpTopic(argv)
	if !ok {
		return false
	}
	printServeHelp(os.Stdout, fs, category)
	return true
}

func serveHelpTopic(argv []string) (string, bool) {
	if len(argv) == 0 {
		return "", false
	}
	if argv[0] == "-h" || argv[0] == "--help" {
		return "", true
	}
	if argv[0] == "help" {
		if len(argv) == 1 {
			return "", true
		}
		return argv[1], true
	}
	if len(argv) == 2 && (argv[1] == "-h" || argv[1] == "--help") && serveHelpCategoryNamed(argv[0]) {
		return argv[0], true
	}
	return "", false
}

func serveHelpCategoryNamed(name string) bool {
	if name == "all" {
		return true
	}
	for _, category := range serveHelpCategories {
		if category.name == name {
			return true
		}
	}
	return false
}

func printServeHelp(w io.Writer, fs *flag.FlagSet, topic string) {
	if topic == "all" {
		fmt.Fprintln(w, "Usage: fak serve [flags]")
		fmt.Fprintln(w, "\nAll flags (detailed):")
		old := fs.Output()
		fs.SetOutput(w)
		fs.PrintDefaults()
		fs.SetOutput(old)
		return
	}
	if topic != "" {
		for _, category := range serveHelpCategories {
			if category.name == topic {
				fmt.Fprintf(w, "Usage: fak serve [flags]\n\n%s: %s\n", category.name, category.summary)
				printServeCategoryFlags(w, fs, category)
				fmt.Fprintln(w, "\nDetailed flag text: fak serve help all")
				return
			}
		}
		fmt.Fprintf(w, "Unknown serve help category %q.\n\n", topic)
	}
	fmt.Fprintln(w, "Usage: fak serve [flags]")
	fmt.Fprintln(w, "       fak serve help <category>")
	fmt.Fprintln(w, "       fak serve <category> --help")
	fmt.Fprintln(w, "\nStart the model gateway. The default listener serves a live dashboard at http://127.0.0.1:8080/; Rich dashboards start on first click. Help is grouped by operator task:")
	for _, category := range serveHelpCategories {
		fmt.Fprintf(w, "  %-9s %s\n", category.name, category.summary)
	}
	fmt.Fprintln(w, "  all       Every flag with detailed explanations.")
	fmt.Fprintln(w, "\nExamples:")
	fmt.Fprintln(w, "  fak serve help native")
	fmt.Fprintln(w, "  fak serve context --help")
	fmt.Fprintln(w, "  fak serve help all")
}

func printServeCategoryFlags(w io.Writer, fs *flag.FlagSet, category serveHelpCategory) {
	names := append([]string(nil), category.flags...)
	sort.Strings(names)
	for _, name := range names {
		f := fs.Lookup(name)
		if f == nil {
			continue
		}
		arg, usage := flag.UnquoteUsage(f)
		if arg == "" && !isBoolFlag(f) {
			arg = "value"
		}
		if arg != "" {
			arg = " " + arg
		}
		fmt.Fprintf(w, "  --%s%s\n      %s\n", name, arg, conciseFlagSummary(usage))
	}
}
