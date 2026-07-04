package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macbench"
)

func cmdMacBench(argv []string) { os.Exit(runMacBench(os.Stdout, os.Stderr, argv)) }

func runMacBench(stdout, stderr io.Writer, argv []string) int {
	suite := macbench.SuiteAll
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		suite = macbench.Suite(argv[0])
		argv = argv[1:]
	}
	def := macbench.DefaultOptions()
	fs := flag.NewFlagSet("macbench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gateway := fs.String("gateway", envOrDefault("FAK_MAC_GATEWAY", def.Gateway), "fak serve gateway on the Mac; defaults to loopback for on-node runs")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", def.Model), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	keyFile := fs.String("gateway-key-file", "~/.fak-gateway-key", "file holding the gateway bearer when the env var is empty; empty disables file lookup")
	timeout := fs.Duration("timeout", 2*time.Hour, "overall benchmark timeout")
	decodeTokens := fs.String("decode-tokens", "256,512", "comma-separated max_tokens for decode-longgen")
	prefillTokens := fs.String("prefill-tokens", "128,512,2048,4096", "comma-separated prompt-token targets for prefill-sweep")
	concurrency := fs.Int("concurrency", 2, "concurrent requests for the 2stream suite")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	key, err := resolveMacBenchKey(*keyEnv, *keyFile)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: %v\n", err)
		return 2
	}
	dec, err := parseIntCSV(*decodeTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: --decode-tokens: %v\n", err)
		return 2
	}
	pre, err := parseIntCSV(*prefillTokens)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: --prefill-tokens: %v\n", err)
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak macbench: --timeout must be positive")
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	rep, err := macbench.Run(ctx, macbench.Options{
		Gateway:       *gateway,
		Model:         *model,
		Key:           key,
		Suite:         suite,
		DecodeTokens:  dec,
		PrefillTokens: pre,
		Concurrency:   *concurrency,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench: %v\n", err)
		return 1
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, rep)
	} else {
		renderMacBench(stdout, rep)
	}
	if rep.HasErrors() {
		return 1
	}
	return 0
}

func renderMacBench(w io.Writer, rep macbench.Report) {
	fmt.Fprintf(w, "macbench %s gateway=%s model=%s health=%v\n", rep.Suite, rep.Gateway, rep.Model, rep.Health.OK)
	for _, row := range rep.Rows {
		if row.Error != "" {
			fmt.Fprintf(w, "- %s: ERROR %s\n", row.Name, row.Error)
			continue
		}
		switch {
		case row.PrefillTokensPerSecond > 0:
			fmt.Fprintf(w, "- %s: %.2f tok/s prefill (prompt=%d ttft=%.3fs completion=%d)\n",
				row.Name, row.PrefillTokensPerSecond, row.PromptTokens, row.TTFTSeconds, row.CompletionTokens)
		case row.TokensPerSecond > 0:
			fmt.Fprintf(w, "- %s: %.2f tok/s decode (completion=%d wall=%.3fs streams=%d)\n",
				row.Name, row.TokensPerSecond, row.CompletionTokens, row.WallSeconds, row.Streams)
		default:
			fmt.Fprintf(w, "- %s: no metric\n", row.Name)
		}
	}
	if rep.Headline != "" {
		fmt.Fprintf(w, "headline: %s\n", rep.Headline)
	}
	for _, err := range rep.Errors {
		fmt.Fprintf(w, "error: %s\n", err)
	}
}

func resolveMacBenchKey(envName, keyFile string) (string, error) {
	envName = strings.TrimSpace(envName)
	if envName == "" {
		envName = "FAK_GATEWAY_KEY"
	}
	if key := strings.TrimSpace(os.Getenv(envName)); key != "" {
		return key, nil
	}
	keyFile = strings.TrimSpace(keyFile)
	if keyFile == "" {
		return "", nil
	}
	if strings.HasPrefix(keyFile, "~/") || strings.HasPrefix(keyFile, `~\`) {
		if home, err := os.UserHomeDir(); err == nil {
			keyFile = filepath.Join(home, keyFile[2:])
		}
	}
	b, err := os.ReadFile(keyFile)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read --gateway-key-file: %w", err)
	}
	return strings.TrimSpace(string(b)), nil
}

func parseIntCSV(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("%q is not a positive integer", part)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one value is required")
	}
	return out, nil
}
