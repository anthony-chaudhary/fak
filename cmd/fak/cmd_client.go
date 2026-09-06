package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

func cmdClient(argv []string) {
	if code := runClient(os.Stdout, os.Stderr, argv); code != 0 {
		os.Exit(code)
	}
}

func runClient(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "help" || argv[0] == "-h" || argv[0] == "--help" {
		fmt.Fprintln(stdout, "Usage:")
		fmt.Fprintln(stdout, "  fak client health [--url <baseURL>]")
		fmt.Fprintln(stdout, "  fak client ping [--url <baseURL>]")
		return 0
	}

	switch argv[0] {
	case "health", "ping":
		fs := flag.NewFlagSet("client "+argv[0], flag.ContinueOnError)
		fs.SetOutput(stderr)
		url := fs.String("url", "http://127.0.0.1:8080", "fak gateway base URL")
		if err := fs.Parse(argv[1:]); err != nil {
			return 2
		}
		c := fakclient.New(*url)
		if err := c.Health(context.Background()); err != nil {
			fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, "OK")
		return 0
	default:
		fmt.Fprintf(stderr, "fak client: unknown subcommand %q\n", argv[0])
		fmt.Fprintln(stderr, "Usage:")
		fmt.Fprintln(stderr, "  fak client health [--url <baseURL>]")
		fmt.Fprintln(stderr, "  fak client ping [--url <baseURL>]")
		return 2
	}
}
