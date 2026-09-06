package main

import (
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/pkg/conversationprofile"
)

func cmdConversationProfile(argv []string) {
	if code := runConversationProfile(os.Stdout, os.Stderr, argv); code != 0 {
		os.Exit(code)
	}
}

func runConversationProfile(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "help" || argv[0] == "-h" || argv[0] == "--help" {
		fmt.Fprintln(stdout, "Usage: fak conversation-profile validate <file>")
		return 0
	}

	switch argv[0] {
	case "validate":
		if len(argv) < 2 {
			fmt.Fprintln(stderr, "Usage: fak conversation-profile validate <file>")
			return 2
		}
		data, err := os.ReadFile(argv[1])
		if err != nil {
			fmt.Fprintf(stderr, "error reading file: %v\n", err)
			return 1
		}
		_, err = conversationprofile.Parse(data)
		if err != nil {
			fmt.Fprintf(stderr, "validation error: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, "OK")
		return 0
	default:
		fmt.Fprintf(stderr, "fak conversation-profile: unknown subcommand %q\n", argv[0])
		fmt.Fprintln(stderr, "Usage: fak conversation-profile validate <file>")
		return 2
	}
}
