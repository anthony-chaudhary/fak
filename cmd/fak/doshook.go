package main

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/doshook"
)

func envFromOS() map[string]string {
	out := make(map[string]string)
	for _, kv := range os.Environ() {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) == 2 {
			out[parts[0]] = parts[1]
		}
	}
	return out
}

func cmdDosHook(argv []string) {
	doshook.Run(argv, os.Stdin, os.Stdout, os.Stderr, envFromOS(), nil)
	os.Exit(0)
}
