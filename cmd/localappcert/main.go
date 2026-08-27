// localappcert validates a captured Mac local-app certification matrix.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/localappcert"
)

func main() {
	matrix := flag.String("matrix", "", "captured fak.local-app-certification/1 JSON")
	asJSON := flag.Bool("json", false, "emit a machine-readable verdict")
	flag.Parse()
	if *matrix == "" {
		fmt.Fprintln(os.Stderr, "localappcert: --matrix is required")
		os.Exit(2)
	}
	m, err := localappcert.Load(*matrix)
	if err == nil {
		err = localappcert.Validate(m)
	}
	if *asJSON {
		verdict := map[string]any{"schema": "fak.local-app-certification-verdict/1", "ok": err == nil, "matrix": *matrix}
		if err != nil {
			verdict["error"] = err.Error()
		}
		_ = json.NewEncoder(os.Stdout).Encode(verdict)
	} else if err == nil {
		fmt.Println("localappcert: PASS")
	} else {
		fmt.Fprintln(os.Stderr, err)
	}
	if err != nil {
		os.Exit(1)
	}
}
