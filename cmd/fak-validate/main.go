// fak-validate exposes repository validation without linking the full fak CLI.
package main

import (
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
	"github.com/anthony-chaudhary/fak/internal/validate"
)

func main() {
	if code, handled := runStrixKnownHostsBrokerEarly(os.Stdout, os.Stderr, os.Args[1:], amdgpu.RunStrixKnownHostsBrokerChild); handled {
		os.Exit(code)
	}
	os.Exit(validate.Run(os.Stdout, os.Stderr, os.Args[1:]))
}

func runStrixKnownHostsBrokerEarly(stdout, stderr io.Writer, argv []string, child func(string, string, io.Writer) error) (int, bool) {
	if len(argv) == 0 || argv[0] != amdgpu.StrixKnownHostsOperand {
		return 0, false
	}
	if len(argv) != 3 || argv[1] == "" || argv[2] == "" {
		fmt.Fprintln(stderr, "STRIX_HOST_TRUST_REFUSED")
		return 2, true
	}
	var entry bytes.Buffer
	if err := child(argv[1], argv[2], &entry); err != nil {
		fmt.Fprintln(stderr, "STRIX_HOST_TRUST_REFUSED")
		return 1, true
	}
	if _, err := io.Copy(stdout, &entry); err != nil {
		fmt.Fprintln(stderr, "STRIX_HOST_TRUST_REFUSED")
		return 1, true
	}
	return 0, true
}
