package main

import (
	"encoding/base64"
	"flag"
	"fmt"
	"io"
	"os"
)

func main() {
	contract64 := flag.String("contract-base64", "", "base64-encoded contract-preserving output")
	flag.Parse()
	if *contract64 == "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: crossauditfixture --contract-base64 VALUE < candidate")
		os.Exit(2)
	}
	contract, err := base64.StdEncoding.DecodeString(*contract64)
	if err != nil {
		fmt.Fprintln(os.Stderr, "invalid contract encoding")
		os.Exit(2)
	}
	candidate, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read candidate:", err)
		os.Exit(2)
	}
	if string(candidate) == string(contract) {
		fmt.Print("PASS")
		return
	}
	fmt.Print("FAIL")
	os.Exit(1)
}
