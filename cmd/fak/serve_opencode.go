package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

func runServeOpenCodeConfig(sf *serveFlags, out io.Writer, write bool) {
	addr := *sf.addr
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	baseURL := addr
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		baseURL = "http://" + baseURL
	}
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL = strings.TrimSuffix(baseURL, "/") + "/v1"
	}
	modelID := *sf.model
	if modelID == "" {
		modelID = projectassets.DefaultOpenCodeModelID
	}
	if write {
		modified, err := projectassets.EnsureOpenCodeProviderConfig(".", baseURL, modelID)
		if err != nil {
			fmt.Fprintf(out, "fak serve: update opencode.json: %v\n", err)
			os.Exit(1)
		}
		if modified {
			fmt.Fprintf(out, "fak serve: updated opencode.json with provider \"fak\" (%s, model: %s)\n", baseURL, modelID)
		} else {
			fmt.Fprintf(out, "fak serve: opencode.json is already configured with provider \"fak\"\n")
		}
		return
	}
	raw, err := projectassets.GenerateOpenCodeConfig(baseURL, modelID)
	if err != nil {
		fmt.Fprintf(out, "fak serve: generate opencode.json: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(out, string(raw))
}
