package main

import (
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

func runServePiConfig(sf *serveFlags, out io.Writer, write bool) {
	addr := ""
	if sf.addr != nil {
		addr = *sf.addr
	}
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	baseURL := projectassets.NormalizePiBaseURL(addr)
	modelID := ""
	if sf.model != nil {
		modelID = *sf.model
	}
	if modelID == "" || modelID == "mock" {
		if sf.ggufPath != nil && *sf.ggufPath != "" && *sf.ggufPath != "default" {
			modelID = *sf.ggufPath
		} else {
			modelID = projectassets.DefaultPiModelID
		}
	}
	targetPath := ""
	if sf.piConfigPath != nil && *sf.piConfigPath != "" {
		targetPath = *sf.piConfigPath
	}
	if write {
		path, modified, err := projectassets.EnsurePiProviderConfig(targetPath, baseURL, modelID)
		if err != nil {
			fmt.Fprintf(out, "fak serve: update %s: %v\n", path, err)
			os.Exit(1)
		}
		if modified {
			fmt.Fprintf(out, "fak serve: updated %s with provider \"fak\" (%s, model: %s)\n", path, baseURL, modelID)
		} else {
			fmt.Fprintf(out, "fak serve: %s is already configured with provider \"fak\" (%s, model: %s)\n", path, baseURL, modelID)
		}
		return
	}
	raw, err := projectassets.GeneratePiConfig(baseURL, modelID)
	if err != nil {
		fmt.Fprintf(out, "fak serve: generate models.json: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(out, string(raw))
}
