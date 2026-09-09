package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

func runServeCodexConfig(sf *serveFlags, out io.Writer, write bool) {
	addr := ""
	if sf.addr != nil {
		addr = *sf.addr
	}
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

	modelID := ""
	if sf.model != nil {
		modelID = strings.TrimSpace(*sf.model)
	}
	if modelID == "" || modelID == "mock" {
		if sf.ggufPath != nil && *sf.ggufPath != "" && *sf.ggufPath != "default" {
			modelID = *sf.ggufPath
		} else {
			modelID = projectassets.DefaultCodexModelID
		}
	}

	targetPath := ""
	if sf.codexConfigPath != nil && *sf.codexConfigPath != "" {
		targetPath = *sf.codexConfigPath
	} else {
		targetPath = projectassets.ResolveCodexConfigFile("", "")
	}

	if write {
		modified, err := projectassets.EnsureCodexProviderConfig(targetPath, baseURL, modelID, projectassets.DefaultCodexWireAPI, projectassets.DefaultCodexEnvKey)
		if err != nil {
			fmt.Fprintf(out, "fak serve: update %s: %v\n", targetPath, err)
			os.Exit(1)
		}
		if modified {
			fmt.Fprintf(out, "fak serve: updated %s with provider \"fak\" (%s, model: %s)\n", targetPath, baseURL, modelID)
		} else {
			fmt.Fprintf(out, "fak serve: %s is already configured with fak serve backend\n", targetPath)
		}
		return
	}

	raw := projectassets.GenerateCodexConfig(baseURL, modelID, projectassets.DefaultCodexWireAPI, projectassets.DefaultCodexEnvKey)
	fmt.Fprint(out, raw)
}
