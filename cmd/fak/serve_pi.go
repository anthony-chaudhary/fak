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
		// Also write the SAFE Pi compaction settings so a long session compacts inside the
		// resident envelope instead of running to the hard cap. See projectassets/pi_settings.go.
		settingsPath, sModified, sErr := projectassets.EnsurePiSafeCompaction("", projectassets.PiSafeContextBudget(projectassets.DefaultPiServedWindow))
		if sErr != nil {
			fmt.Fprintf(out, "fak serve: update %s: %v\n", settingsPath, sErr)
			os.Exit(1)
		}
		if sModified {
			fmt.Fprintf(out, "fak serve: wrote safe Pi compaction to %s\n", settingsPath)
		}
		return
	}
	raw, err := projectassets.GeneratePiConfigForWindow(baseURL, modelID, projectassets.DefaultPiServedWindow)
	if err != nil {
		fmt.Fprintf(out, "fak serve: generate models.json: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(out, string(raw))
}
