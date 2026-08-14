package main

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

const guardResponseProfileDisableCommand = "remove --output-profile (or set it to full)"

type guardResponseProfileCapture struct {
	Canonical      string `json:"canonical"`
	Family         string `json:"family"`
	Intensity      string `json:"intensity"`
	FragmentDigest string `json:"fragment_digest"`
	Harness        string `json:"harness"`
	ActivationSeam string `json:"activation_seam"`
	DisableCommand string `json:"disable_command"`
	SourceRevision string `json:"source_revision,omitempty"`
	SourceDigest   string `json:"source_digest,omitempty"`
}

func injectGuardResponseProfile(command []string, selection string) ([]string, *guardResponseProfileCapture, error) {
	selection = strings.TrimSpace(selection)
	if selection == "" || strings.EqualFold(selection, syspromptmmu.StyleFull) {
		return append([]string(nil), command...), nil, nil
	}
	profile := syspromptmmu.DescribeStyle(selection)
	if !profile.Known || !profile.Applied {
		return nil, nil, fmt.Errorf("RESPONSE_PROFILE_UNKNOWN: %q is not a shipped response profile", selection)
	}
	if len(command) == 0 {
		return nil, nil, fmt.Errorf("RESPONSE_PROFILE_NO_HARNESS: response profile requires a wrapped harness")
	}
	harness := strings.ToLower(strings.TrimSuffix(filepath.Base(command[0]), filepath.Ext(command[0])))
	if harness != "claude" {
		return nil, nil, fmt.Errorf("RESPONSE_PROFILE_UNSUPPORTED_HARNESS: %s has no witnessed instruction injection seam (supported: claude)", harness)
	}
	injected := make([]string, 0, len(command)+2)
	injected = append(injected, command[0], "--append-system-prompt", profile.Segment)
	injected = append(injected, command[1:]...)
	capture := &guardResponseProfileCapture{
		Canonical: profile.Style, Family: profile.Family, Intensity: profile.Intensity,
		FragmentDigest: profile.Witness, Harness: harness,
		ActivationSeam: "claude --append-system-prompt", DisableCommand: guardResponseProfileDisableCommand,
		SourceRevision: profile.SourceRevision, SourceDigest: profile.SourceDigest,
	}
	return injected, capture, nil
}

func marshalGuardResponseProfileCapture(capture *guardResponseProfileCapture) ([]byte, error) {
	if capture == nil {
		return nil, nil
	}
	return json.Marshal(capture)
}
