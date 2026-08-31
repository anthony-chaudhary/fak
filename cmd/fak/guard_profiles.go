package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

const guardProfilesDisableCommand = "set --output-profile full and --work-profile standard"

type guardProfileCapture struct {
	Schema            string `json:"schema"`
	OutputProfile     string `json:"output_profile"`
	WorkProfile       string `json:"work_profile"`
	OutputDigest      string `json:"output_digest,omitempty"`
	WorkDigest        string `json:"work_digest,omitempty"`
	CompositeDigest   string `json:"composite_digest"`
	Harness           string `json:"harness"`
	ActivationSeam    string `json:"activation_seam"`
	DisableCommand    string `json:"disable_command"`
	DefaultActivation bool   `json:"default_activation"`
}

func injectGuardProfiles(command []string, outputSelection, workSelection string, explicit bool) ([]string, *guardProfileCapture, error) {
	if len(command) == 0 || guardCodexAuthManagementCommand(command) {
		return append([]string(nil), command...), nil, nil
	}
	output := syspromptmmu.DescribeStyle(outputSelection)
	if !output.Known {
		return nil, nil, fmt.Errorf("RESPONSE_PROFILE_UNKNOWN: %q; supported: %s", outputSelection, strings.Join(syspromptmmu.StyleNames(), ", "))
	}
	work := syspromptmmu.DescribeWorkProfile(workSelection)
	if !work.Known {
		return nil, nil, fmt.Errorf("WORK_PROFILE_UNKNOWN: %q; supported: %s", workSelection, strings.Join(syspromptmmu.WorkProfileNames(), ", "))
	}
	if !output.Applied && !work.Applied {
		return append([]string(nil), command...), nil, nil
	}

	harness := strings.ToLower(strings.TrimSuffix(filepath.Base(command[0]), ".exe"))
	if harness != "claude" && harness != "codex" {
		if explicit {
			return nil, nil, fmt.Errorf("PROFILE_UNSUPPORTED_HARNESS: %s has no witnessed profile injection seam; %s", harness, guardProfilesDisableCommand)
		}
		return append([]string(nil), command...), nil, nil
	}

	segments := make([]string, 0, 2)
	if output.Applied {
		segments = append(segments, output.Segment)
	}
	if work.Applied {
		segments = append(segments, work.Segment)
	}
	fragment := strings.Join(segments, "\n\n")
	sum := sha256.Sum256([]byte(fragment))
	activationSeam := "claude --append-system-prompt"
	injected := make([]string, 0, len(command)+2)
	switch harness {
	case "claude":
		injected = append(injected, command[0], "--append-system-prompt", fragment)
	case "codex":
		activationSeam = "codex -c developer_instructions"
		injected = append(injected, command[0], "-c", "developer_instructions="+strconv.Quote(fragment))
	}
	injected = append(injected, command[1:]...)
	capture := &guardProfileCapture{
		Schema: "fak.guard-profiles.v2", OutputProfile: output.Style, WorkProfile: work.Profile,
		OutputDigest: output.Witness, WorkDigest: work.Witness,
		CompositeDigest: "sha256:" + hex.EncodeToString(sum[:]), Harness: harness,
		ActivationSeam: activationSeam, DisableCommand: guardProfilesDisableCommand,
		DefaultActivation: !explicit,
	}
	return injected, capture, nil
}

func marshalGuardProfileCapture(capture *guardProfileCapture) ([]byte, error) {
	return json.MarshalIndent(capture, "", "  ")
}
