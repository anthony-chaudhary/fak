package main

import (
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// resolveWorkspaceAdmission folds the --workspace-admission knob (flag or
// FAK_WORKSPACE_ADMISSION env-as-default) and the legacy permissive bool into
// the single gateway lease-admission mode the Config carries:
//
//	explicit mode (open|permissive|strict) > permissive bool true > open default.
//
// An unknown token fails loud with the valid set (the same fail-on-bad-mode
// discipline as resolveUltracodePosture); an empty value is the resolved open
// default unless the legacy permissive bool arms permissive.
func resolveWorkspaceAdmission(mode string, permissive bool) (string, error) {
	resolved, err := gateway.ParseWorkspaceAdmissionMode(mode)
	if err != nil {
		return "", err
	}
	if resolved == "" {
		if permissive {
			resolved = gateway.WorkspaceAdmissionPermissive
		} else {
			resolved = gateway.DefaultWorkspaceAdmissionMode
		}
	}
	return string(resolved), nil
}
