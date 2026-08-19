package main

import (
	"fmt"
	"os"
	"strings"
)

func resolveNativeCodeWorkspace(native, enabled bool, configured string) (string, error) {
	if !native || !enabled {
		return "", nil
	}
	if workspace := strings.TrimSpace(configured); workspace != "" {
		return workspace, nil
	}
	workspace, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("resolve current workspace for default native code tools: %w", err)
	}
	return workspace, nil
}
