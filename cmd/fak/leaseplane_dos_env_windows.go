//go:build windows

package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"golang.org/x/sys/windows/registry"
)

// Shell Folders stores the OS-expanded roaming path; KnownFolderPath expansion
// depends on USERPROFILE, which Ops isolates. Restore only the DOS runtime lookup.
func dosLeaseAuthorityEnvironment(base []string) ([]string, error) {
	key, err := registry.OpenKey(registry.CURRENT_USER, `Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders`, registry.QUERY_VALUE)
	if err != nil {
		return nil, err
	}
	defer key.Close()
	appData, kind, err := key.GetStringValue("AppData")
	if err != nil {
		return nil, err
	}
	if kind != registry.SZ || !filepath.IsAbs(appData) {
		return nil, fmt.Errorf("Windows roaming AppData is not an expanded absolute path")
	}
	env := make([]string, 0, len(base)+1)
	for _, item := range base {
		name, _, _ := strings.Cut(item, "=")
		if !strings.EqualFold(name, "APPDATA") {
			env = append(env, item)
		}
	}
	return append(env, "APPDATA="+appData), nil
}
