package gateway

import (
	"net"
	"strings"
)

// positiveSSHReadProbe proves the local workspace effects of one constrained
// identity probe. It supplies no remote authorization and never skips the kernel.
func positiveSSHReadProbe(command string) bool {
	if command == "" || len(command) > 4096 {
		return false
	}
	for _, c := range command {
		if c < 0x20 || c > 0x7e || strings.ContainsRune("'\"`$;|&<>\\(){}[]!*?~#", c) {
			return false
		}
	}
	words := strings.Fields(command)
	if len(words) < 2 {
		return false
	}
	switch words[0] {
	case "/usr/bin/ssh", "/bin/ssh", "C:/Windows/System32/OpenSSH/ssh.exe":
	default:
		return false
	}
	required := map[string]string{
		"BatchMode": "yes", "StrictHostKeyChecking": "yes", "UpdateHostKeys": "no",
		"ClearAllForwardings": "yes", "PermitLocalCommand": "no",
		"ProxyCommand": "none", "ProxyJump": "none", "ControlMaster": "no",
		"ControlPath": "none", "AddKeysToAgent": "no",
	}
	seen := make(map[string]bool)
	i := 1
	for i < len(words) && strings.HasPrefix(words[i], "-") {
		flag := words[i]
		i++
		if flag == "--" {
			break
		}
		if flag == "-o" {
			if i == len(words) {
				return false
			}
			key, value, ok := strings.Cut(words[i], "=")
			i++
			if !ok || seen[key] {
				return false
			}
			want, known := required[key]
			if key == "ConnectTimeout" {
				want, known = "5", true
			}
			if !known || value != want {
				return false
			}
			seen[key] = true
			continue
		}
		if seen[flag] {
			return false
		}
		seen[flag] = true
		switch flag {
		case "-F":
			if i == len(words) || words[i] != "none" {
				return false
			}
			i++
		case "-T", "-a", "-x", "-4", "-n":
		default:
			return false
		}
	}
	if len(words)-i != 2 || !seen["-F"] || !seen["-T"] || !seen["-a"] || !seen["-x"] {
		return false
	}
	for key := range required {
		if !seen[key] {
			return false
		}
	}
	if !sshReadProbeDestination(words[i]) {
		return false
	}
	return words[i+1] == "/bin/hostname" || words[i+1] == "/usr/bin/hostname"
}

func sshReadProbeDestination(destination string) bool {
	user, host, ok := strings.Cut(destination, "@")
	if !ok || user == "" || host == "" || len(user) > 64 || len(host) > 253 {
		return false
	}
	for _, c := range user {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false
		}
	}
	if user[0] == '-' {
		return false
	}
	if strings.Trim(host, "0123456789.") == "" && net.ParseIP(host) == nil {
		return false
	}
	for _, label := range strings.Split(host, ".") {
		if label == "" || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, c := range label {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-') {
				return false
			}
		}
	}
	return true
}
