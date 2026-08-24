package hostresurrect

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

func EncodeRequest(req Request) (string, error) {
	if req.Schema != Schema || strings.TrimSpace(req.EventID) == "" || strings.TrimSpace(req.Session) == "" || !filepath.IsAbs(req.CWD) || len(req.Command) == 0 || strings.TrimSpace(req.ResumeHandle) == "" {
		return "", fmt.Errorf("incomplete host resurrection request")
	}
	exe := strings.ToLower(filepath.Base(req.Command[0]))
	if exe != "claude" && exe != "claude.exe" && exe != "codex" && exe != "codex.exe" {
		return "", fmt.Errorf("unsupported relaunch executable %q", req.Command[0])
	}
	resume := 0
	for i, arg := range req.Command {
		if arg == "--resume" && i+1 < len(req.Command) && req.Command[i+1] == req.ResumeHandle {
			resume++
		}
	}
	if resume != 1 {
		return "", fmt.Errorf("relaunch command must contain exactly one matching --resume")
	}
	b, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

func DecodeRequest(encoded string) (Request, error) {
	b, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return Request{}, err
	}
	var req Request
	if err := json.Unmarshal(b, &req); err != nil {
		return Request{}, err
	}
	if _, err := EncodeRequest(req); err != nil {
		return Request{}, err
	}
	return req, nil
}
