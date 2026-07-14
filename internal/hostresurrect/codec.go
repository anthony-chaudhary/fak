package hostresurrect

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
)

func EncodeRequest(req Request) (string, error) {
	if req.Schema != Schema || req.EventID == "" || req.Session == "" || req.CWD == "" || len(req.Command) == 0 || req.ResumeHandle == "" {
		return "", fmt.Errorf("incomplete host resurrection request")
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
