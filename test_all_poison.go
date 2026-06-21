package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"os"
)

func main() {
	ctx := context.Background()
	raw, _ := os.ReadFile("testdata/poison.json")
	var fixture struct {
		Results []struct {
			Name    string `json:"name"`
			Tool    string `json:"tool"`
			Payload string `json:"payload"`
		} `json:"results"`
	}
	json.Unmarshal(raw, &fixture)

	for _, res := range fixture.Results {
		m := ctxmmu.New()
		c := &abi.ToolCall{Tool: res.Tool, Args: abi.Ref{Kind: abi.RefInline}, Meta: map[string]string{}}
		r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(res.Payload)}}
		v := m.Admit(ctx, c, r)

		// Resolve the resulting payload
		resultBytes := []byte{}
		if r.Payload.Kind == abi.RefInline {
			resultBytes = r.Payload.Inline
		} else {
			if res := abi.ActiveResolver(); res != nil {
				b, _ := res.Resolve(ctx, r.Payload)
				resultBytes = b
			}
		}

		// Check for secrets
		hasSecret := bytes.Contains(resultBytes, []byte("sk-")) || bytes.Contains(resultBytes, []byte("AKIA"))
		hasInjection := bytes.Contains(resultBytes, []byte("ignore previous")) || bytes.Contains(resultBytes, []byte("exfiltrate"))

		fmt.Printf("%s: verdict=%d, quarantined=%v, payload_len=%d, has_secret=%v, has_injection=%v\n",
			res.Name, v.Kind, ctxmmu.Quarantined(r), len(resultBytes), hasSecret, hasInjection)
		if v.Kind == 3 {
			fmt.Printf("  -> Quarantine verdict OK\n")
		}
	}
}
