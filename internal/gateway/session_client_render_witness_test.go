package gateway

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestSessionReferenceRenderWitnesses(t *testing.T) {
	actions := SessionCapabilityCorpus(sessionClientCapabilities)
	d := SessionClientDescriptor{Schema: sessionClientSchema, SessionID: "render-session", ExecutionEpoch: "epoch-render", Capabilities: append([]string(nil), sessionClientCapabilities...), CapabilityDigest: capabilityDigest(sessionClientCapabilities), Actions: actions, Terminal: SessionTerminalView{Transcript: "ready\n", ByteLength: 6, Digest: terminalView([]byte("ready\n")).Digest}}
	terminal, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	var browser bytes.Buffer
	if err := sessionClientPage.Execute(&browser, d.SessionID); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(browser.String(), "@media(max-width:520px)") {
		t.Fatal("browser witness lacks constrained/mobile layout")
	}
	for _, action := range actions {
		if !strings.Contains(string(terminal), `"id": "`+action.ID+`"`) {
			t.Fatalf("terminal descriptor lacks %s", action.ID)
		}
	}
	t.Logf("terminal-wide:\n%s", terminal)
	t.Logf("browser-wide-and-mobile:\n%s", browser.String())
}
