package adjudicator

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func missionTestBasePolicy() Policy {
	return Policy{
		Allow: map[string]bool{
			"Edit":       true,
			"edit":       true,
			"Write":      true,
			"write":      true,
			"write_file": true,
			"Bash":       true,
			"bash":       true,
			"Read":       true,
			"read":       true,
		},
	}
}

func TestMissionWriteSetEnforcement(t *testing.T) {
	basePolicy := missionTestBasePolicy()

	t.Run("StructuredTools_EditAndWrite", func(t *testing.T) {
		mc := MissionContract{
			WriteSet: []string{"allowed_file.txt", "src/module.go"},
		}
		a := NewWithMission(basePolicy, mc)
		ctx := context.Background()

		// Allowed Edit targeting WriteSet
		callEditAllow := &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"allowed_file.txt"}`)},
		}
		v := a.Adjudicate(ctx, callEditAllow)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("Edit allowed_file.txt: got kind=%v, want VerdictAllow", v.Kind)
		}

		// Allowed Write targeting WriteSet
		callWriteAllow := &abi.ToolCall{
			Tool: "Write",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"file_path":"src/module.go","content":"package main"}`)},
		}
		v = a.Adjudicate(ctx, callWriteAllow)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("Write src/module.go: got kind=%v, want VerdictAllow", v.Kind)
		}

		// Denied Edit outside WriteSet
		callEditDeny := &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"outside.txt"}`)},
		}
		v = a.Adjudicate(ctx, callEditDeny)
		if v.Kind != abi.VerdictDeny {
			t.Fatalf("Edit outside.txt: got kind=%v, want VerdictDeny", v.Kind)
		}
		if v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("Edit outside.txt reason: got %v (%s), want %v (%s)",
				v.Reason, abi.ReasonName(v.Reason), ReasonMissionWriteSetViolation, ReasonMissionWriteSetViolationName)
		}

		// Denied Write outside WriteSet
		callWriteDeny := &abi.ToolCall{
			Tool: "Write",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"file_path":"other/forbidden.go","content":"package other"}`)},
		}
		v = a.Adjudicate(ctx, callWriteDeny)
		if v.Kind != abi.VerdictDeny {
			t.Fatalf("Write other/forbidden.go: got kind=%v, want VerdictDeny", v.Kind)
		}
		if v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("Write other/forbidden.go reason: got %v (%s), want %v (%s)",
				v.Reason, abi.ReasonName(v.Reason), ReasonMissionWriteSetViolation, ReasonMissionWriteSetViolationName)
		}
	})

	t.Run("BashShellRedirectionsAndPipes", func(t *testing.T) {
		mc := MissionContract{
			WriteSet: []string{"allowed.txt"},
		}
		a := NewWithMission(basePolicy, mc)
		ctx := context.Background()

		// echo > allowed succeeds
		callEchoAllow := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"echo hello > allowed.txt"}`)},
		}
		v := a.Adjudicate(ctx, callEchoAllow)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("echo > allowed.txt: got kind=%v, want VerdictAllow", v.Kind)
		}

		// echo > outside refused
		callEchoDeny := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"echo hello > outside.txt"}`)},
		}
		v = a.Adjudicate(ctx, callEchoDeny)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("echo > outside.txt: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}

		// >> refused
		callAppendDeny := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"echo hello >> outside.txt"}`)},
		}
		v = a.Adjudicate(ctx, callAppendDeny)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("echo >> outside.txt: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}

		// tee refused
		callTeeDeny := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"echo hello | tee outside.txt"}`)},
		}
		v = a.Adjudicate(ctx, callTeeDeny)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("tee outside.txt: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}

		// sed -i refused
		callSedDeny := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"sed -i 's/a/b/' outside.txt"}`)},
		}
		v = a.Adjudicate(ctx, callSedDeny)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("sed -i outside.txt: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}

		// subshells refused
		callSubshellDeny := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"(echo hello > outside.txt)"}`)},
		}
		v = a.Adjudicate(ctx, callSubshellDeny)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("subshell (echo > outside.txt): got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}

		callNestedSubshellDeny := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"sh -c 'echo hello > outside.txt'"}`)},
		}
		v = a.Adjudicate(ctx, callNestedSubshellDeny)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("nested subshell sh -c: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}

		// read-only commands stay allowed
		for _, cmd := range []string{
			"cat allowed.txt",
			"cat outside.txt",
			"grep hello allowed.txt",
			"ls -la",
		} {
			callRead := &abi.ToolCall{
				Tool: "Bash",
				Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(fmt.Sprintf(`{"command":%q}`, cmd))},
			}
			v = a.Adjudicate(ctx, callRead)
			if v.Kind != abi.VerdictAllow {
				t.Fatalf("read-only command %q: got kind=%v, want VerdictAllow", cmd, v.Kind)
			}
		}
	})

	t.Run("AsymmetricDirectoryExtraction", func(t *testing.T) {
		tmpDir := t.TempDir()
		extractDir := filepath.Join(tmpDir, "extracted")
		if err := os.MkdirAll(extractDir, 0755); err != nil {
			t.Fatalf("MkdirAll failed: %v", err)
		}

		preExistingFile := filepath.Join(extractDir, "existing.txt")
		if err := os.WriteFile(preExistingFile, []byte("pre-existing content"), 0644); err != nil {
			t.Fatalf("WriteFile preExistingFile failed: %v", err)
		}

		allowedPreExisting := filepath.Join(extractDir, "allowed_existing.txt")
		if err := os.WriteFile(allowedPreExisting, []byte("allowed pre-existing content"), 0644); err != nil {
			t.Fatalf("WriteFile allowedPreExisting failed: %v", err)
		}

		newFile := filepath.Join(extractDir, "new.txt")
		outsideFile := filepath.Join(tmpDir, "outside.txt")

		mc := MissionContract{
			WriteSet:    []string{allowedPreExisting},
			ExtractInto: []string{extractDir},
		}
		a := NewWithMission(basePolicy, mc)
		ctx := context.Background()

		// 1. Creating new files under ExtractInto allowed
		newPayload, _ := json.Marshal(map[string]any{"filePath": newFile, "content": "new content"})
		callNew := &abi.ToolCall{
			Tool: "Write",
			Args: abi.Ref{Kind: abi.RefInline, Inline: newPayload},
		}
		v := a.Adjudicate(ctx, callNew)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("creating new file under ExtractInto: got kind=%v (%s), want VerdictAllow", v.Kind, abi.ReasonName(v.Reason))
		}

		// 2. Modifying pre-existing files under ExtractInto refused unless in WriteSet
		prePayload, _ := json.Marshal(map[string]any{"filePath": preExistingFile})
		callPreExisting := &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: prePayload},
		}
		v = a.Adjudicate(ctx, callPreExisting)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("modifying pre-existing file under ExtractInto not in WriteSet: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}

		// Also via Bash command
		cmdOverwrite := fmt.Sprintf("echo overwrite > %q", filepath.ToSlash(preExistingFile))
		bashPrePayload, _ := json.Marshal(map[string]any{"command": cmdOverwrite})
		callPreExistingBash := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: bashPrePayload},
		}
		v = a.Adjudicate(ctx, callPreExistingBash)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("Bash overwrite of pre-existing file under ExtractInto: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}

		// 3. Modifying allowed pre-existing file succeeds
		allowedPrePayload, _ := json.Marshal(map[string]any{"filePath": allowedPreExisting})
		callAllowedPreExisting := &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: allowedPrePayload},
		}
		v = a.Adjudicate(ctx, callAllowedPreExisting)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("modifying allowed pre-existing file in WriteSet: got kind=%v (%s), want VerdictAllow", v.Kind, abi.ReasonName(v.Reason))
		}

		// Also via Bash command
		cmdUpdated := fmt.Sprintf("echo updated > %q", filepath.ToSlash(allowedPreExisting))
		bashAllowedPayload, _ := json.Marshal(map[string]any{"command": cmdUpdated})
		callAllowedPreExistingBash := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: bashAllowedPayload},
		}
		v = a.Adjudicate(ctx, callAllowedPreExistingBash)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("Bash modify of allowed pre-existing file in WriteSet: got kind=%v (%s), want VerdictAllow", v.Kind, abi.ReasonName(v.Reason))
		}

		// 4. Writing outside ExtractInto refused
		outsidePayload, _ := json.Marshal(map[string]any{"filePath": outsideFile, "content": "outside content"})
		callOutside := &abi.ToolCall{
			Tool: "Write",
			Args: abi.Ref{Kind: abi.RefInline, Inline: outsidePayload},
		}
		v = a.Adjudicate(ctx, callOutside)
		if v.Kind != abi.VerdictDeny || v.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("writing outside ExtractInto: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v.Kind, v.Reason)
		}
	})

	t.Run("ContractPropagationModes", func(t *testing.T) {
		ctx := context.Background()

		// Mode 1: NewWithMission
		mc1 := MissionContract{WriteSet: []string{"a.txt"}}
		a1 := NewWithMission(basePolicy, mc1)
		v1Allow := a1.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"a.txt"}`)},
		})
		if v1Allow.Kind != abi.VerdictAllow {
			t.Fatalf("NewWithMission allowed target: got kind=%v, want VerdictAllow", v1Allow.Kind)
		}
		v1Deny := a1.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"b.txt"}`)},
		})
		if v1Deny.Kind != abi.VerdictDeny || v1Deny.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("NewWithMission denied target: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v1Deny.Kind, v1Deny.Reason)
		}

		// Mode 2: ContextWithMissionContract
		mc2 := &MissionContract{WriteSet: []string{"c.txt"}}
		a2 := New(basePolicy)
		ctx2 := ContextWithMissionContract(ctx, mc2)
		v2Allow := a2.Adjudicate(ctx2, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"c.txt"}`)},
		})
		if v2Allow.Kind != abi.VerdictAllow {
			t.Fatalf("ContextWithMissionContract allowed target: got kind=%v, want VerdictAllow", v2Allow.Kind)
		}
		v2Deny := a2.Adjudicate(ctx2, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"d.txt"}`)},
		})
		if v2Deny.Kind != abi.VerdictDeny || v2Deny.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("ContextWithMissionContract denied target: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v2Deny.Kind, v2Deny.Reason)
		}

		// Mode 3: SetMissionContract
		mc3 := &MissionContract{WriteSet: []string{"e.txt"}}
		a3 := New(basePolicy)
		a3.SetMissionContract(mc3)
		v3Allow := a3.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"e.txt"}`)},
		})
		if v3Allow.Kind != abi.VerdictAllow {
			t.Fatalf("SetMissionContract allowed target: got kind=%v, want VerdictAllow", v3Allow.Kind)
		}
		v3Deny := a3.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"f.txt"}`)},
		})
		if v3Deny.Kind != abi.VerdictDeny || v3Deny.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("SetMissionContract denied target: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v3Deny.Kind, v3Deny.Reason)
		}

		// Mode 4: ToolCall.Meta with JSON
		a4 := New(basePolicy)
		v4Allow := a4.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"g.txt"}`)},
			Meta: map[string]string{
				"mission_contract": `{"write_set":["g.txt"]}`,
			},
		})
		if v4Allow.Kind != abi.VerdictAllow {
			t.Fatalf("ToolCall.Meta JSON allowed target: got kind=%v, want VerdictAllow", v4Allow.Kind)
		}
		v4Deny := a4.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"h.txt"}`)},
			Meta: map[string]string{
				"mission_contract": `{"write_set":["g.txt"]}`,
			},
		})
		if v4Deny.Kind != abi.VerdictDeny || v4Deny.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("ToolCall.Meta JSON denied target: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v4Deny.Kind, v4Deny.Reason)
		}

		// Mode 4b: ToolCall.Meta with write_set key
		v4bAllow := a4.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"i.txt"}`)},
			Meta: map[string]string{
				"write_set": "i.txt",
			},
		})
		if v4bAllow.Kind != abi.VerdictAllow {
			t.Fatalf("ToolCall.Meta write_set allowed target: got kind=%v, want VerdictAllow", v4bAllow.Kind)
		}
		v4bDeny := a4.Adjudicate(ctx, &abi.ToolCall{
			Tool: "Edit",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"filePath":"j.txt"}`)},
			Meta: map[string]string{
				"write_set": "i.txt",
			},
		})
		if v4bDeny.Kind != abi.VerdictDeny || v4bDeny.Reason != ReasonMissionWriteSetViolation {
			t.Fatalf("ToolCall.Meta write_set denied target: got kind=%v reason=%v, want Deny with ReasonMissionWriteSetViolation", v4bDeny.Kind, v4bDeny.Reason)
		}
	})
}

func TestMissionWriteHelpers(t *testing.T) {
	// Test ExtractCommandWriteTargets on touch, curl, wget, tee, sed
	cases := []struct {
		cmd     string
		wantTgt string
	}{
		{"touch created.txt", "created.txt"},
		{"curl -o downloaded.json https://example.com/api", "downloaded.json"},
		{"wget -O archive.tar.gz https://example.com/archive", "archive.tar.gz"},
		{"echo data > out.txt", "out.txt"},
		{"echo data >> append.txt", "append.txt"},
		{"tee pipe.txt", "pipe.txt"},
		{"sed -i 's/a/b/' in_place.txt", "in_place.txt"},
	}

	for _, tc := range cases {
		tgts := ExtractCommandWriteTargets(tc.cmd)
		found := false
		for _, tgt := range tgts {
			if tgt == tc.wantTgt {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("ExtractCommandWriteTargets(%q): want target %q in %v", tc.cmd, tc.wantTgt, tgts)
		}
	}

	// Test VerifyCommandMissionWrites and VerifyMissionWriteSet
	mc := MissionContract{
		WriteSet: []string{"allowed.txt", "downloaded.json"},
	}

	if err := VerifyCommandMissionWrites("touch allowed.txt", mc); err != nil {
		t.Fatalf("VerifyCommandMissionWrites allowed: got error %v", err)
	}
	if err := VerifyCommandMissionWrites("curl -o downloaded.json http://example", mc); err != nil {
		t.Fatalf("VerifyCommandMissionWrites allowed curl: got error %v", err)
	}
	if err := VerifyCommandMissionWrites("touch forbidden.txt", mc); err == nil {
		t.Fatalf("VerifyCommandMissionWrites forbidden: want error, got nil")
	}

	if err := VerifyMissionWriteSet([]string{"allowed.txt"}, mc); err != nil {
		t.Fatalf("VerifyMissionWriteSet allowed: got error %v", err)
	}
	if err := VerifyMissionWriteSet([]string{"allowed.txt", "forbidden.txt"}, mc); err == nil {
		t.Fatalf("VerifyMissionWriteSet forbidden: want error, got nil")
	}
}
