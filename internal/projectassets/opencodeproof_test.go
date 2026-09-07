package projectassets

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Invoke the real generated JS hook using the installed OpenCode output shape.
// Its stdout contains one JSON receipt, while guidance travels in tool output.
func TestOpenCodeProofOutputContract(t *testing.T) {
	if _, err := exec.LookPath("node"); err != nil {
		t.Skip("node is required for the actual plugin witness")
	}
	root := t.TempDir()
	plugin := filepath.Join(root, "proof.mjs")
	if err := os.WriteFile(plugin, []byte(DefaultOpenCodePlugin), 0644); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "node", "--input-type=module", "-", plugin, root)
	cmd.Stdin = strings.NewReader(`
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const hooks = await (await import(pathToFileURL(process.argv[2]).href)).default({directory:process.argv[3]});
const call = async (tool, args) => {
  const result = {title:'kept',output:'tool-result',metadata:{kept:true}};
  await hooks['tool.execute.after']({tool,args,sessionID:'witness',callID:'one'}, result);
  assert.equal(result.title,'kept'); assert.deepEqual(result.metadata,{kept:true});
  assert.ok(result.output.startsWith('tool-result'));
  return result.output;
};
const ordinary = await call('write',{filePath:'docs/example.md'});
assert.ok(ordinary.includes('[dos-proof-guard]'));
assert.ok(!ordinary.includes('Halo-related'));
const hardware = await call('apply_patch',{patchText:'*** Begin Patch\n*** Update File: internal/compute/example.go\n@@\n-x\n+y\n*** End Patch'});
assert.ok(hardware.includes('fak-dev amd-strix-probe'));
assert.ok(hardware.includes('--strix --ablate=none'));
assert.ok(hardware.includes('exclusive lease'));
assert.equal(await call('read',{filePath:'internal/compute/example.go'}),'tool-result');
console.log(JSON.stringify({guidanceDelivered:true,hardwareGuidance:true,stdoutJSON:true}));
`)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("actual JS hook: %v\n%s", err, out)
	}
	var receipt struct{ GuidanceDelivered, HardwareGuidance, StdoutJSON bool }
	if err := json.Unmarshal(out, &receipt); err != nil {
		t.Fatalf("stdout is not one JSON receipt: %v\n%s", err, out)
	}
	if !receipt.GuidanceDelivered || !receipt.HardwareGuidance || !receipt.StdoutJSON {
		t.Fatalf("incomplete receipt: %s", out)
	}
}
