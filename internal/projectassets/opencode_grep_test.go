package projectassets

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestOpenCodeGrepLongLine(t *testing.T) {
	root := t.TempDir()
	if err := SyncOpenCodePlugin(root); err != nil {
		t.Fatal(err)
	}
	if err := VerifyOpenCodePlugin(root); err != nil {
		t.Fatal(err)
	}
	pluginPath := filepath.Join(root, ".opencode", "plugins", "fak-grep.js")
	if _, err := os.Stat(pluginPath); err != nil {
		t.Fatal(err)
	}
	node, err := exec.LookPath("node")
	sdk := os.Getenv("FAK_OPENCODE_SDK")
	if err != nil || sdk == "" {
		t.Skip("live regression needs node, rg and FAK_OPENCODE_SDK pointing to installed plugin/dist/tool.js")
	}
	script := filepath.Join(root, "witness.mjs")
	if err := os.WriteFile(script, []byte(openCodeGrepWitness), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(node, script, pluginPath, sdk, root)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("actual SDK/rg long-line witness: %v\n%s", err, out)
	} else {
		t.Log(string(out))
	}
}

const openCodeGrepWitness = `import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
const [pluginFile, sdk, root] = process.argv.slice(2);
let code = await readFile(pluginFile, 'utf8');
code = code.replace('@opencode-ai/plugin', pathToFileURL(sdk).href);
const { default: plugin } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
const hooks = await plugin({});
assert.deepEqual(Object.keys(hooks.tool), ['grep']);
await writeFile(path.join(root,'long.txt'), 'prefix ' + 'x'.repeat(100000) + ' needle\nordinary needle\n');
let requests=[];
const ctx={directory:root,worktree:root,abort:new AbortController().signal,ask:async(r)=>{requests.push(r)},metadata:()=>{}};
let result=await hooks.tool.grep.execute({pattern:'needle',path:'long.txt'},ctx);
assert.match(result.output,/long\.txt:1:/);
assert.match(result.output,/long\.txt:2:ordinary needle/);
assert.match(result.output,/preview truncated/);
assert.ok(Buffer.byteLength(result.output)<20000);
assert.equal(requests[0].permission,'grep');
assert.equal((await hooks.tool.grep.execute({pattern:'absent',path:'long.txt'},ctx)).output,'No matches found');
await assert.rejects(hooks.tool.grep.execute({pattern:'[',path:'long.txt'},ctx));
await assert.rejects(hooks.tool.grep.execute({pattern:'needle'}, {...ctx,ask:async()=>{throw new Error('permission denied')}}),/permission denied/);
await assert.rejects(hooks.tool.grep.execute({pattern:'needle'}, {...ctx,abort:AbortSignal.abort()}));
await writeFile(path.join(root,'oversize.txt'), 'needle' + 'x'.repeat(1100000) + '\n');
await assert.rejects(hooks.tool.grep.execute({pattern:'needle',path:'oversize.txt'},ctx),/exceeds 1 MiB/);
await writeFile(path.join(root,'many.txt'), 'needle\n'.repeat(102));
assert.match((await hooks.tool.grep.execute({pattern:'needle',path:'many.txt'},ctx)).output,/at least 101 matching lines observed/);
console.log('PASS actual SDK same-name grep: 100KB line, ordinary line, bounded preview, no-match, invalid regex, permission denial, abort');
`
