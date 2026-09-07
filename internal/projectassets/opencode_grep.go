package projectassets

import (
	"fmt"
	"os"
	"path/filepath"
)

const openCodeGrepPath = ".opencode/plugins/fak-grep.js"

func syncOpenCodeGrep(root string) error {
	return os.WriteFile(filepath.Join(root, filepath.FromSlash(openCodeGrepPath)), []byte(defaultOpenCodeGrep), 0644)
}

func verifyOpenCodeGrep(root string) error {
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(openCodeGrepPath)))
	if err != nil {
		return err
	}
	if string(data) != defaultOpenCodeGrep {
		return fmt.Errorf("OpenCode grep plugin differs from canonical asset")
	}
	return nil
}

const defaultOpenCodeGrep = `import { tool } from "@opencode-ai/plugin";
import { spawn } from "node:child_process";
import { realpath } from "node:fs/promises";
import path from "node:path";
export default async function FakGrep() {
  return { tool: { grep: tool({
    description: "Search file contents with exact paths and line numbers and bounded previews. Supports matching lines up to a 1 MiB JSON record; larger records fail explicitly.",
    args: { pattern: tool.schema.string(), path: tool.schema.string().optional(), include: tool.schema.string().optional() },
    async execute(args, context) {
      if (!args.pattern) throw new Error("pattern is required");
      context.abort.throwIfAborted();
      const root = await realpath(context.directory);
      let target;
      try {
        target = await realpath(path.resolve(root, args.path || "."));
      } catch (e) {
        if (e && e.code === "ENOENT") return { title: "grep", output: "No matches found", metadata: { matches: 0, truncated: false } };
        throw e;
      }
      const relative = path.relative(root, target);
      if (relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative)) {
        await context.ask({ permission: "external_directory", patterns: [target], always: [target], metadata: { path: target } });
      }
      await context.ask({ permission: "grep", patterns: [args.pattern], always: ["*"], metadata: { pattern: args.pattern, path: target, include: args.include } });
      if (!args.pattern) throw new Error("pattern is required");
      context.abort.throwIfAborted();
      const argv = ["--no-config", "--json", "--hidden", "--glob", "!**/.git/**", "--regexp", args.pattern];
      if (args.include) argv.push("--glob", args.include);
      argv.push("--", target);
      return await new Promise((resolve, reject) => {
        const child = spawn(process.platform === "win32" ? "rg.exe" : "rg", argv, { cwd: root, windowsHide: true, signal: context.abort });
        let pending = Buffer.alloc(0), output = "", stderr = "", matches = 0, clipped = false, limited = false, failure;
        function fail(error) { if (!failure) { failure = error; child.kill(); } }
        child.stdout.on("data", chunk => {
          if (failure || limited) return;
          let offset = 0;
          while (offset < chunk.length) {
            const end = chunk.indexOf(10, offset), stop = end < 0 ? chunk.length : end;
            if (pending.length + stop - offset > 1024 * 1024) { fail(new Error("grep JSON record exceeds 1 MiB; narrow search or use a structured reader")); return; }
            pending = Buffer.concat([pending, chunk.subarray(offset, stop)]);
            offset = stop + 1;
            if (end < 0) break;
            try {
              const event = JSON.parse(pending.toString("utf8"));
              pending = Buffer.alloc(0);
              if (event.type !== "match") continue;
              const data = event.data;
              if (typeof data.path?.text !== "string" || typeof data.lines?.text !== "string") continue;
              matches++;
              if (matches > 100) { clipped = true; limited = true; child.kill(); return; }
              let preview = data.lines.text.replace(/[\r\n]+$/, "");
              if (preview.length > 2000) preview = preview.slice(0, 2000) + " [preview truncated]";
              const line = path.relative(root, data.path.text) + ":" + data.line_number + ":" + preview + "\n";
              if (Buffer.byteLength(output) + Buffer.byteLength(line) <= 16384) output += line;
              else clipped = true;
            } catch (error) { fail(error); return; }
          }
        });
        child.stderr.on("data", chunk => { stderr = (stderr + chunk.toString()).slice(0, 4096); });
        child.on("error", reject);
        child.on("close", (code, signal) => {
          if (failure) return reject(failure);
          if (context.abort.aborted) return reject(context.abort.reason || new Error("grep aborted"));
          if (!limited && (signal || (code !== 0 && code !== 1))) return reject(new Error(stderr || "ripgrep failed: " + code));
          if (!limited && pending.length) return reject(new Error("incomplete ripgrep JSON record"));
          if (clipped) output += "[results truncated; " + (limited ? "at least " : "") + matches + " matching lines observed]\n";
          resolve({ title: "grep", output: output || "No matches found", metadata: { matches, truncated: clipped } });
        });
      });
    },
  }) } };
}
`
