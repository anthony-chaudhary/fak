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
import { existsSync, statSync } from "node:fs";
import { realpath } from "node:fs/promises";
import path from "node:path";

function findRipgrep() {
  const isWin = process.platform === "win32";
  const bin = isWin ? "rg.exe" : "rg";
  const pathEnv = process.env.PATH || (isWin ? (process.env.Path || process.env.path) : "") || "";
  if (pathEnv) {
    for (const dir of pathEnv.split(path.delimiter)) {
      if (!dir) continue;
      const candidate = path.join(dir.replace(/^"(.*)"$/, "$1"), bin);
      try { if (existsSync(candidate)) return candidate; } catch {}
    }
  }
  if (isWin) {
    const progFiles = process.env.ProgramFiles || "C:\\Program Files";
    const progFilesX86 = process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)";
    const localAppData = process.env.LOCALAPPDATA || (process.env.USERPROFILE ? path.join(process.env.USERPROFILE, "AppData", "Local") : "");
    const appData = process.env.APPDATA || (process.env.USERPROFILE ? path.join(process.env.USERPROFILE, "AppData", "Roaming") : "");
    const userProfile = process.env.USERPROFILE || process.env.HOME || "";
    const candidates = [
      "C:\\Program Files\\Git\\usr\\bin\\rg.exe",
      "C:\\Program Files (x86)\\Git\\usr\\bin\\rg.exe",
      path.join(process.env.USERPROFILE || "", ".cargo", "bin", "rg.exe"),
      path.join(process.env.LOCALAPPDATA || "", "Programs", "Git", "usr", "bin", "rg.exe"),
      userProfile && path.join(userProfile, ".cache", "opencode", "bin", "rg.exe"),
      path.join(progFiles, "Git", "usr", "bin", "rg.exe"),
      path.join(progFilesX86, "Git", "usr", "bin", "rg.exe"),
      localAppData && path.join(localAppData, "Programs", "Git", "usr", "bin", "rg.exe"),
      localAppData && path.join(localAppData, "Programs", "Microsoft VS Code", "resources", "app", "node_modules", "@vscode/ripgrep", "bin", "rg.exe"),
      appData && path.join(appData, "Programs", "Microsoft VS Code", "resources", "app", "node_modules", "@vscode/ripgrep", "bin", "rg.exe"),
      path.join(progFiles, "Microsoft VS Code", "resources", "app", "node_modules", "@vscode/ripgrep", "bin", "rg.exe"),
      path.join(progFilesX86, "Microsoft VS Code", "resources", "app", "node_modules", "@vscode/ripgrep", "bin", "rg.exe"),
      localAppData && path.join(localAppData, "Programs", "Microsoft VS Code Insiders", "resources", "app", "node_modules", "@vscode/ripgrep", "bin", "rg.exe"),
      userProfile && path.join(userProfile, ".cargo", "bin", "rg.exe"),
      userProfile && path.join(userProfile, "scoop", "shims", "rg.exe"),
      userProfile && path.join(userProfile, "scoop", "apps", "ripgrep", "current", "rg.exe"),
      "C:\\ProgramData\\chocolatey\\bin\\rg.exe",
    ].filter(Boolean);
    for (const candidate of candidates) {
      try { if (existsSync(candidate)) return candidate; } catch {}
    }
  } else {
    const home = process.env.HOME || "";
    const candidates = [
      "/usr/local/bin/rg",
      "/usr/bin/rg",
      "/opt/homebrew/bin/rg",
      home && path.join(home, ".cargo", "bin", "rg"),
      home && path.join(home, ".cache", "opencode", "bin", "rg"),
    ].filter(Boolean);
    for (const candidate of candidates) {
      try { if (existsSync(candidate)) return candidate; } catch {}
    }
  }
  return null;
}

function findGit() {
  const isWin = process.platform === "win32";
  const bin = isWin ? "git.exe" : "git";
  const pathEnv = process.env.PATH || (isWin ? (process.env.Path || process.env.path) : "") || "";
  if (pathEnv) {
    for (const dir of pathEnv.split(path.delimiter)) {
      if (!dir) continue;
      const candidate = path.join(dir.replace(/^"(.*)"$/, "$1"), bin);
      try { if (existsSync(candidate)) return candidate; } catch {}
    }
  }
  if (isWin) {
    const progFiles = process.env.ProgramFiles || "C:\\Program Files";
    const progFilesX86 = process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)";
    const localAppData = process.env.LOCALAPPDATA || (process.env.USERPROFILE ? path.join(process.env.USERPROFILE, "AppData", "Local") : "");
    const candidates = [
      path.join(progFiles, "Git", "cmd", "git.exe"),
      path.join(progFiles, "Git", "bin", "git.exe"),
      path.join(progFilesX86, "Git", "cmd", "git.exe"),
      path.join(progFilesX86, "Git", "bin", "git.exe"),
      localAppData && path.join(localAppData, "Programs", "Git", "cmd", "git.exe"),
      "C:\\ProgramData\\chocolatey\\bin\\git.exe",
    ].filter(Boolean);
    for (const candidate of candidates) {
      try { if (existsSync(candidate)) return candidate; } catch {}
    }
  } else {
    const candidates = ["/usr/bin/git", "/usr/local/bin/git", "/opt/homebrew/bin/git"];
    for (const candidate of candidates) {
      try { if (existsSync(candidate)) return candidate; } catch {}
    }
  }
  return null;
}

function fallbackOutput() {
  return { title: "grep", output: "No ripgrep binary found; install ripgrep or run git grep", metadata: { matches: 0, truncated: false, error: "rg_not_found" } };
}


function runGitGrep(gitBin, root, target, args, context) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const done = (fn, val) => { if (!settled) { settled = true; fn(val); } };
    const gitArgv = ["grep", "--no-index", "-n", "-I", "-e", args.pattern, "--"];
    const relTarget = path.relative(root, target);
    let isFile = false;
    try { isFile = statSync(target).isFile(); } catch {}
    const isExternal = relTarget === ".." || relTarget.startsWith(".." + path.sep) || path.isAbsolute(relTarget);
    const targetPath = isExternal ? target : (relTarget.replace(/\\/g, "/") || ".");
    if (isFile || isExternal) {
      gitArgv.push(targetPath);
    } else if (args.include) {
      if (targetPath !== ".") {
        gitArgv.push(path.posix.join(targetPath, args.include.replace(/\\/g, "")));
      } else {
        gitArgv.push(args.include.replace(/\\/g, "/"));
      }
    } else {
      gitArgv.push(targetPath);
    }
    const child = spawn(gitBin, gitArgv, { cwd: root, windowsHide: true, signal: context.abort });
    let output = "", stderr = "", matches = 0, clipped = false, limited = false, failure;
    function fail(error) { if (!failure) { failure = error; child.kill(); } }
    let lineBuffer = "";
    child.stdout.on("data", chunk => {
      if (failure || limited) return;
      lineBuffer += chunk.toString("utf8");
      let idx;
      while ((idx = lineBuffer.indexOf("\n")) >= 0) {
        const rawLine = lineBuffer.slice(0, idx).replace(/\r$/, "");
        lineBuffer = lineBuffer.slice(idx + 1);
        if (Buffer.byteLength(rawLine) > 1024 * 1024) {
          fail(new Error("grep JSON record exceeds 1 MiB; narrow search or use a structured reader"));
          return;
        }
        if (!rawLine) continue;
        const m = rawLine.match(/^(.*?):(\d+):(.*)$/);
        if (!m) continue;
        matches++;
        if (matches > 100) { clipped = true; limited = true; child.kill(); return; }
        let preview = m[3].replace(/[\r\n]+$/, "");
        if (preview.length > 2000) preview = preview.slice(0, 2000) + " [preview truncated]";
        const relPath = path.isAbsolute(m[1]) ? path.relative(root, m[1]) : m[1];
        const line = relPath + ":" + m[2] + ":" + preview + "\n";
        if (Buffer.byteLength(output) + Buffer.byteLength(line) <= 16384) output += line;
        else clipped = true;
      }
      if (Buffer.byteLength(lineBuffer) > 1024 * 1024) {
        fail(new Error("grep JSON record exceeds 1 MiB; narrow search or use a structured reader"));
        return;
      }
    });
    child.stderr.on("data", chunk => { stderr = (stderr + chunk.toString()).slice(0, 4096); });
    child.on("error", err => {
      if (err && err.code === "ENOENT") {
        done(resolve, fallbackOutput());
      } else {
        done(reject, err);
      }
    });
    child.on("close", (code, signal) => {
      if (settled) return;
      if (failure) return done(reject, failure);
      if (context.abort.aborted) return done(reject, context.abort.reason || new Error("grep aborted"));
      if (!limited && (signal || (code !== 0 && code !== 1))) return done(reject, new Error(stderr || "git grep failed: " + code));
      if (!limited && lineBuffer.length > 0) {
        const rawLine = lineBuffer.replace(/\r$/, "");
        lineBuffer = "";
        if (Buffer.byteLength(rawLine) > 1024 * 1024) return done(reject, new Error("grep JSON record exceeds 1 MiB; narrow search or use a structured reader"));
        const m = rawLine.match(/^(.*?):(\d+):(.*)$/);
        if (m) {
          matches++;
          if (matches > 100) { clipped = true; limited = true; }
          else {
            let preview = m[3].replace(/[\r\n]+$/, "");
            if (preview.length > 2000) preview = preview.slice(0, 2000) + " [preview truncated]";
            const relPath = path.isAbsolute(m[1]) ? path.relative(root, m[1]) : m[1];
            const line = relPath + ":" + m[2] + ":" + preview + "\n";
            if (Buffer.byteLength(output) + Buffer.byteLength(line) <= 16384) output += line;
            else clipped = true;
          }
        }
      }
      if (clipped) output += "[results truncated; " + (limited ? "at least " : "") + matches + " matching lines observed]\n";
      done(resolve, { title: "grep", output: output || "No matches found", metadata: { matches, truncated: clipped } });
    });
  });
}

function runRipgrep(rgBin, root, target, argv, context, gitBin, fallbackToGit) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const done = (fn, val) => { if (!settled) { settled = true; fn(val); } };
    const child = spawn(rgBin, argv, { cwd: root, windowsHide: true, signal: context.abort });
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
    child.on("error", err => {
      if (err && err.code === "ENOENT") {
        if (gitBin && fallbackToGit) {
          fallbackToGit().then(r => done(resolve, r), e => done(reject, e));
        } else {
          done(resolve, fallbackOutput());
        }
      } else {
        done(reject, err);
      }
    });
    child.on("close", (code, signal) => {
      if (settled) return;
      if (failure) return done(reject, failure);
      if (context.abort.aborted) return done(reject, context.abort.reason || new Error("grep aborted"));
      if (!limited && (signal || (code !== 0 && code !== 1))) return done(reject, new Error(stderr || "ripgrep failed: " + code));
      if (!limited && pending.length) return done(reject, new Error("incomplete ripgrep JSON record"));
      if (clipped) output += "[results truncated; " + (limited ? "at least " : "") + matches + " matching lines observed]\n";
      done(resolve, { title: "grep", output: output || "No matches found", metadata: { matches, truncated: clipped } });
    });
  });
}

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
      const rgBin = findRipgrep();
      const gitBin = findGit();
      if (!rgBin) {
        if (gitBin) {
          return await runGitGrep(gitBin, root, target, args, context);
        }
        return fallbackOutput();
      }
      const argv = ["--no-config", "--json", "--hidden", "--glob", "!**/.git/**", "--regexp", args.pattern];
      if (args.include) argv.push("--glob", args.include);
      argv.push("--", target);
      return await runRipgrep(rgBin, root, target, argv, context, gitBin, () => runGitGrep(gitBin, root, target, args, context));
    },
  }) } };
}
`
