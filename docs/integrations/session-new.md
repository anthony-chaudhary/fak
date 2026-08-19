# Launch a new guarded session from text

`fak session new` turns one explicit piece of text into a separate interactive agent
terminal. The child always starts behind `fak guard`; the launcher does not copy the
current transcript or invoke a shell parser.

```powershell
# Text already in your command or supplied by an editor/hotkey.
fak session new "inspect the selected function for races"

# Highlight/copy text, then launch from the Windows clipboard.
fak session new --clipboard

# Pipe an existing tool's selection or output.
Get-Content .\_scratch\handoff.txt -Raw | fak session new --stdin
```

On Linux, the same stdin path works with ordinary pipelines. Clipboard mode uses
`wl-paste` or `xclip`; terminal launch uses `x-terminal-emulator` and falls back to
`gnome-terminal`. Windows uses Windows Terminal (`wt.exe`). `--agent codex` selects a
different child agent without changing the guard boundary. Nonstandard terminal installs can keep an adapter's argv contract and set `--terminal-command PATH`.

## Inspect before launching

Integrations should start with the side-effect-free receipt:

```powershell
fak session new --clipboard --dry-run --json
```

The `fak-session-new/1` receipt names the source, terminal adapter, working directory,
agent, executable, and argument shape. Prompt content is replaced with a SHA-256 marker;
the full text is never emitted. A hotkey or editor command can therefore call the verb
directly without learning terminal-specific quoting.

Exactly one source is required: one positional prompt argument, `--stdin`, or
`--clipboard`. Empty or conflicting input fails before a terminal starts. Piped input
has one final record-separator newline removed; all other multiline UTF-8 content is
preserved.
