# Alpha fence: keep your private workflows private

Your skills, prompts, runbooks and strategy notes are your edge. fak lets you use them next
to a public repository without trusting yourself, a teammate or an agent to never paste one
into it.

You point fak at the private files once. It stores **salted hashes of their distinctive
lines** and nothing else. From then on every leak gate refuses content that reproduces one
of those lines, and names the position without repeating the text.

## Use it

```sh
# Fence a private directory (or individual files) for the repository in the current directory.
fak public-scrub fence ~/private/skills ~/private/prompts

# The same audits that guard a public release now also enforce your fence.
fak public-scrub audit-staged          # what you are about to commit
fak public-scrub audit-range A..B      # what a branch adds
fak public-scrub audit-tree            # everything tracked
fak public-scrub audit-message MSGFILE # a commit message
```

A refusal looks like this and exits 1:

```text
PUBLIC_LEAK docs/howto.md:2 [alpha fence: alpha-1]  reproduces a fenced private line (text withheld)
```

Outbound text that fak scans before it leaves the machine goes through the same check. An
agent working in the repository hits the same refusal a person does.

The private files must live outside the repository, or be untracked in it. The builder
refuses a file that the repository tracks.

## What it guarantees

- **No plaintext list of secrets.** The fence at `.fak/alpha-fence.json` holds 12-byte
  salted SHA-256 digests under a random per-fence salt. Entry labels are opaque (`alpha-1`)
  unless you pass `--label-paths`.
- **Reformatting does not evade it.** Lines are compared after lower-casing, collapsing
  whitespace, dropping invisible format characters and stripping leading and trailing
  markup. Re-indenting, re-casing, bulleting or commenting out a private line still
  matches, and so does a private line placed inside a quoted string, a code span, a table
  cell or a JSON string with escaped newlines.
- **Published text is not fenced.** Lines that already appear in the repository's
  committed files are left out when the fence is built, so shared license headers and
  boilerplate do not trip the gate. Files with uncommitted edits are not counted as
  published. Short and punctuation-only lines are never fenced.
- **Refusals do not leak.** A finding carries a file, a line number and the entry label.
- **A broken fence is loud.** A fence that is malformed, empty or present but unreadable
  is itself a finding, never a silent pass.
- **The fence stays local.** The builder adds the fence to the repository's local git
  exclude file when no ignore rule covers it, and the audits refuse a fence that is staged
  or tracked. The audits read the fence only from your working tree, so a committed copy
  cannot weaken the check.
- **You are told what is not covered.** The builder names every input it could not fence
  (unreadable, larger than 4 MiB, binary or UTF-16) on stderr.
- **Nothing leaves the machine.** Building and checking are local file reads and local
  `git` calls; no network call is involved.

## What it does not guarantee

- It matches **lines**, not ideas. A paraphrase, a translation, a private line split across
  two lines, extra words added inside the line, or look-alike characters from another
  alphabet are not detected. Use it with the needle and secret-shape gates, not instead of
  review.
- It is **membership-testable**. Anyone holding the fence file can test whether a guessed
  line is in it. That is why it is kept out of version control; do not share it.
- It protects the repository it is built for, as of when it was built. Rebuild it after
  you change your private files, and build one per repository you want fenced.

## Tuning

| Flag | Default | Meaning |
| --- | --- | --- |
| `--min-hits N` | 1 | distinct fenced lines one file or payload must reproduce before it is refused |
| `--min-line-len N` | 24 | shortest normalized line worth fencing |
| `--label-paths` | off | label entries with their file path instead of an opaque id |
| `--root DIR` | git toplevel | repository the fence protects |
| `--json` | off | machine-readable build summary, including skipped inputs |

Raise `--min-hits` if your private files share many ordinary sentences with public writing.
