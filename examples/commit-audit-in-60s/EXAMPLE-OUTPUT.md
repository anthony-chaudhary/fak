# Example output

This is a representative captured run of `dos commit-audit` against the throwaway
repository built by `run.sh`. The commit SHAs are generated inside the temporary
repo and will differ on each run; the important witness is the verdict and exit
code pair.

```text
== 1/2 THE OVER-CLAIM ==
UNWITNESSED 2cdd383  [subject-only]  code-effect claim but the diff touches no SOURCE file (only: README.md) - the claim rests on the subject text
exit=1

commit-audit: 1/1 commit(s) make a claim their diff does not witness.

== 2/2 THE HONEST COMMIT ==
witnessed   c4c8495  [diff-witnessed]  code-effect claim witnessed by a touched source file
exit=0
```
