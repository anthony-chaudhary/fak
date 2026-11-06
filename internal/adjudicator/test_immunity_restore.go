package adjudicator

import "strings"

// isHeadRestoreCommand reports whether cmd is exactly one git invocation whose
// only effect is restoring paths to their committed HEAD content (#14209):
//
//	git restore [--worktree|-W] [--staged|-S] [-q] (--source=HEAD|--source HEAD|-s HEAD) [--] <paths>
//	git restore --staged [--worktree] [-q] [--] <paths>   (source defaults to HEAD)
//	git checkout [-q] HEAD -- <paths>
//
// Restoring from HEAD cannot author new test content, so it is the recovery
// path for a corrupted test file. Fail closed: any other source (a sha,
// HEAD~1, a branch, a stash, the index), any other flag, git global options,
// or any shell composition/expansion is not a HEAD restore.
func isHeadRestoreCommand(cmd string) bool {
	if strings.ContainsAny(cmd, ";&|<>`$(){}\n\r*?[]~!\\") {
		return false
	}
	fields := strings.Fields(cmd)
	if len(fields) < 3 || fields[0] != "git" {
		return false
	}
	for i := range fields {
		f := fields[i]
		if len(f) >= 2 && (f[0] == '"' || f[0] == '\'') && f[len(f)-1] == f[0] {
			f = f[1 : len(f)-1]
		}
		if strings.ContainsAny(f, `"'`) || f == "" {
			return false
		}
		fields[i] = f
	}
	switch fields[1] {
	case "restore":
		return isHeadRestoreArgs(fields[2:])
	case "checkout":
		return isHeadCheckoutArgs(fields[2:])
	}
	return false
}

func isHeadRestoreArgs(args []string) bool {
	staged := false
	source := ""
	var paths []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--":
			paths = append(paths, args[i+1:]...)
			i = len(args)
		case a == "--worktree" || a == "-W" || a == "-q" || a == "--quiet":
		case a == "--staged" || a == "-S" || a == "-SW" || a == "-WS":
			staged = true
		case a == "--source" || a == "-s":
			if i+1 >= len(args) || source != "" {
				return false
			}
			source = args[i+1]
			i++
		case strings.HasPrefix(a, "--source="):
			if source != "" {
				return false
			}
			source = strings.TrimPrefix(a, "--source=")
		case strings.HasPrefix(a, "-"):
			return false
		default:
			paths = append(paths, a)
		}
	}
	if len(paths) == 0 {
		return false
	}
	for _, p := range paths {
		if strings.HasPrefix(p, ":") {
			return false
		}
	}
	if source == "" {
		// Without --source, git restores the worktree from the index, which may
		// hold staged tampering; only --staged defaults the source to HEAD.
		return staged
	}
	return source == "HEAD"
}

func isHeadCheckoutArgs(args []string) bool {
	i := 0
	for i < len(args) && (args[i] == "-q" || args[i] == "--quiet") {
		i++
	}
	if len(args) < i+3 {
		return false
	}
	if args[i] != "HEAD" || args[i+1] != "--" {
		return false
	}
	for _, p := range args[i+2:] {
		if strings.HasPrefix(p, ":") {
			return false
		}
	}
	return true
}
