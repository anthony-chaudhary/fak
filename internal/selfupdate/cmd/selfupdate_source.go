package selfupdatecmd

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

type selfUpdateSourceOptions struct {
	Ref, Revision       string
	RefSet, RevisionSet bool
}

type selfUpdateSourceSelection struct {
	Selector string
	Revision string
}

func selfUpdateSourceFlags(fs *flag.FlagSet, ref, revision, installer, configInstaller string) (selfUpdateSourceOptions, error) {
	opts := selfUpdateSourceOptions{Ref: ref, Revision: revision}
	conflict := ""
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "ref":
			opts.RefSet = true
		case "revision":
			opts.RevisionSet = true
		default:
			if strings.HasPrefix(f.Name, "manifest-") || strings.HasPrefix(f.Name, "msix-") || f.Name == "build-gc" {
				conflict = "--" + f.Name
			}
		}
	})
	if !opts.RefSet && !opts.RevisionSet {
		return opts, nil
	}
	if opts.RefSet && opts.RevisionSet {
		return opts, fmt.Errorf("SOURCE_CONFLICT: --ref and --revision are mutually exclusive")
	}
	if selected, err := selfupdate.ResolveInstaller(installer, configInstaller); err != nil || selected != selfupdate.InstallerNative {
		conflict = "the selected installer"
	}
	if conflict != "" {
		return opts, fmt.Errorf("SOURCE_CONFLICT: --ref/--revision cannot be combined with %s", conflict)
	}
	if opts.RevisionSet {
		if !isFullGitCommit(opts.Revision) {
			return opts, fmt.Errorf("SOURCE_INVALID: --revision requires exactly one full 40-hex commit (no abbreviated or dirty revision)")
		}
	} else if opts.Ref == "" || opts.Ref != strings.TrimSpace(opts.Ref) || strings.HasPrefix(opts.Ref, "-") {
		return opts, fmt.Errorf("SOURCE_INVALID: --ref requires a non-empty origin branch or tag")
	}
	return opts, nil
}

// Resolve once before comparison or admission. Every later stage consumes Revision,
// never the mutable ref, and --check observes only the already available Git objects.
func resolveSelfUpdateSource(ctx context.Context, run selfinstall.Runner, root string, opts selfUpdateSourceOptions, manifest selfUpdateManifestSelection, check bool) (selfUpdateSourceSelection, error) {
	source := selfUpdateSourceSelection{Selector: opts.Ref}
	switch {
	case manifest.Disposition == "update":
		source.Selector, source.Revision = "signed-manifest", manifest.TargetRevision
		if manifest.Artifact == nil {
			selfUpdateFetchOrigin(ctx, run, root, check)
		}
	case opts.RevisionSet:
		source.Selector = opts.Revision
		source.Revision = selfUpdateSourceCommit(ctx, run, root, strings.ToLower(opts.Revision))
		if source.Revision == "" && !check {
			if _, ok := run(ctx, root, "git", "fetch", "--quiet", "--no-tags", "origin", strings.ToLower(opts.Revision)); !ok {
				return source, fmt.Errorf("SOURCE_UNRESOLVED: cannot fetch revision %q from origin", opts.Revision)
			}
			source.Revision = selfUpdateSourceCommit(ctx, run, root, strings.ToLower(opts.Revision))
		}
		// An annotated tag object's full hash is not itself a commit pin.
		if !strings.EqualFold(source.Revision, opts.Revision) {
			return source, fmt.Errorf("SOURCE_UNRESOLVED: revision %q does not identify an available commit", opts.Revision)
		}
	case opts.RefSet:
		var err error
		source.Revision, err = resolveSelfUpdateRef(ctx, run, root, opts.Ref, check)
		if err != nil {
			return source, err
		}
	default:
		selfUpdateFetchOrigin(ctx, run, root, check)
		source.Revision = selfUpdateSourceCommit(ctx, run, root, opts.Ref)
	}
	if !isFullGitCommit(source.Revision) && (!check || opts.RefSet || opts.RevisionSet) {
		return source, fmt.Errorf("SOURCE_UNRESOLVED: cannot resolve source %q to a full commit in %s", source.Selector, root)
	}
	return source, nil
}

func selfUpdateSourceCommit(ctx context.Context, run selfinstall.Runner, root, ref string) string {
	out, ok := run(ctx, root, "git", "rev-parse", "--verify", "--quiet", "--end-of-options", ref+"^{commit}")
	if revision := strings.TrimSpace(out); ok && isFullGitCommit(revision) {
		return strings.ToLower(revision)
	}
	return ""
}

type selfUpdateSourceRef struct {
	remote, local string
}

func selfUpdateSourceRefs(ref string) ([]selfUpdateSourceRef, error) {
	branch := func(name string) selfUpdateSourceRef {
		return selfUpdateSourceRef{"refs/heads/" + name, "refs/remotes/origin/" + name}
	}
	tag := func(name string) selfUpdateSourceRef {
		return selfUpdateSourceRef{"refs/tags/" + name, "refs/tags/" + name}
	}
	for _, prefix := range []string{"refs/remotes/origin/", "origin/", "refs/heads/"} {
		if name, ok := strings.CutPrefix(ref, prefix); ok {
			return []selfUpdateSourceRef{branch(name)}, nil
		}
	}
	for _, prefix := range []string{"refs/tags/", "tags/"} {
		if name, ok := strings.CutPrefix(ref, prefix); ok {
			return []selfUpdateSourceRef{tag(name)}, nil
		}
	}
	if strings.HasPrefix(ref, "refs/") {
		return nil, fmt.Errorf("SOURCE_INVALID: --ref %q must name an origin branch or tag", ref)
	}
	return []selfUpdateSourceRef{branch(ref), tag(ref)}, nil
}

func resolveSelfUpdateRef(ctx context.Context, run selfinstall.Runner, root, ref string, check bool) (string, error) {
	candidates, err := selfUpdateSourceRefs(ref)
	if err != nil {
		return "", err
	}
	for _, candidate := range candidates {
		if _, ok := run(ctx, root, "git", "check-ref-format", candidate.remote); !ok {
			return "", fmt.Errorf("SOURCE_INVALID: --ref %q is not an origin branch or tag name", ref)
		}
		// A symbolic ref (including origin/HEAD or a tag alias) is not a concrete
		// source. Never resolve it, or let a fetch follow a symbolic destination.
		if _, symbolic := run(ctx, root, "git", "symbolic-ref", "--quiet", candidate.local); symbolic {
			return "", fmt.Errorf("SOURCE_INVALID: ref %q is a symbolic alias; select a concrete origin branch or tag", ref)
		}
	}
	if check {
		revision := ""
		for _, candidate := range candidates {
			if resolved := selfUpdateSourceCommit(ctx, run, root, candidate.local); resolved != "" {
				if revision != "" {
					return "", fmt.Errorf("SOURCE_CONFLICT: ref %q names both an origin branch and a tag; qualify the ref", ref)
				}
				revision = resolved
			}
		}
		if revision == "" {
			return "", fmt.Errorf("SOURCE_UNRESOLVED: ref %q is not available locally (--check does not fetch)", ref)
		}
		return revision, nil
	}
	selected := candidates[0]
	if len(candidates) > 1 {
		selected, err = selectSelfUpdateOriginRef(ctx, run, root, ref, candidates)
		if err != nil {
			return "", err
		}
	}
	refspec := selected.remote + ":" + selected.local
	if strings.HasPrefix(selected.remote, "refs/heads/") {
		refspec = "+" + refspec // Match normal remote-tracking branch fetch semantics.
	}
	if _, ok := run(ctx, root, "git", "fetch", "--quiet", "--no-tags", "origin", refspec); !ok {
		return "", fmt.Errorf("SOURCE_UNRESOLVED: cannot fetch ref %q from origin", ref)
	}
	revision := selfUpdateSourceCommit(ctx, run, root, selected.local)
	if revision == "" {
		return "", fmt.Errorf("SOURCE_UNRESOLVED: ref %q does not resolve to a commit", ref)
	}
	return revision, nil
}

func selectSelfUpdateOriginRef(ctx context.Context, run selfinstall.Runner, root, ref string, candidates []selfUpdateSourceRef) (selfUpdateSourceRef, error) {
	args := []string{"ls-remote", "--refs", "origin"}
	for _, candidate := range candidates {
		args = append(args, candidate.remote)
	}
	out, ok := run(ctx, root, "git", args...)
	if !ok {
		return selfUpdateSourceRef{}, fmt.Errorf("SOURCE_UNRESOLVED: cannot find ref %q on origin", ref)
	}
	selected := selfUpdateSourceRef{}
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 || !isFullGitCommit(fields[0]) {
			continue
		}
		for _, candidate := range candidates {
			if fields[1] == candidate.remote {
				if selected.remote != "" {
					return selfUpdateSourceRef{}, fmt.Errorf("SOURCE_CONFLICT: ref %q names both an origin branch and a tag; qualify the ref", ref)
				}
				selected = candidate
			}
		}
	}
	if selected.remote == "" {
		return selected, fmt.Errorf("SOURCE_UNRESOLVED: ref %q is absent from origin", ref)
	}
	return selected, nil
}
