package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

const (
	guardArbitrateModeOff     = "off"
	guardArbitrateModeShadow  = "shadow"
	guardArbitrateModeEnforce = "enforce"
	guardArbitrateTTL         = 30 * time.Second
)

var (
	guardArbitrateShadowLimit = 2 * time.Second
	guardArbitrateLive        = func(store *leaseref.Store, ctx context.Context, now time.Time) ([]leaseref.Record, []string, error) {
		return store.Live(ctx, now)
	}
)

type guardArbitrateConfig struct {
	Mode  string
	Lane  string
	Tree  []string
	Force bool
	Root  string
	// ShowShadowNotice opts into advisory collision narration. The zero value keeps
	// successful startup clean; enforce-mode refusals remain visible as errors.
	ShowShadowNotice bool
}

type guardArbitrateFlagValue struct {
	cfg *guardArbitrateConfig
}

func (v guardArbitrateFlagValue) String() string {
	if v.cfg == nil {
		return ""
	}
	parts := []string{"mode=" + v.cfg.Mode}
	if v.cfg.Lane != "" {
		parts = append(parts, "lane="+v.cfg.Lane)
	}
	for _, tree := range v.cfg.Tree {
		parts = append(parts, "tree="+tree)
	}
	if v.cfg.Force {
		parts = append(parts, "force=true")
	}
	return strings.Join(parts, ",")
}

func (v guardArbitrateFlagValue) Set(raw string) error {
	if v.cfg == nil {
		return errors.New("nil guard lease config")
	}
	for _, field := range strings.Split(raw, ",") {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		key, value, ok := strings.Cut(field, "=")
		if !ok {
			return fmt.Errorf("invalid lease field %q (want key=value)", field)
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		switch key {
		case "mode":
			mode, err := normalizeGuardArbitrateMode(value)
			if err != nil {
				return err
			}
			v.cfg.Mode = mode
		case "lane":
			v.cfg.Lane = value
		case "tree":
			if value == "" {
				return errors.New("empty lease tree")
			}
			v.cfg.Tree = append(v.cfg.Tree, value)
		case "force":
			force, err := strconv.ParseBool(value)
			if err != nil {
				return fmt.Errorf("invalid lease force %q: %w", value, err)
			}
			v.cfg.Force = force
		default:
			return fmt.Errorf("unknown lease field %q (want mode, lane, tree, or force)", key)
		}
	}
	return nil
}

type guardArbitrateLease struct {
	store       *leaseref.Store
	record      leaseref.Record
	stop        chan struct{}
	done        chan struct{}
	releaseOnce sync.Once
}

func normalizeGuardArbitrateMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", guardArbitrateModeShadow:
		return guardArbitrateModeShadow, nil
	case guardArbitrateModeOff:
		return guardArbitrateModeOff, nil
	case guardArbitrateModeEnforce:
		return guardArbitrateModeEnforce, nil
	default:
		return "", fmt.Errorf("invalid lease mode %q (want off, shadow, or enforce)", raw)
	}
}

func guardArbitrateAcquire(ctx context.Context, stderr io.Writer, cfg guardArbitrateConfig) (*guardArbitrateLease, error) {
	mode, err := normalizeGuardArbitrateMode(cfg.Mode)
	if err != nil {
		return nil, err
	}
	if mode == guardArbitrateModeOff {
		return nil, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if mode == guardArbitrateModeShadow {
		// The startup report was only the trigger that made #9656 look like a Codex
		// hang. The root was the default shadow admission walking 117 non-session
		// refs with one `git cat-file blob` process per ref (10.4s while otherwise
		// quiet, about 97s under host load). Shadow advice cannot sit indefinitely
		// between that report and exec, so bound the complete admission operation.
		// Enforce deliberately retains the caller's context and fails closed below.
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, guardArbitrateShadowLimit)
		defer cancel()
	}
	root := strings.TrimSpace(cfg.Root)
	if root == "" {
		wd, wdErr := os.Getwd()
		if wdErr != nil {
			fmt.Fprintf(stderr, "fak guard: arbitrate fail-open; working tree unavailable: %v\n", wdErr)
			return nil, nil
		}
		root = findRepoRoot(wd)
	}
	root, err = filepath.Abs(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard: arbitrate fail-open; working tree unavailable: %v\n", err)
		return nil, nil
	}
	tax, err := regionadmit.LoadTaxonomy(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard: arbitrate fail-open; lane taxonomy unavailable: %v\n", err)
		return nil, nil
	}
	requestTree := cleanGuardArbitrateTree(cfg.Tree)
	if len(requestTree) == 0 && strings.TrimSpace(cfg.Lane) == "" {
		requestTree = []string{"**/*"}
	}
	commonDir, err := resolveGitCommonDir(root)
	if err != nil {
		if mode == guardArbitrateModeEnforce {
			return nil, fmt.Errorf("COLLISION_RISK: guard admission git directory unavailable: %w", err)
		}
		fmt.Fprintf(stderr, "fak guard: arbitrate fail-open; git directory unavailable: %v\n", err)
		return nil, nil
	}
	store := leaseref.NewInDir(root)
	lock, err := gpulease.Acquire(gpulease.Options{Path: filepath.Join(commonDir, "fak-guard-arbitrate.lock"), Timeout: 2 * time.Second, Logf: func(string, ...any) {}})
	if err != nil {
		if errors.Is(err, gpulease.ErrBusy) || errors.Is(err, gpulease.ErrTimeout) {
			if mode == guardArbitrateModeShadow {
				if cfg.ShowShadowNotice {
					fmt.Fprintf(stderr, "fak guard: arbitrate shadow would refuse: COLLISION_RISK admission serialization is busy: %v\n", err)
				}
				return nil, nil
			}
			return nil, fmt.Errorf("COLLISION_RISK: guard admission serialization is busy: %w", err)
		}
		if mode == guardArbitrateModeEnforce {
			return nil, fmt.Errorf("COLLISION_RISK: guard admission serialization is unavailable: %w", err)
		}
		fmt.Fprintf(stderr, "fak guard: arbitrate fail-open; admission lock unavailable: %v\n", err)
		return nil, nil
	}
	unlock := func() { lock.Release() }
	lockOwned := true
	defer func() {
		if lockOwned {
			unlock()
		}
	}()
	live, _, err := guardArbitrateLive(store, ctx, time.Now())
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			if mode == guardArbitrateModeEnforce {
				return nil, fmt.Errorf("COLLISION_RISK: guard lease admission could not read the live ledger: %w", ctxErr)
			}
			fmt.Fprintf(stderr, "fak guard: lease admission timed out after %s; shadow mode continuing without a lease\n", guardArbitrateShadowLimit)
			return nil, nil
		}
		fmt.Fprintf(stderr, "fak guard: arbitrate fail-open; live lease ledger unavailable: %v\n", err)
		return nil, nil
	}
	req := regionadmit.Request{
		Actor: guardArbitrateHolder(),
		Lane:  strings.TrimSpace(cfg.Lane),
		Tree:  requestTree,
	}
	dec := regionadmit.Decide(req, regionLeases(live), tax)
	if !dec.Admit && cfg.Force && dec.Rung != regionadmit.RungExclusiveLive {
		dec = regionadmit.Decision{Admit: true}
	}
	if !dec.Admit {
		conflict := "unknown"
		if dec.Conflict != nil {
			conflict = dec.Conflict.ID
		}
		if mode == guardArbitrateModeShadow {
			if cfg.ShowShadowNotice {
				fmt.Fprintf(stderr, "fak guard: arbitrate shadow would refuse %s: %s conflict=%s detail=%s\n", regionLabel(req, tax), dec.Reason, conflict, dec.Detail)
			}
			return nil, nil
		}
		return nil, fmt.Errorf("%s: conflicting lease %s: %s", dec.Reason, conflict, dec.Detail)
	}
	if mode == guardArbitrateModeShadow {
		return nil, nil
	}

	// Re-read immediately before publishing. This narrows the decision/write window and,
	// after publication, the read-back below makes a same-window contender converge on one
	// deterministic winner instead of allowing two overlapping guard leases to survive.
	live, _, err = guardArbitrateLive(store, ctx, time.Now())
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, fmt.Errorf("COLLISION_RISK: guard lease admission could not re-read the live ledger: %w", ctxErr)
		}
		fmt.Fprintf(stderr, "fak guard: arbitrate fail-open; live lease ledger unavailable before acquire: %v\n", err)
		return nil, nil
	}
	dec = regionadmit.Decide(req, regionLeases(live), tax)
	if !dec.Admit && !(cfg.Force && dec.Rung != regionadmit.RungExclusiveLive) {
		conflict := "unknown"
		if dec.Conflict != nil {
			conflict = dec.Conflict.ID
		}
		return nil, fmt.Errorf("%s: conflicting lease %s: %s", dec.Reason, conflict, dec.Detail)
	}

	id := guardArbitrateLeaseID()
	rec := leaseref.Record{
		ID:          id,
		TreeGlobs:   append([]string(nil), regionadmit.ResolveTree(req, tax)...),
		Holder:      req.Actor,
		AcquiredAt:  time.Now().Unix(),
		TTLSeconds:  int64(guardArbitrateTTL / time.Second),
		Description: "fak guard lifecycle lease",
	}
	if _, err := store.Acquire(ctx, rec); err != nil {
		fmt.Fprintf(stderr, "fak guard: arbitrate fail-open; lease publish unavailable: %v\n", err)
		return nil, nil
	}
	lease := &guardArbitrateLease{store: store, record: rec, stop: make(chan struct{}), done: make(chan struct{})}
	if err := lease.confirmWinner(ctx, req, tax); err != nil {
		lease.release(context.Background())
		return nil, err
	}
	unlock()
	lockOwned = false
	go lease.renew()
	return lease, nil
}

func (g *guardArbitrateLease) confirmWinner(ctx context.Context, req regionadmit.Request, tax regionadmit.Taxonomy) error {
	live, _, err := g.store.Live(ctx, time.Now())
	if err != nil {
		return fmt.Errorf("confirm guard lease: %w", err)
	}
	selfFound := false
	for _, r := range live {
		if r.ID == g.record.ID {
			selfFound = true
		}
	}
	if !selfFound {
		return errors.New("guard lease publication was not visible on read-back")
	}
	// If two launchers raced through the pre-publish read, the lexically smaller lease id
	// wins. Every contender computes the same order from the shared ledger.
	sort.Slice(live, func(i, j int) bool { return live[i].ID < live[j].ID })
	for _, r := range live {
		if r.ID == g.record.ID {
			continue
		}
		dec := regionadmit.Decide(regionadmit.Request{Actor: req.Actor, Lane: req.Lane, Tree: req.Tree, SelfID: g.record.ID}, []regionadmit.Lease{regionLeases([]leaseref.Record{r})[0]}, tax)
		if !dec.Admit && r.ID < g.record.ID {
			return fmt.Errorf("%s: conflicting lease %s won concurrent guard admission: %s", dec.Reason, r.ID, dec.Detail)
		}
	}
	return nil
}

func (g *guardArbitrateLease) Close() {
	if g == nil {
		return
	}
	g.releaseOnce.Do(func() {
		close(g.stop)
		<-g.done
		g.release(context.Background())
	})
}

func (g *guardArbitrateLease) release(ctx context.Context) {
	_, _ = g.store.ReleaseFenced(ctx, g.record.ID, g.record.Holder, g.record.Generation, time.Now())
}

func (g *guardArbitrateLease) renew() {
	defer close(g.done)
	ticker := time.NewTicker(guardArbitrateTTL / 3)
	defer ticker.Stop()
	for {
		select {
		case <-g.stop:
			return
		case now := <-ticker.C:
			rec, verdict, err := g.store.Renew(context.Background(), g.record.ID, g.record.Holder, int64(guardArbitrateTTL/time.Second), now)
			if err == nil && verdict.OK {
				g.record = rec
			}
		}
	}
}

func cleanGuardArbitrateTree(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, raw := range in {
		v := filepath.ToSlash(strings.TrimSpace(raw))
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

func guardArbitrateHolder() string {
	host, _ := os.Hostname()
	if strings.TrimSpace(host) == "" {
		host = "host"
	}
	return fmt.Sprintf("guard:%s:%d", host, os.Getpid())
}

func guardArbitrateLeaseID() string {
	return "guard-" + strconv.Itoa(os.Getpid()) + "-" + strings.ToLower(slackoutbox.NewNonce()[:12])
}

func resolveGitCommonDir(root string) (string, error) {
	dot := filepath.Join(root, ".git")
	fi, err := os.Stat(dot)
	if err != nil {
		return "", err
	}
	if fi.IsDir() {
		return filepath.Clean(dot), nil
	}
	b, err := os.ReadFile(dot)
	if err != nil {
		return "", err
	}
	line := strings.TrimSpace(string(b))
	rest, ok := strings.CutPrefix(line, "gitdir:")
	if !ok {
		return "", fmt.Errorf("%s is not a gitdir pointer: %q", dot, line)
	}
	gitDir := strings.TrimSpace(rest)
	if gitDir == "" {
		return "", fmt.Errorf("%s contains empty gitdir pointer", dot)
	}
	if !filepath.IsAbs(gitDir) {
		gitDir = filepath.Join(root, gitDir)
	}
	gitDir = filepath.Clean(gitDir)

	commonBytes, err := os.ReadFile(filepath.Join(gitDir, "commondir"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return gitDir, nil
		}
		return "", err
	}
	common := strings.TrimSpace(string(commonBytes))
	if common == "" {
		return gitDir, nil
	}
	if !filepath.IsAbs(common) {
		common = filepath.Join(gitDir, common)
	}
	return filepath.Clean(common), nil
}
