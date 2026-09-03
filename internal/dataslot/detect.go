package dataslot

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	maxWalkDepth = 5
	sqliteMagic  = "SQLite format 3\x00"
)

var ignoredDirs = map[string]bool{
	".git":         true,
	"node_modules": true,
	"vendor":       true,
	".venv":        true,
	"venv":         true,
	"__pycache__":  true,
	"_scratch":     true,
	".fak":         true,
	"dist":         true,
	"build":        true,
}

// Detect walks the repository root and discovers dormant database artifacts,
// returning descriptors in dormant state with zero network connections.
func Detect(root string) ([]DataSlotDescriptor, error) {
	return DetectWorkspace(context.Background(), root)
}

// DetectWorkspace performs context-aware dormant database discovery across the workspace.
func DetectWorkspace(ctx context.Context, root string) ([]DataSlotDescriptor, error) {
	cleanRoot := filepath.Clean(root)
	info, err := os.Stat(cleanRoot)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("root path %q is not a directory", cleanRoot)
	}

	var results []DataSlotDescriptor
	seenIDs := make(map[string]bool)

	// Step 1: Scan for file databases and migration files
	walkErr := filepath.WalkDir(cleanRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip inaccessible items fail-closed
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}

		rel, err := filepath.Rel(cleanRoot, path)
		if err != nil {
			return nil
		}
		if rel == "." {
			return nil
		}

		parts := strings.Split(filepath.ToSlash(rel), "/")
		if len(parts) > maxWalkDepth {
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}

		name := d.Name()
		if d.IsDir() {
			if ignoredDirs[name] || strings.HasPrefix(name, ".") && name != "." {
				return fs.SkipDir
			}
			// Check for Alembic directory
			if name == "alembic" {
				id := "alembic:" + filepath.ToSlash(rel)
				if !seenIDs[id] {
					seenIDs[id] = true
					results = append(results, DataSlotDescriptor{
						ID:              id,
						Family:          FamilyPostgres, // Default common for Alembic, stays dormant
						Status:          StatusUnmaterialized,
						SourceArtifact:  filepath.ToSlash(rel),
						MigrationEngine: MigrationAlembic,
						MigrationPath:   filepath.ToSlash(rel),
						ReadOnly:        true,
					})
				}
			}
			// Check for generic migrations directory
			if name == "migrations" || name == "sql" {
				engine, fam := inspectMigrationsDir(cleanRoot, path)
				if engine != MigrationNone {
					id := string(engine) + ":" + filepath.ToSlash(rel)
					if !seenIDs[id] {
						seenIDs[id] = true
						results = append(results, DataSlotDescriptor{
							ID:              id,
							Family:          fam,
							Status:          StatusUnmaterialized,
							SourceArtifact:  filepath.ToSlash(rel),
							MigrationEngine: engine,
							MigrationPath:   filepath.ToSlash(rel),
							ReadOnly:        true,
						})
					}
				}
			}
			return nil
		}

		// Files: inspect candidate database and configuration files
		lowerName := strings.ToLower(name)

		// 1. Prisma schema
		if lowerName == "schema.prisma" {
			desc := parsePrismaSchema(cleanRoot, path, rel)
			if !seenIDs[desc.ID] {
				seenIDs[desc.ID] = true
				results = append(results, desc)
			}
			return nil
		}

		// 2. Alembic ini
		if lowerName == "alembic.ini" {
			id := "alembic:" + filepath.ToSlash(rel)
			if !seenIDs[id] {
				seenIDs[id] = true
				results = append(results, DataSlotDescriptor{
					ID:              id,
					Family:          FamilyPostgres,
					Status:          StatusUnmaterialized,
					SourceArtifact:  filepath.ToSlash(rel),
					MigrationEngine: MigrationAlembic,
					MigrationPath:   filepath.ToSlash(rel),
					ReadOnly:        true,
				})
			}
			return nil
		}

		// 3. Drizzle config
		if strings.HasPrefix(lowerName, "drizzle.config.") {
			id := "drizzle:" + filepath.ToSlash(rel)
			if !seenIDs[id] {
				seenIDs[id] = true
				results = append(results, DataSlotDescriptor{
					ID:              id,
					Family:          FamilyPostgres,
					Status:          StatusUnmaterialized,
					SourceArtifact:  filepath.ToSlash(rel),
					MigrationEngine: MigrationDrizzle,
					MigrationPath:   filepath.ToSlash(rel),
					ReadOnly:        true,
				})
			}
			return nil
		}

		// 3b. dbt project
		if lowerName == "dbt_project.yml" || lowerName == "dbt_project.yaml" {
			id := "dbt:" + filepath.ToSlash(rel)
			if !seenIDs[id] {
				seenIDs[id] = true
				results = append(results, DataSlotDescriptor{
					ID:             id,
					Family:         FamilyDBT,
					Status:         StatusReady,
					SourceArtifact: filepath.ToSlash(rel),
					ReadOnly:       true,
				})
			}
			return nil
		}

		// 4. Docker Compose
		if isComposeFile(lowerName) {
			composeDescs := parseComposeFile(cleanRoot, path, rel)
			for _, desc := range composeDescs {
				if !seenIDs[desc.ID] {
					seenIDs[desc.ID] = true
					results = append(results, desc)
				}
			}
			return nil
		}

		// 5. File databases (SQLite, DuckDB)
		if isDBFileExt(lowerName) {
			desc, ok := inspectDBFile(path, rel)
			if ok && !seenIDs[desc.ID] {
				seenIDs[desc.ID] = true
				results = append(results, desc)
			}
		}

		return nil
	})

	if walkErr != nil {
		return nil, walkErr
	}

	// Deterministic sort: Family ASC, then ID ASC
	sort.Slice(results, func(i, j int) bool {
		if results[i].Family != results[j].Family {
			return results[i].Family < results[j].Family
		}
		return results[i].ID < results[j].ID
	})

	return results, nil
}

func isDBFileExt(name string) bool {
	ext := filepath.Ext(name)
	switch ext {
	case ".db", ".sqlite", ".sqlite3", ".duckdb":
		return true
	default:
		return false
	}
}

func inspectDBFile(fullPath, relPath string) (DataSlotDescriptor, bool) {
	slashRel := filepath.ToSlash(relPath)
	ext := strings.ToLower(filepath.Ext(fullPath))

	if ext == ".duckdb" {
		return DataSlotDescriptor{
			ID:             "duckdb:" + slashRel,
			Family:         FamilyDuckDB,
			Status:         StatusReady,
			SourceArtifact: slashRel,
			LocalPath:      slashRel,
			ReadOnly:       true,
		}, true
	}

	// Read first 16 bytes for SQLite magic
	f, err := os.Open(fullPath)
	if err != nil {
		return DataSlotDescriptor{}, false
	}
	defer f.Close()

	header := make([]byte, 16)
	n, _ := io.ReadFull(f, header)
	if n == 16 && bytes.Equal(header, []byte(sqliteMagic)) {
		return DataSlotDescriptor{
			ID:             "sqlite:" + slashRel,
			Family:         FamilySQLite,
			Status:         StatusReady,
			SourceArtifact: slashRel,
			LocalPath:      slashRel,
			ReadOnly:       true,
		}, true
	}

	// If extension is specifically .sqlite or .sqlite3, identify as SQLite even if newly touched
	if ext == ".sqlite" || ext == ".sqlite3" {
		return DataSlotDescriptor{
			ID:             "sqlite:" + slashRel,
			Family:         FamilySQLite,
			Status:         StatusReady,
			SourceArtifact: slashRel,
			LocalPath:      slashRel,
			ReadOnly:       true,
		}, true
	}

	return DataSlotDescriptor{}, false
}

func parsePrismaSchema(root, fullPath, relPath string) DataSlotDescriptor {
	slashRel := filepath.ToSlash(relPath)
	desc := DataSlotDescriptor{
		ID:              "prisma:" + slashRel,
		Family:          FamilyPostgres, // default fallback
		Status:          StatusUnmaterialized,
		SourceArtifact:  slashRel,
		MigrationEngine: MigrationPrisma,
		MigrationPath:   filepath.ToSlash(filepath.Dir(relPath)),
		ReadOnly:        true,
	}

	f, err := os.Open(fullPath)
	if err != nil {
		return desc
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	providerRE := regexp.MustCompile(`provider\s*=\s*"([^"]+)"`)
	urlFileRE := regexp.MustCompile(`"file:(.+?)"`)

	var provider string
	var sqliteURLFile string
	inDatasource := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "datasource") {
			inDatasource = true
			continue
		}
		if inDatasource && strings.HasPrefix(line, "}") {
			inDatasource = false
			continue
		}
		if inDatasource {
			if m := providerRE.FindStringSubmatch(line); len(m) > 1 {
				provider = strings.ToLower(m[1])
			}
			if m := urlFileRE.FindStringSubmatch(line); len(m) > 1 {
				sqliteURLFile = m[1]
			}
		}
	}

	switch provider {
	case "sqlite":
		desc.Family = FamilySQLite
		if sqliteURLFile != "" {
			// Check if file exists relative to prisma schema dir
			dbTarget := filepath.Join(filepath.Dir(fullPath), sqliteURLFile)
			if _, err := os.Stat(dbTarget); err == nil {
				desc.Status = StatusReady
				if relTarget, err := filepath.Rel(root, dbTarget); err == nil {
					desc.LocalPath = filepath.ToSlash(relTarget)
				}
			}
		}
	case "postgresql", "postgres":
		desc.Family = FamilyPostgres
	case "mysql":
		desc.Family = FamilyMySQL
	}

	return desc
}

func inspectMigrationsDir(root, dirPath string) (MigrationEngine, DatabaseFamily) {
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return MigrationNone, FamilySQLite
	}

	hasGoose := false
	hasFlyway := false
	hasSQL := false

	flywayRE := regexp.MustCompile(`(?i)^V\d+__.*\.sql$`)

	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(strings.ToLower(name), ".sql") {
			hasSQL = true
			if flywayRE.MatchString(name) {
				hasFlyway = true
			}
			// Check file header for goose annotation
			if !hasGoose {
				if f, err := os.Open(filepath.Join(dirPath, name)); err == nil {
					s := bufio.NewScanner(f)
					for i := 0; i < 10 && s.Scan(); i++ {
						if strings.Contains(s.Text(), "+goose Up") {
							hasGoose = true
							break
						}
					}
					f.Close()
				}
			}
		}
	}

	if hasFlyway {
		return MigrationFlyway, FamilyPostgres
	}
	if hasGoose {
		return MigrationGoose, FamilyPostgres
	}
	if hasSQL {
		return MigrationGoose, FamilyPostgres
	}

	return MigrationNone, FamilySQLite
}

func isComposeFile(name string) bool {
	return name == "docker-compose.yml" || name == "docker-compose.yaml" ||
		name == "compose.yml" || name == "compose.yaml"
}

func parseComposeFile(root, fullPath, relPath string) []DataSlotDescriptor {
	f, err := os.Open(fullPath)
	if err != nil {
		return nil
	}
	defer f.Close()

	var descs []DataSlotDescriptor
	scanner := bufio.NewScanner(f)
	slashRel := filepath.ToSlash(relPath)

	imageRE := regexp.MustCompile(`^\s*image:\s*["']?([^"'\s]+)["']?`)

	for scanner.Scan() {
		line := scanner.Text()
		m := imageRE.FindStringSubmatch(line)
		if len(m) <= 1 {
			continue
		}

		rawImg := strings.TrimSpace(m[1])
		img := strings.ToLower(rawImg)
		baseImg := img
		if lastSlash := strings.LastIndex(baseImg, "/"); lastSlash >= 0 {
			baseImg = baseImg[lastSlash+1:]
		}

		switch {
		case strings.HasPrefix(baseImg, "postgres"):
			descs = append(descs, DataSlotDescriptor{
				ID:             fmt.Sprintf("compose:%s:postgres", slashRel),
				Family:         FamilyPostgres,
				Status:         StatusReady,
				SourceArtifact: slashRel,
				ServiceImage:   rawImg,
				Port:           5432,
				ReadOnly:       true,
			})
		case strings.HasPrefix(baseImg, "mysql"), strings.HasPrefix(baseImg, "mariadb"):
			descs = append(descs, DataSlotDescriptor{
				ID:             fmt.Sprintf("compose:%s:mysql", slashRel),
				Family:         FamilyMySQL,
				Status:         StatusReady,
				SourceArtifact: slashRel,
				ServiceImage:   rawImg,
				Port:           3306,
				ReadOnly:       true,
			})
		case strings.HasPrefix(baseImg, "redis"), strings.HasPrefix(baseImg, "valkey"):
			descs = append(descs, DataSlotDescriptor{
				ID:             fmt.Sprintf("compose:%s:redis", slashRel),
				Family:         FamilyRedis,
				Status:         StatusReady,
				SourceArtifact: slashRel,
				ServiceImage:   rawImg,
				Port:           6379,
				ReadOnly:       true,
			})
		}
	}

	return descs
}
