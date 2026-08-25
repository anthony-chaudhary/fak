package agentreadinessscore

import (
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// gather reads the git-tracked tree and runs every pure KPI.
func gather(root string) ([]KPI, map[string]int) {
	inputs := readGatherInputs(root)
	trackedList, present := inputs.trackedList, inputs.present
	texts, pasteTexts := inputs.texts, inputs.pasteTexts
	agentsText := texts[agentsFile]

	// discover
	configPresent := map[string]bool{}
	for _, c := range agentConfigs {
		configPresent[c.label] = anyPresent(present, c.paths)
	}
	llmsPresent := present(llmsFile)
	llmsFullPresent := present(llmsFullFile)
	idPresent, idMissing := findIdentity(texts)
	var deadLinks []string
	for _, doc := range orientationDocs {
		if !present(doc) {
			continue
		}
		for _, l := range localLinks(safeRead(root, doc), doc, root) {
			if !l.exists {
				deadLinks = append(deadLinks, doc+" -> "+l.target)
			}
		}
	}

	// adopt
	fcFound, fcWhere := findFirstCommand(texts)
	instFound, instWhere := findInstallOneliner(texts)
	claimsPresent := present(claimsFile)
	claimsText := ""
	if claimsPresent {
		claimsText = safeRead(root, claimsFile)
	}
	untagged := untaggedClaims(claimsText, claimsPresent)
	recipePresent := map[string]bool{}
	for _, r := range requiredRecipes {
		recipePresent[r.label] = anyPresent(present, r.paths)
	}
	codexPresent := present(codexFile)
	codexText := ""
	if codexPresent {
		codexText = safeRead(root, codexFile)
	}

	// build
	scaffold := present(leafScaffold)
	extending := present(extendingFile)
	contributing := present(contribFile)
	contributingLinked := has(agentsText, contribFile) || has(texts["README.md"], contribFile)
	greenGate := has(agentsText, greenGateTokens...)
	guardMissing := missingGuardrails(agentsText)

	var toolFiles []string
	toolRe := regexp.MustCompile(`tools/[^/]*scorecard[^/]*\.py$`)
	for _, f := range trackedList {
		if toolRe.MatchString(f) && !strings.HasSuffix(f, "_test.py") {
			toolFiles = append(toolFiles, f)
		}
	}
	sort.Strings(toolFiles)
	jsonTools := 0
	var missingJSON []string
	for _, f := range toolFiles {
		if toolHasJSON(safeRead(root, f)) {
			jsonTools++
		} else {
			missingJSON = append(missingJSON, f)
		}
	}

	// paste-and-run success facts
	badPaths := badFencedPaths(pasteTexts, root)
	fcrFound, fcrPolicyOK, fcrPolicyRef, fcrNeedsKey := firstCommandFacts(texts, root)
	sellsMake := has(agentsText, "make ci")
	windowsBridge := has(agentsText, windowsBridgeTokens...)

	// executable-truth facts
	mainGoText := ""
	if present(mainGo) {
		mainGoText = safeRead(root, mainGo)
	}
	verbs := dispatchVerbs(mainGoText)
	unknownVerbs := unknownCommandVerbs(pasteTexts, verbs)
	recipeTexts := map[string]string{}
	for rel, txt := range pasteTexts {
		if strings.HasPrefix(rel, pasteDocsGlob+"/") && !strings.HasSuffix(rel, "/README.md") {
			recipeTexts[rel] = txt
		}
	}
	for _, r := range requiredRecipes {
		for _, p := range r.paths {
			if strings.HasSuffix(p, ".md") {
				if _, ok := recipeTexts[p]; !ok && present(p) {
					recipeTexts[p] = safeRead(root, p)
				}
			}
		}
	}
	deadRecipe := deadRecipeLinks(root, recipeTexts)
	configIntegrity := agentConfigIntegrity(root)

	// hardened bar
	dosText := ""
	if present(dosToml) {
		dosText = safeRead(root, dosToml)
	}
	reasonTokens := dosReasonTokens(dosText)
	var recoveryParts []string
	for _, d := range refusalRecoveryDocs {
		if present(d) {
			recoveryParts = append(recoveryParts, safeRead(root, d))
		}
	}
	recoveryText := strings.Join(recoveryParts, "\n")
	unmappedReasons := queryBackedRefusalRecovery(reasonTokens, dosText, recoveryText)
	qsFound, qsSignal := quickstartSignal(texts)
	goModText := ""
	if present(goModFile) {
		goModText = safeRead(root, goModFile)
	}
	goDirective := goDirectiveRe.MatchString(goModText)
	goDocNamed := false
	for _, d := range toolchainDocs {
		if goVersDocRe.MatchString(texts[d]) {
			goDocNamed = true
			break
		}
	}

	kpis := []KPI{
		kpiAgentsEntrypoint(agentsText, present(agentsFile)),
		kpiAgentConfig(missingAgentConfigs(configPresent)),
		kpiLLMSMap(llmsPresent, llmsFullPresent),
		kpiIdentityStatement(idPresent, idMissing),
		kpiEntryLinksResolve(deadLinks),
		kpiFirstCommand(fcFound, fcWhere),
		kpiFirstCommandRuns(fcrFound, fcrPolicyOK, fcrPolicyRef, fcrNeedsKey),
		kpiInstallOneliner(instFound, instWhere),
		kpiHonestyLedger(claimsPresent, untagged),
		kpiIntegrationRecipes(missingRecipes(recipePresent)),
		kpiCodexRecipeCurrent(codexRecipeGaps(codexText, codexPresent)),
		kpiFencedPathsResolve(badPaths),
		kpiCommandVerbsResolve(unknownVerbs),
		kpiRecipeLinksResolve(deadRecipe),
		kpiAgentConfigValid(configIntegrity),
		kpiExtensionScaffold(scaffold, extending),
		kpiGuardrailsSurfaced(guardMissing),
		kpiContributorContract(contributing, contributingLinked, greenGate),
		kpiPlatformGuidanceConsistent(sellsMake, windowsBridge),
		kpiMachineConsumable(jsonTools, len(toolFiles), missingJSON),
		kpiRefusalRecoveryMapped(unmappedReasons, len(reasonTokens)),
		kpiQuickstartSuccessSignal(qsFound, qsSignal),
		kpiToolchainPinned(goDirective, goDocNamed),
		launchEntryContractKPI(root),
	}

	substantiveConfigs := 0
	for _, cfg := range frontierHarnessConfigs {
		var chosen string
		for _, p := range cfg.paths {
			if present(p) {
				chosen = p
				break
			}
		}
		if chosen != "" && len(strings.TrimSpace(safeRead(root, chosen))) >= configMinChars {
			substantiveConfigs++
		}
	}
	substantiveRecipes := 0
	for _, txt := range recipeTexts {
		if isSubstantiveRecipe(txt) {
			substantiveRecipes++
		}
	}
	facts := map[string]int{
		"integration_recipes": substantiveRecipes,
		"harness_configs":     substantiveConfigs,
		"refusal_recoveries":  mathx.MaxInt(0, len(reasonTokens)-len(unmappedReasons)),
		"machine_consumable":  jsonTools,
	}
	return kpis, facts
}

type gatherInputs struct {
	trackedList []string
	present     func(string) bool
	texts       map[string]string
	pasteTexts  map[string]string
}

func readGatherInputs(root string) gatherInputs {
	trackedList := gitLines([]string{"ls-files"}, root)
	tracked := map[string]bool{}
	for _, f := range trackedList {
		tracked[f] = true
	}
	present := func(rel string) bool {
		return tracked[rel] || fileExists(root, rel)
	}

	readDocs := map[string]bool{}
	for _, set := range [][]string{identityDocs, firstCommandDocs, installDocs, orientationDocs, pasteDocs, {agentsFile, "README.md"}} {
		for _, d := range set {
			readDocs[d] = true
		}
	}
	texts := map[string]string{}
	for d := range readDocs {
		texts[d] = safeRead(root, d)
	}

	pasteTexts := map[string]string{}
	for _, d := range pasteDocs {
		pasteTexts[d] = texts[d]
	}
	if matches, _ := filepath.Glob(filepath.Join(root, filepath.FromSlash(pasteDocsGlob), "*.md")); matches != nil {
		for _, md := range matches {
			relOS, err := filepath.Rel(root, md)
			if err != nil {
				continue
			}
			rel := filepath.ToSlash(relOS)
			pasteTexts[rel] = safeRead(root, rel)
		}
	}
	return gatherInputs{
		trackedList: trackedList,
		present:     present,
		texts:       texts,
		pasteTexts:  pasteTexts,
	}
}
