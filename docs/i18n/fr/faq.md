---
title: "fak — FAQ de premier contact (introduction en français / French introduction)"
description: "FAQ de premier contact pour fak : un binaire Go placé devant votre agent IA qui vérifie chaque tool call avant exécution et réutilise le travail des sessions longues — plus sûr, moins cher, plus rapide ; auto-hébergement compatible RGPD et AI Act européen, Apache-2.0."
---

# fak — FAQ de premier contact

> Ceci est une **page d'entrée localisée (entry point)**, pas une traduction complète de
> la documentation. La documentation canonique est en anglais — cette page répond aux
> premières questions puis vous renvoie vers la
> [documentation anglaise](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Avertissement :** cette page est générée automatiquement et attend une relecture
> native — les corrections par issue/PR sont bienvenues.
>
> Voir aussi : [l'introduction en français](./README.md) — toutes les langues sur le
> [hub i18n](../README.md).

## Q1. Qu'est-ce que fak ?

Un seul binaire Go statique que vous placez **devant l'agent IA que vous utilisez déjà**
(Claude Code, Codex, Cursor, tout client OpenAI/Anthropic/MCP), en repointant une seule
base URL, sans réécriture. Il rend les sessions longues moins chères (il évacue les
vieux tours tout en gardant le préfixe du prompt-cache du provider identique octet par
octet), route chaque tool call, peut exécuter des modèles GGUF locaux in-process, et
enregistre un verdict auditable pour chaque appel.

## Q2. Dois-je changer de modèle ou réécrire mon agent ?

Non. fak **gouverne et met en cache** le modèle que vous utilisez déjà. Vous
l'enveloppez avec :

```bash
fak manage claude
```

ou vous repointez une seule base URL vers `fak serve`.

## Q3. Où vont mes données — est-ce conforme ?

Auto-hébergement d'abord : un seul binaire statique devant un modèle local ou un provider
national, avec une résidence des données **fail-closed**, un capability floor en
**default-deny**, et un journal d'audit inviolable pour chaque tool call. Vos données ne
quittent pas votre machine.

C'est la brique technique qu'un audit **RGPD** demande (la donnée reste sur votre
infrastructure, aucun chemin « transféré par défaut vers un pays tiers »), et le journal
d'audit append-only chaîné par hachage répond au journal de décisions attendu par
l'**AI Act européen**. Ce n'est pas un conseil juridique. Détails :
[Data residency & compliance](../../explainers/data-residency-and-compliance.md).

## Q4. Combien ça coûte ? Est-ce vraiment gratuit ?

**Apache-2.0, gratuit, auto-hébergé.** Pas de carte bancaire, pas de facture
transfrontalière, pas d'entité juridique — donc zéro euro de licence et aucun contrat
fournisseur à signer. `git clone` puis `go install`, c'est tout le chemin.

## Q5. À quel point est-ce moins cher ou plus rapide ?

Sur un run mesuré de **50 tours × 5 agents**, environ **4,1× moins de travail** qu'une
stack warm-cache **optimisée** — et le travail évité, ce sont des tokens que vous ne payez
pas, donc directement un levier de marge en euros. Le chiffre de **~60×** (environ
19 heures ramenées à environ 19 minutes) ne tient **que** face à une boucle naïve qui
renvoie tout à chaque tour — jamais comme chiffre de tête. Le gain de réutilisation est
réservé à l'auto-hébergement, pour les flottes à lecture intensive.

## Q6. Quels modèles fonctionnent ?

**Qwen2/Qwen3 et GLM-MoE** sont prouvés bit-exact dans le moteur de référence in-kernel.
Tout le reste (DeepSeek, Mistral, n'importe quel modèle open-weights) est fronté via
l'interface compatible OpenAI : Ollama / vLLM / SGLang / llama.cpp / LM Studio.

## Q7. Comment bloque-t-il la prompt injection ?

Deux barrières **structurelles**, pas un classifieur : un capability floor en default-deny
(un outil dangereux n'est jamais sur l'allow-list) et la mise en quarantaine des résultats
(les résultats d'outils empoisonnés n'atteignent jamais le contexte du modèle). En tests
réels, l'injection a atteint la baseline non protégée **5/5** et fak l'a murée **5/5**.

## Q8. Comment l'installer ?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

ou, depuis un clone :

```bash
go build -o fak ./cmd/fak
```

Si `proxy.golang.org` est inaccessible depuis votre réseau, réglez un miroir de modules
via `GOPROXY` avant d'installer.

## La preuve en 60 secondes (pas de clé, pas de modèle, pas de GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q9. Où aller ensuite ?

- [README (la vue d'ensemble complète)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — un modèle local en 10 minutes](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — installer le binaire](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — brancher votre agent](../../integrations/README.md)
- [Data residency & compliance — pour le RGPD et l'AI Act](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — la source de chaque chiffre](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ce qui est shipped/simulé/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Licence : [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
