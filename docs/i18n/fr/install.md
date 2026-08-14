---
title: "fak — installation et démarrage (démarrage rapide en français / French getting started)"
description: "Page d'installation et de démarrage de fak en français : de la copie propre au kernel qui tourne, puis à un modèle servi derrière lui. Le pitch complet est dans le README ; ici, seulement la carte des tiers et les commandes d'installation."
---

# fak — installation et démarrage (français)

> Ceci est une **page d'entrée localisée (entry point)**, pas une traduction complète de
> la documentation. La documentation canonique est en anglais — cette page vous amène
> d'une copie propre jusqu'au kernel qui tourne, puis vous renvoie vers la
> [documentation anglaise](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Avertissement :** cette page est générée automatiquement et attend une relecture
> native — les corrections par issue/PR sont bienvenues.
>
> Toutes les langues sur le [hub i18n](../README.md). Pour le pitch dense en français,
> voir la [page d'introduction](./README.md).

C'est la **porte d'entrée « installer et lancer »**. Le pitch dense — ce qu'est fak, pourquoi
c'est pertinent en Europe (RGPD, journal d'audit aligné sur l'EU AI Act), les chiffres de
coût en euros — est dans l'[introduction en français](./README.md) et dans le
[README anglais](https://github.com/anthony-chaudhary/fak/blob/main/README.md). Cette page
se limite à la carte des tiers et aux commandes.

`fak` est **un seul binaire Go** : un artefact statique unique, sans dépendance externe
(pas de Python, pas de toolchain CUDA). Ce binaire se place entre votre agent IA et ses
tool calls : il vérifie chaque appel *avant* son exécution et réutilise le travail partagé
des sessions longues. Vous ne réécrivez pas votre agent — vous pointez une base URL vers
`fak serve`, ou vous enveloppez un agent existant en une commande : `fak manage -- claude`.

## Prérequis

- **Go 1.26+.** `fak/go.mod` déclare `go 1.26`. Avec le `GOTOOLCHAIN=auto` par défaut, un
  `go` plus ancien télécharge automatiquement la bonne toolchain au premier build (réseau
  requis une seule fois) ; sinon, installez Go 1.26 depuis <https://go.dev/dl/>. Vérifiez
  avec `go version`.
- **C'est tout pour le Tier 0** : pas de GPU, pas de clé d'API, pas de réseau.

## Les tiers, en un coup d'œil

Quatre choses à faire avec le binaire, par coût de mise en place croissant — **rien de
nouveau ne s'installe entre elles** :

| Tier | Ce que vous obtenez | Mise en place |
|---|---|---|
| **0 — Essayer le kernel** | Exécuter/mesurer la frontière d'adjudication hors ligne | `go build` (aucun téléchargement) |
| **1 — Placer un vrai modèle derrière** | Mettre le kernel devant un modèle servi ailleurs (Ollama / vLLM / llama.cpp / un provider cloud) | + un serveur compatible OpenAI |
| **1b — Modèle local en une commande** | Un modèle GGUF local, in-kernel, avec votre agent existant — sans clé, sans réseau | `fak manage --gguf qwen2.5:7b -- claude` (~5 Go GGUF, mis en cache) |
| **2 — Le modèle fusionné in-kernel** | La passe forward SmolLM2 pure-Go que le kernel possède | + export Python (poids réels) |

Pour simplement **servir un modèle utile avec fak devant**, visez le **Tier 1**. Le modèle
in-kernel du Tier 2 est une *passe forward de référence* prouvée bit-exact face à
HuggingFace, pas un moteur de service de qualité chat.

## Installer le binaire

**Depuis la copie (contributeur) :**

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak
./fak help
```

**Ou directement avec Go** — le chemin de module `github.com/anthony-chaudhary/fak` est la
racine du dépôt, donc il s'installe directement :

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Note Windows.** Compilez avec `go build -o fak.exe ./cmd/fak` : un `-o fak` sans
> extension laisse un fichier que cmd.exe / PowerShell ne peuvent pas lancer par son nom.

## La preuve en 60 secondes (pas de clé, pas de modèle, pas de GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

Le `capability floor` en **default-deny** s'exécute *dans* le kernel, sur le même call path :
une action absente de l'allow-list ne peut pas s'exécuter, quoi que l'on ait fait croire au
modèle. En test live, une prompt injection a atteint la baseline non protégée 5/5 ; fak l'a
murée 5/5.

## Licence et coût

**Apache-2.0**, libre, auto-hébergé — pas de carte bancaire, pas de facture transfrontalière,
pas d'entité juridique à créer. `git clone` puis
`go install github.com/anthony-chaudhary/fak/cmd/fak@latest`, c'est tout le chemin.

## Où aller ensuite

- [README anglais — la vue d'ensemble complète](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — un modèle local en 10 minutes](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — la référence d'installation complète et la carte des tiers](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — la source de chaque chiffre](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ce qui est shipped/simulé/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Data residency & compliance — pour le RGPD et l'EU AI Act](../../explainers/data-residency-and-compliance.md)
- [Integrations — brancher votre agent](../../integrations/README.md)
- [Introduction en français (le pitch dense)](./README.md)

Licence : [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
</content>
</invoke>
