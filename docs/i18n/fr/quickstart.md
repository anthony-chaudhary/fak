---
title: "fak — démarrage rapide (un modèle local en 10 minutes)"
description: "Guide de démarrage rapide en français pour fak : de zéro à une IA locale gouvernée en ~10 minutes — hors ligne, sans clé, sans facture cloud, les données restent sur votre machine. Auto-hébergement compatible RGPD et AI Act européen ; le CPU suffit pour les petits modèles."
---

# fak — démarrage rapide (un modèle local en 10 minutes)

> Ceci est une **page d'entrée localisée (entry point)**, pas une traduction complète de
> la documentation. La documentation canonique est en anglais — cette page vous met en
> route en ~10 minutes, puis vous renvoie vers la
> [documentation anglaise](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Avertissement :** cette page est générée automatiquement et attend une relecture
> native — les corrections par issue/PR sont bienvenues.
>
> **Voir aussi :** [introduction en français](./README.md) · toutes les langues sur le
> [hub i18n](../README.md).

## La promesse

En une dizaine de minutes, vous passez de zéro à une **IA locale gouvernée** : elle
tourne **hors ligne**, sans clé d'API, sans facture cloud, et **vos données ne quittent
pas votre machine** — un point direct pour le RGPD et le journal d'audit attendu par
l'AI Act européen. Un **CPU suffit** pour les petits modèles ; aucun GPU requis.

## Le chemin le plus rapide

Enveloppez votre agent existant avec un modèle local, en une seule commande :

```bash
fak manage --gguf qwen2.5:7b -- claude
```

`fak` se place entre l'agent et ses tool calls : il vérifie chaque tool call *avant* son
exécution (capability floor en default-deny) et sert le modèle local en local. Vous ne
réécrivez pas votre agent — vous l'enveloppez.

## La preuve en 60 secondes (pas de clé, pas de modèle, pas de GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

Ce que cela montre : une action hors de la liste d'autorisation est refusée **par
structure** — un `DENY` fail-closed vérifié *dans* le kernel, sur le même call path — et
la boucle d'agent termine sa tâche même lorsqu'une prompt injection est présente. Sur un
banc d'essai en direct, l'injection a atteint la baseline non protégée **5/5** ; fak l'a
murée **5/5**. Les *résultats* d'outils suspects sont mis en quarantaine hors du contexte
du modèle — le détecteur qui les signale est traité comme ~100 % contournable par
conception : un bonus, jamais le plancher.

## fak en bref

**fak est un seul binaire Go statique** qui se place entre votre agent IA et ses tool
calls. Il vérifie chaque appel *avant* son exécution, et réutilise le travail partagé et
stable au fil d'une session longue. Résultat : la **même** boucle d'agent devient plus
sûre, moins chère et plus rapide, sans réécriture. Vous ne remplacez pas votre modèle —
fak le gouverne et le met en cache. **Qwen2/Qwen3 et GLM-MoE** sont prouvés bit-exact
dans le moteur de référence in-kernel ; tout le reste (DeepSeek, Mistral, n'importe quel
modèle open-weights) est fronté via l'interface compatible OpenAI — Ollama / vLLM /
SGLang / llama.cpp / LM Studio ou toute API compatible OpenAI.

## À quelle vitesse ?

Le chiffre honnête : sur une session mesurée de 50 tours × 5 agents, fak fait **~4,1×
moins de travail** qu'une stack warm-cache **déjà optimisée**. C'est un levier de marge
direct — le coût des tokens se compte en euros, et la réutilisation du travail partagé le
réduit d'autant sur les flottes à lecture intensive.

Le chiffre spectaculaire de **~60×** (environ 19 heures ramenées à ~19 minutes) n'est
vrai **que face à une boucle naïve** qui re-envoie tout à chaque tour ; ce n'est **pas**
le chiffre de référence. Le gain de réutilisation est **self-host uniquement**. fak
garde le préfixe caché **identique octet par octet** en évacuant les vieux tours du
milieu, si bien que la remise prompt-cache du provider survit — fak **garantit**
l'identité octet-à-octet du préfixe ; que le provider réutilise réellement le cache est
sa décision, que fak relaie plutôt qu'il ne la revendique.

## Licence et coût

fak est **Apache-2.0**, libre, auto-hébergé — pas de carte bancaire, pas de facture
transfrontalière, pas d'entité juridique à créer. `git clone` puis
`go install github.com/anthony-chaudhary/fak/cmd/fak@latest`, c'est tout le chemin.

## Où aller ensuite

- [README (la vue d'ensemble complète)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — un modèle local en 10 minutes](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — installer le binaire](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Data residency & compliance — pour le RGPD et l'AI Act](../../explainers/data-residency-and-compliance.md)
- [Integrations — brancher votre agent](../../integrations/README.md)
- [BENCHMARK-AUTHORITY — la source de chaque chiffre](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ce qui est shipped/simulé/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

Licence : [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
