---
title: "fak — le Fused Agent Kernel (introduction en français / French introduction)"
description: "Page d'entrée en français pour fak : un binaire Go qui vérifie chaque tool call avant exécution — la même boucle d'agent devient plus sûre, moins chère, plus rapide ; auto-hébergement compatible RGPD et journal d'audit aligné sur l'AI Act européen."
---

# fak — le Fused Agent Kernel (introduction en français)

> Ceci est une **page d'entrée localisée (entry point)**, pas une traduction complète de
> la documentation. La documentation canonique est en anglais — cette page vous donne
> l'essentiel, la preuve en 60 secondes et le chemin d'installation, puis vous renvoie
> vers la [documentation anglaise](../../../README.md).
> **Avertissement :** cette traduction est générée automatiquement et attend une relecture
> native — les corrections par issue/PR sont bienvenues.
>
> **Auf Deutsch:** [Deutsch](../de/README.md) — toutes les langues sur le
> [hub i18n](../README.md).

## fak en une ligne

**fak est un binaire Go** qui se place entre votre agent IA et ses tool calls — il vérifie
chaque tool call *avant* son exécution, et réutilise le travail stable dans les sessions
longues. Résultat : la même boucle d'agent devient **plus sûre, moins chère et plus
rapide**, sans rien changer d'autre.

Vous ne réécrivez pas votre agent — vous pointez une base URL vers `fak`, et chaque tool
call passe d'abord par le capability floor.

```bash
fak guard -- claude    # enveloppe votre agent existant en une seule commande
```

## Pourquoi c'est pertinent pour les startups européennes

- **Les données restent sur votre infrastructure (RGPD/GDPR).** fak est self-host-first :
  un binaire statique placé devant un **modèle local** (`fak guard --gguf …`) ou le
  provider de votre choix — fail-closed sur chaque backend, capability floor en
  default-deny, et un journal d'audit inviolable pour chaque tool call. Il n'existe aucun
  chemin « transféré par défaut vers un pays tiers » à analyser. Détails :
  [Data residency & compliance](../../explainers/data-residency-and-compliance.md).
- **Le journal d'audit de l'AI Act européen est déjà livré (article 12, applicable le
  2 août 2026).** fak écrit un journal de décisions append-only, chaîné par hachage
  SHA-256, vérifiable hors ligne avec `fak audit verify` — la correspondance entre les
  obligations de l'article 12 et le mécanisme livré est documentée dans
  [EU AI Act Article 12 conformance](../../standards/eu-ai-act-article-12-conformance.md).
  Ce n'est pas un conseil juridique — mais c'est la brique technique qu'un audit demande.
- **Le prix des tokens est un levier de marge.** fak réutilise le travail partagé des
  sessions longues (le system prompt + la liste d'outils — le KV cache du travail déjà
  fait) : sur un run de 50 tours × 5 agents, **~4,1× moins de travail** qu'une stack
  warm-cache optimisée (~60× par rapport à une boucle naïve de re-envoi ; le chiffre
  honnête est 4,1×). Le routage per-aspect envoie en plus les parties bon marché vers des
  modèles moins chers. Chaque chiffre est tracé dans
  [BENCHMARK-AUTHORITY](../../../BENCHMARK-AUTHORITY.md).
- **Apache-2.0, zéro friction d'achat.** fak est libre, open source et auto-hébergé — pas
  de contrat fournisseur, pas de carte bancaire, pas de compte. `git clone` et
  `go install`, c'est tout le chemin.
- **Un binaire statique, zéro dépendance externe.** Des ops simples pour une petite
  équipe — pas de sidecar, pas d'authorizer séparé. Du laptop à la flotte, le même
  artefact ; vous ajoutez des flags, pas des composants.

## Quels problèmes fak résout

- **Les sessions longues cessent d'être chères.** La remise prompt-cache du provider ne
  tient que si le préfixe caché reste identique octet par octet ; fak évacue les vieux
  tours tout en gardant le préfixe byte-identique — la remise ne casse pas.
- **Sécurité default-deny.** La politique de permissions s'exécute *dans* le kernel, sur
  le même call path. Empêcher une action irréversible ne dépend pas de la « détection »
  d'une attaque — le levier n'a jamais été câblé. C'est **fail-closed**, pas fail-open.
- **Prompt injection / résultats d'outils empoisonnés.** Les *résultats* d'outils suspects
  sont placés en quarantaine pour qu'ils n'entrent jamais dans le contexte du modèle — par
  structure, pas par un classifieur.

## La preuve en 60 secondes (pas de clé, pas de modèle, pas de GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection bloquée, tâche quand même accomplie
```

## Avec votre modèle

fak ne remplace pas votre modèle — il le gouverne et le met en cache. **Qwen2/Qwen3 et
GLM-MoE** sont prouvés bit-exact dans le moteur de référence in-kernel ; tout le reste
(Mistral, DeepSeek, n'importe quel modèle open-weights) est fronté via l'interface
compatible OpenAI — par Ollama / vLLM / SGLang / llama.cpp / LM Studio ou toute API
compatible OpenAI.

## Où aller ensuite

- [README (la vue d'ensemble complète)](../../../README.md)
- [START-HERE — un modèle local en 10 minutes](../../../START-HERE.md)
- [Getting Started — installer le binaire](../../../GETTING-STARTED.md)
- [Integrations — brancher votre agent](../../integrations/README.md)
- [Data residency & compliance — pour le RGPD](../../explainers/data-residency-and-compliance.md)
- [EU AI Act article 12 — conformité du journal d'audit](../../standards/eu-ai-act-article-12-conformance.md)
- [BENCHMARK-AUTHORITY — la source de chaque chiffre](../../../BENCHMARK-AUTHORITY.md)
- [CLAIMS — ce qui est shipped/simulé/stub](../../../CLAIMS.md)

Licence : [Apache-2.0](../../../LICENSE).
