---
title: "fak — インストールと実行（日本語版スタートガイド / Japanese install guide）"
description: "fak の日本語インストール入口ページ：クリーンな checkout から動くカーネルまで。Go 1.26+ だけで Tier 0 はオフライン・キー不要・GPU 不要。go build / go install のコマンドと、モデルを前段に置くまでの Tier マップ。self-host・Apache-2.0。"
---

# fak — インストールと実行（日本語版スタートガイド）

> これは**ローカライズされた入口ページ (entry point)** であり、ドキュメントの完全な翻訳では
> ありません。正式なドキュメントは英語です——このページはインストールと実行の入口だけを示し、
> 詳しい説明は[英語の README](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> へ引き継ぎます。
> **注記:** この翻訳は機械生成であり、ネイティブレビュー待ち。issue/PR による修正歓迎。
>
> **他の言語・全体像:** 日本語の入口は [./README.md](./README.md)、全言語は
> [i18n ハブ](../README.md) から。

これは **インストールと実行の入口ページ**です。fak の中身の詳しい説明（濃い pitch）は
[README](./README.md) にあります。このページはクリーンな checkout から、カーネルを動かし、
その後段にモデルを置くまでを、コピー&ペーストできるコマンドで最短でたどります。

fak は **1 個の Go バイナリ**で、AI エージェントとそのツール呼び出しの間に入り、すべての
tool call を*実行する前に*審査し、長いセッションでは繰り返しの共有作業を再利用します。
エージェントを書き換える必要はありません——base URL を `fak serve` に向けるか、既存の
エージェントを 1 コマンドで包むだけです。

```bash
fak manage claude      # 既存のエージェントを 1 コマンドで包む
```

> **日本市場向けメモ。** fak は self-host ファーストの Apache-2.0 バイナリで、データは自社
> インフラに残せます（個人情報保護法 / APPI のデータレジデンシー要件に取り組む際の技術的な
> 構成要素であり、法的助言ではありません）。ベンダー契約もクレジットカードも越境インボイスも
> 不要なので、**トークン単価（円）** がそのままマージンのレバーになります。長いセッションでの
> 共有作業の再利用は **self-host 時のみ**・read-heavy なフリートに効き、チューニング済みの
> warm-cache スタックに対して**約 4.1× 少ない作業**です（naive な再送ループに対しては約 60×
> ですが、誠実な headline は 4.1× です）。数字の出典は
> [BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)。

---

## 0. 前提条件

- **Go 1.26+。** `fak/go.mod` は `go 1.26` を宣言します。Go 既定の `GOTOOLCHAIN=auto` なら、
  古い `go` でも初回ビルド時に正しいツールチェインを自動ダウンロードします（一度だけネットワーク
  が必要）。それ以外は <https://go.dev/dl/> から Go 1.26 を入れてください。`go version` で確認。
- **Tier 0 はこれだけ**：GPU 不要・API キー不要・ネットワーク不要。
- **Tier 1** は追加で任意の OpenAI 互換モデルサーバー（例：Ollama）が必要です。

---

## Tier マップ（セットアップコストの小さい順）

同じ 1 個のバイナリでできることが 4 段階あり、**段の間で新しいものは何もインストールされません**。

| Tier | 何が得られるか | セットアップ |
|---|---|---|
| **0 — カーネルを試す** | 審査境界（adjudication boundary）をオフラインで動かして測る | `go build` |
| **1 — 実モデルを前段に置く** | 別で serve するモデル（Ollama / vLLM / llama.cpp / クラウド）の前にカーネルを立てる | + 稼働中の OpenAI 互換サーバー |
| **1b — ローカルモデルを 1 コマンドで** | 既存エージェントとローカル GGUF モデルを in-kernel で実行——キー・ネットワーク・第 2 端末なし | `fak manage --gguf qwen2.5:7b -- claude` |
| **2 — 融合された in-kernel モデル** | カーネルが所有する pure-Go の forward pass | + （実重み用の）Python export |

モデルを前段に置いて**実用的に serve したいだけ**なら **Tier 1** です。Tier 2 の in-kernel
モデルは HuggingFace に対して逐ビット一致（bit-exact）が実証されたリファレンス forward pass
であり、チャット品質の serving エンジンではありません。

fak はモデルを置き換えません——govern して cache します。**Qwen2/Qwen3 と GLM-MoE** は
in-kernel リファレンスエンジンで bit-exact が実証済み。それ以外（DeepSeek・Mistral・あらゆる
オープンウェイトのモデル）は OpenAI 互換ワイヤ経由（Ollama / vLLM / SGLang / llama.cpp /
LM Studio または任意の OpenAI 互換 API）で前段に置けます。

---

## 1. バイナリを入手する

**Go でインストール**（モジュールパスがリポジトリのルートなので直接入ります）：

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # -> $(go env GOBIN) / $GOPATH/bin
```

**clone からビルド**（コントリビューター）：

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # -> ./fak
./fak help
```

> **Windows メモ。** ビルドは `go build -o fak.exe ./cmd/fak` としてください。`-o fak`
> （拡張子なし）だと cmd.exe / PowerShell が名前で起動できないファイルが残ります。以降この
> ガイドで `./fak` と書かれた箇所は `.\fak.exe` に読み替えてください。

---

## 2. 動作確認（60 秒・キー不要・モデル不要・GPU 不要）

Tier 0 はすべてオフラインで決定的です。`fak/` の中から実行してください。

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

`preflight` は **default-deny の capability floor** を示します——許可リストにない操作は
fail-closed で拒否され、モデルが何を説得されても実行できません。`agent --offline` は
prompt injection の A/B です：疑わしいツールの*結果*は quarantine（隔離）に入り、そもそも
モデルのコンテキストへ到達しません——分類器ではなく、構造によって。

---

## 次に読むもの

- [README（全体像・濃い pitch）](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 分でローカルモデルを動かす](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — バイナリのインストールと Tier マップ（英語の元ページ）](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — エージェントを接続](../../integrations/README.md)
- [Data residency & compliance — APPI のデータレジデンシー](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — すべての数字の出典](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — shipped/simulated/stub の区別](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

ライセンス：[Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)。
