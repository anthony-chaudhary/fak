---
title: "fak — はじめての FAQ（日本語 / Japanese first-contact FAQ）"
description: "fak の日本語 FAQ 入口ページ：fak とは何か、モデルを書き換えずに使えるか、データはどこに残るのか（個人情報保護法 / APPI）、費用、速さ、対応モデル、prompt injection の防ぎ方、インストールまでを最短で。"
---

# fak — はじめての FAQ（日本語）

> これは**ローカライズされた入口ページ (entry point)** であり、ドキュメントの完全な翻訳では
> ありません。正式なドキュメントは英語です——このページは最初によくある質問に短く答えたうえで、
> [英語ドキュメント](https://github.com/anthony-chaudhary/fak/blob/main/README.md) へ引き継ぎます。
> **注記:** この翻訳は機械生成であり、ネイティブレビュー待ち。issue/PR による修正歓迎。
>
> **関連ページ:** [日本語イントロダクション](./README.md) · 全言語は [i18n ハブ](../README.md) から。

## Q1. fak とは何ですか？

あなたが**すでに動かしている AI エージェント**（Claude Code、Codex、Cursor、任意の
OpenAI/Anthropic/MCP クライアント）の前に置く、**1 個の静的な Go バイナリ**です。base URL を
1 つ向け直すだけで導入でき、エージェントの書き換えは不要です。長いセッションを安くし（古い
ターンを捨てつつ、provider の prompt-cache プレフィックスをバイト単位で同一に保つ）、tool call
ごとにルーティングし、ローカルの GGUF モデルをプロセス内で動かすこともでき、すべての呼び出しに
ついて監査可能な判定を記録します。

## Q2. モデルを乗り換えたり、エージェントを書き換えたりする必要はありますか？

いいえ。fak は**あなたが今使っているモデルを govern して cache** します。1 コマンドで包むか、
base URL を 1 つ `fak serve` に向け直すだけです。

```bash
fak manage claude
```

## Q3. データはどこへ行きますか——コンプライアンスは大丈夫ですか？

**self-host ファースト**です：ローカルモデルまたは国内 provider の前に立つ 1 個の静的バイナリで、
fail-closed のデータレジデンシー、default-deny の capability floor、そしてすべての tool call に
ついて改ざん検知可能な監査ログを備えます。**あなたのデータはマシンから出ません。** これは
国内でのデータ保護を定める**個人情報保護法（APPI）**のもとでデータレジデンシーを検討するチームに
とって扱いやすい構成です（本ページは技術的な性質の説明であって、法的助言ではありません）。

## Q4. 費用はいくらですか？ 本当に無料ですか？

**Apache-2.0**、無料、self-host です。クレジットカードも、越境請求書も、法人格も不要——円建ての
ベンダー契約も発生しません。`git clone` と `go install` が導入経路のすべてです。

## Q5. どれくらい安く、または速くなりますか？

トークン課金は円建てで効いてくるため、そのままマージンのレバーになります。計測済みの
**50 ターン × 5 エージェント**のセッションで、チューニング済みの warm-cache スタックより
**約 4.1× 少ない作業**です。約 60×（およそ 19 時間 → 約 19 分）という数字は、**naive な
「毎回すべて再送する」ループに対してのみ**成り立つもので、見出しの数字にはしません。この再利用の
効きは self-host のみ、かつ read 主体のフリート向けです。

## Q6. どのモデルが使えますか？

**Qwen2/Qwen3 と GLM-MoE** は in-kernel のリファレンスエンジンで bit-exact（逐ビット一致）が
実証済みです。それ以外（DeepSeek、Mistral、あらゆるオープンウェイトのモデル）は OpenAI 互換
ワイヤ経由で前段に置けます——Ollama / vLLM / SGLang / llama.cpp / LM Studio。

## Q7. prompt injection はどう止めますか？

分類器ではなく、**2 つの構造的なゲート**で止めます：default-deny の capability floor（危険な
ツールはそもそも allow-list に載らない）と、result quarantine（汚染されたツール結果はモデルの
コンテキストへ到達しない）。ライブテストでは、injection は無防備なベースラインに 5/5 到達し、
fak はそれを 5/5 で遮断しました。

## Q8. インストール方法は？

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

クローンから直接ビルドする場合は次のとおりです。社内ネットワークで `proxy.golang.org` に
到達できない場合は、`GOPROXY` に社内モジュールプロキシを設定してください。

```bash
go build -o fak ./cmd/fak
```

### 60 秒で確かめる（key もモデルも GPU も不要）

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q9. 次にどこへ行けばいい？

- [README（全体像）](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 分でローカルモデルを動かす](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — バイナリをインストール](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — エージェントを接続](../../integrations/README.md)
- [Data residency & compliance — データレジデンシー（個人情報保護法 / APPI）](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — すべての数字の出典](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — shipped/simulated/stub の区別](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

ライセンス：[Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)。
