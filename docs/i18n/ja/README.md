---
title: "fak — Fused Agent Kernel（日本語版イントロダクション / Japanese introduction）"
description: "fak の日本語入口ページ：AI エージェントとツール呼び出しの間に入る Go バイナリ——すべての tool call を実行前に審査し、長いセッションでは安定したプレフィックスを再利用。self-host のデータレジデンシー、改ざん検知可能な監査ログ、Apache-2.0 の静的バイナリ。"
---

# fak — Fused Agent Kernel（日本語版イントロダクション）

> これは**ローカライズされた入口ページ (entry point)** であり、ドキュメントの完全な翻訳では
> ありません。正式なドキュメントは英語です——このページは fak の核心、60 秒での実証、
> インストール手順を示したうえで、[英語ドキュメント](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> へ引き継ぎます。
> **注記:** この翻訳は機械生成であり、ネイティブレビュー待ち。issue/PR による修正歓迎。
>
> **他の言語:** [简体中文](../zh/README.md) ほか、全言語は [i18n ハブ](../README.md) から。

## fak を一行で

**fak は Go バイナリ**で、あなたの AI エージェントとそのツール呼び出しの間に入ります——
すべての tool call を*実行する前に*審査し、長いセッションでは繰り返しの作業を再利用します。
結果として、同じエージェントループが、他に何も変えることなく**より安全に、より安く、
より速く**なります。

エージェントを書き換える必要はありません——base URL を `fak` に向けるだけで、すべての
tool call はまず capability floor（既定拒否の権限フロア）を通過します。

```bash
fak manage claude      # 既存のエージェントを 1 コマンドで包む
```

## 日本のチームにとってなぜ効くのか

- **モデルは自由——使っている国産・オープンウェイトのモデルを govern して cache。** fak は
  モデルの置き換えを求めません——モデルを包みます。**Qwen2/Qwen3 と GLM-MoE** は in-kernel の
  リファレンスエンジンで bit-exact（逐ビット一致）が実証済み。**DeepSeek・Mistral をはじめ、
  あらゆるオープンウェイトのモデル**は OpenAI 互換ワイヤ経由（Ollama / vLLM / SGLang /
  llama.cpp / LM Studio または任意の OpenAI 互換 API）で前段に置けます。
- **データは自社インフラに残る（個人情報保護法 / APPI）。** fak は self-host ファースト：
  **ローカルモデル**または任意の provider の前段に立つ 1 個の静的バイナリで、どのバックエンドでも
  fail-closed、default-deny の capability floor、そしてすべての tool call を記録する改ざん検知
  可能な監査ログを備えます。「既定で第三者に転送される」経路を気にする必要がありません。
  これは一般的な「データが自社インフラから出ない」という性質の説明であって、法的助言では
  ありません。詳細：[Data residency & compliance](../../explainers/data-residency-and-compliance.md)。
- **改ざん検知可能な監査ログが最初から。** fak はすべての判断を append-only・SHA-256 で
  ハッシュ連鎖された決定ジャーナルに書き出し、オフラインで検証できます——監査が求める技術的な
  構成要素であり、特定の規格や法令への適合を主張するものではありません。
- **トークン単価はマージンのレバー。** fak は長いセッションで共有された作業（system prompt と
  ツールリスト——これまでの作業の KV cache）を再利用します：50 ターン × 5 エージェントの実行で、
  チューニング済みの warm-cache スタックより**約 4.1× 少ない作業**（naive な再送ループに対しては
  約 60× ですが、誠実な数字は 4.1× です）。さらに per-aspect ルーティングで安く済む部分をより
  安いモデルへ回します。すべての数字は
  [BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
  に根拠があります。
- **Apache-2.0 の 1 個の静的バイナリ——調達のハードルなし。** fak は無料・オープンソース・
  self-host：ベンダー契約もクレジットカードもアカウントも不要。default-deny の capability floor と、
  prompt injection による汚染されたツール結果の quarantine（隔離）を、ラップトップからフリートまで
  同じ 1 つの成果物で提供します。追加するのは flag であって、コンポーネントではありません。

全言語の入口は [i18n ハブ](../README.md) から辿れます。

## fak が解決する問題

- **長いセッションが高コストでなくなる。** provider の prompt-cache 割引は、cached prefix が
  バイト単位で同一である間だけ保たれます；fak は途中の古いターンを捨てつつ prefix をバイト単位で
  同一に保つため、割引が途切れません。
- **default-deny のセキュリティ。** permission ポリシーはカーネル*内*の同じ call path 上で
  動きます。取り返しのつかない操作を防ぐことは、攻撃を「検知」できるかどうかに依存しません——
  そのレバーはこれまで配線されていませんでした。これは fail-open ではなく **fail-closed** です。
- **prompt injection / 汚染されたツール結果。** 疑わしいツールの*結果*は quarantine に
  入れられ、そもそもモデルのコンテキストへ到達しません——分類器（classifier）ではなく、
  構造によって。

## 60 秒で確かめる（キー不要・モデル不要・GPU 不要）

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # インジェクションを阻止、タスクは完了
```

## あなたのモデルと組み合わせる

fak はモデルを置き換えません——モデルを govern して cache します。**Qwen2/Qwen3 と GLM-MoE**
は in-kernel のリファレンスエンジンで bit-exact が実証済みです；それ以外（Mistral・DeepSeek・
あらゆるオープンウェイトのモデル）は OpenAI 互換ワイヤ経由で前段に置きます——Ollama / vLLM /
SGLang / llama.cpp / LM Studio または任意の OpenAI 互換 API。

## 次に読むもの

- [クイックスタート — 約 10 分でローカルモデル](./quickstart.md)
- [インストール — バイナリと tier マップ](./install.md)
- [FAQ — はじめての質問](./faq.md)
- [README（全体像）](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 分でローカルモデルを動かす](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — バイナリをインストール](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — エージェントを接続](../../integrations/README.md)
- [Data residency & compliance — データレジデンシー](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — すべての数字の出典](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — shipped/simulated/stub の区別](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

ライセンス：[Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)。
