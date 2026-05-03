# LangChain Learning Project

LangChain を活用した LLM アプリケーション開発の学習用リポジトリです。
LCEL (LangChain Expression Language) の基本から、並列処理、RAG (Retrieval-Augmented Generation) のデータインジェクションの実装例を含んでいます。

## プロジェクト構成

```text
/langchain_learning
├── src/
│   ├── lced.py           # LCEL の基本実装（Prompt | Model | Parser）
│   ├── lced_parallel.py  # RunnableParallel を用いた並列処理と構造化出力
│   └── rag_flow.py       # ドキュメントのベクトル化と ChromaDB への保存 (RAG インジェクション)
└── .env                  # 環境設定ファイル（要作成）
```

## 各ファイルの詳細

### 1. [lced.py](src/lced.py) - LCEL の基本
LCEL の最も標準的なパイプライン構成を学びます。
- **概要**: ユーザーが入力したトピックについて、LLM が簡潔に 1 行で説明します。
- **構成**: `PromptTemplate` | `ChatOpenAI` | `StrOutputParser` をパイプ記法で連結しています。

### 2. lced_parallel.py - 並列実行と構造化出力
`RunnableParallel` を活用した、より高度なチェーン構成を学びます。
- **概要**: 1 つの入力文に対して、「感情分析」と「要約」を同時に実行します。
- **特徴**: `Pydantic` モデルを用いた `with_structured_output` を採用しており、LLM からのレスポンスを型定義されたオブジェクトとして取得します。

### 3. rag_flow.py - RAG データインジェクション
RAG システムの基盤となる、ドキュメントのベクトル化と永続化のフローです。
- **概要**: テキストファイルを読み込み、チャンク分割してベクトルデータベース（ChromaDB）に保存します。
- **技術スタック**: `TextLoader`, `RecursiveCharacterTextSplitter`, `OpenAIEmbeddings`, `Chroma`
- **コマンド引数**: `python src/rag_flow.py <ファイルパス> --chunk_size 500` のように実行可能です。

## セットアップ

### 1. 依存関係のインストール
このプロジェクトは [Poetry](https://python-poetry.org/) を使用してパッケージ管理を行っています。

```bash
poetry install
```

### 2. 環境変数の設定
ルートディレクトリに `.env` ファイルを作成し、必要な設定を記述してください。

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_NAME=gpt-5-nano
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-small
```

## 使い方

各スクリプトは個別に実行して動作を確認できます。

```bash
# 基本的なチェーンの実行
poetry run python src/lced.py

# 並列処理と構造化出力の実行
poetry run python src/lced_parallel.py

# ドキュメントをベクトル化して ChromaDB に保存
poetry run python src/rag_flow.py data/sample.txt
```