# LangChain Advanced RAG Learning Project

このプロジェクトは、LangChain Expression Language (LCEL) を活用し、検索精度を大幅に向上させた高度な RAG (Retrieval-Augmented Generation) パイプラインを実装した学習用リポジトリです。

## プロジェクト構成

```text
/langchain_learning
├── src/
│   ├── main.py          # アプリケーションのエントリーポイント
│   ├── chains.py        # RAG チェーンの定義（Multi-Query, RRF, Rerank を含む）
│   ├── vectorstore.py   # VectorStore (Chroma) および Retriever の設定
│   ├── utils.py         # Reciprocal Rank Fusion (RRF) などのユーティリティ関数
│   ├── prompts.py       # 各ステップで使用するプロンプトテンプレート
│   └── schema.py        # Pydantic モデルによる構造化データの定義
└── .env                 # 環境設定ファイル
```

### 1. main.py
アプリケーションのエントリーポイントです。環境変数の読み込み、Retrieverの初期化、およびRAGチェーンの呼び出しを行います。

### 2. chains.py
LCELを用いた高度なRAGパイプラインを定義しています。
- **Multi-Query**: 1つの質問から3〜5つの異なる検索クエリを生成し、検索の網羅性を高めます。
- **Reciprocal Rank Fusion (RRF)**: 複数クエリの検索結果を統合し、ランキングを再計算します。
- **Cohere Rerank**: 統合されたドキュメントを再ランク付けし、最も関連性の高い上位4件に絞り込みます。

### 3. vectorstore.py
ベクトルデータベース（Chroma）の操作を担当します。
- `get_retriever`: 保存済みのベクトルストアから検索用インスタンスを取得します。
- `add_documents_to_db`: テキストデータをベクトル化してDBに永続化します。

### 4. utils.py
Reciprocal Rank Fusion (RRF) アルゴリズムを実装しています。複数の検索結果リストを、重み付けを用いて1つのリストに統合します。

### 5. prompts.py
LLMに与えるプロンプトテンプレートを管理しています。
- クエリ生成用（`query_gen_prompt`）
- 最終回答用（`answer_prompt`）

### 6. schema.py
Pydanticを用いたデータ構造の定義です。LLMから構造化されたクエリリスト（`MultiQuery`）を取得するために使用します。

## セットアップ

### 1. 依存関係のインストール

このプロジェクトは Poetry を使用してパッケージ管理を行っています。

```bash
poetry install
```

### 2. 環境変数の設定
ルートディレクトリに `.env` ファイルを作成し、必要な設定を記述してください。

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_NAME=gpt-5-nano
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-small
COHERE_API_KEY=your_cohere_api_key_here
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## 使い方

メインアプリケーションを実行して、RAGによる質問回答を開始します。

```bash
# RAGアプリケーションの実行
poetry run python src/main.py
```

### ドキュメントの追加
新しいテキストデータをベクトルデータベースに登録する場合は、vectorstore.py の add_documents_to_db 関数を利用したスクリプトを作成・実行してください。
