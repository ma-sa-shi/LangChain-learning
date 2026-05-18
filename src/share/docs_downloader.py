import requests
import argparse
from pathlib import Path


def download_raw_rst(library_name, output_dir):
    """指定されたライブラリのRSTファイルをGitHubからダウンロードして保存する関数
    Args:
        library_name (str): ダウンロードするライブラリの名前
        output_dir (str): RSTファイルを保存するディレクトリ
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / f"{library_name}.rst"

    url = f"https://raw.githubusercontent.com/python/cpython/main/Doc/library/{library_name}.rst"

    try:
        response = requests.get(url)
        # 4XXや5XXのステータスコードが返された場合に例外を発生させる
        response.raise_for_status()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"{library_name}.rstを保存しました")

    except requests.exceptions.RequestException as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("libraries", nargs="+", help="ライブラリ名")
    parser.add_argument("-o", "--output", default="docs", help="保存先ディレクトリ")
    args = parser.parse_args()
    for lib in args.libraries:
        download_raw_rst(lib, args.output)
