import requests
import argparse
from pathlib import Path

def download_raw_rst(library_name, output_dir):
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / f"{library_name}.rst"

    url = f"https://raw.githubusercontent.com/python/cpython/main/Doc/library/{library_name}.rst"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded {library_name}.rst")
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download {library_name}.rst: {e}")
    except Exception as e:
        print(f"An error occurred while downloading {library_name}.rst: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("libraries", nargs="+", help="ライブラリ名")
    parser.add_argument("-o", "--output", default="docs", help="保存先ディレクトリ")
    args = parser.parse_args()
    for lib in args.libraries:
        download_raw_rst(lib, args.output)