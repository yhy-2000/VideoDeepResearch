
import sys
from pathlib import Path


KEYWORDS = ("Copyright", "2025", "LlamaFactory")


def main():
    path_list: list[Path] = []
    for check_dir in sys.argv[1:]:
        path_list.extend(Path(check_dir).glob("**/*.py"))

    for path in path_list:
        with open(path.absolute(), encoding="utf-8") as f:
            file_content = f.read().strip().split("\n")
            if not file_content[0]:
                continue

            print(f"Check license: {path}")
            assert all(keyword in file_content[0] for keyword in KEYWORDS), f"File {path} does not contain license."


if __name__ == "__main__":
    main()
