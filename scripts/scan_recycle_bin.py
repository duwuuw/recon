"""Recover deleted files from $RECYCLE.BIN that originally lived under D:\\dl\\raicom."""
from __future__ import annotations

import glob
import os
import shutil

RAICOM = os.path.normcase(r"D:\dl\raicom")
RECYCLE_SID = r"D:\$RECYCLE.BIN\S-1-5-21-886192033-3535514284-1808128163-1001"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "recovered_from_recycle_bin")


def original_path(meta_path: str) -> str | None:
    b = open(meta_path, "rb").read()
    needle = "D:".encode("utf-16-le")
    idx = b.find(needle)
    if idx < 0:
        return None
    chunk = b[idx : idx + 800]
    return chunk.decode("utf-16-le", errors="ignore").split("\x00", 1)[0].strip() or None


def main() -> None:
    hits: list[tuple[str, str, str]] = []
    for meta in glob.glob(os.path.join(RECYCLE_SID, "$I*")):
        orig = original_path(meta)
        if not orig:
            continue
        if not os.path.normcase(orig).startswith(RAICOM):
            continue
        base = os.path.basename(meta)
        if not base.startswith("$I"):
            continue
        rid = "$R" + base[2:]
        data = os.path.join(RECYCLE_SID, rid)
        hits.append((meta, orig, data))

    print("Recycle bin entries originally under D:\\dl\\raicom:", len(hits))
    os.makedirs(OUT, exist_ok=True)
    for meta, orig, data in sorted(hits, key=lambda x: x[1]):
        exists = os.path.isfile(data) or os.path.isdir(data)
        print(("OK " if exists else "NO "), orig)
        if not exists:
            continue
        rel = os.path.relpath(orig, r"D:\dl\raicom")
        dest = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isfile(data):
            shutil.copy2(data, dest)
            print("     ->", dest)
        elif os.path.isdir(data):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(data, dest)
            print("     -> [dir]", dest)


if __name__ == "__main__":
    main()
