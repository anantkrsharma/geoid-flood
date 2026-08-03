#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import get_session

REPO = "links-ads/geoid-flood"
TREES = ("geoid-flood", "geoid-flood-heldout")
LAYERS = ("s1grd", "s1rtc", "s2l2a", "dem", "label", "cloudmask", "floodmask", "permwater", "validity")
SPLITS = ("train", "val", "test")
METADATA = ("data_tiles_s256_st128.csv", "tile_catalog.parquet")

SHARD_RE = re.compile(r"^(?P<tree>[^/]+)/shards/(?P<split>[^/]+)/(?P<layer>[^/]+)/[^/]+\.tar$")


class _Counting:
    """Wraps a stream so we can sha256 and count bytes as tarfile pulls from it."""

    def __init__(self, raw):
        self.raw = raw
        self.h = hashlib.sha256()
        self.n = 0

    def read(self, size=-1):
        chunk = self.raw.read(size)
        self.h.update(chunk)
        self.n += len(chunk)
        return chunk


def _safe_members(tar, root: Path):
    root = root.resolve()
    for info in tar:
        if not info.isreg():
            raise ValueError(f"unexpected member type in shard: {info.name}")
        dest = (root / info.name).resolve()
        if not str(dest).startswith(str(root) + "/"):
            raise ValueError(f"member escapes destination: {info.name}")
        yield info, dest


def stream_shard(repo: str, path: str, dest_root: Path, expect_sha: str | None, token: str | None) -> int:
    """Stream one shard from the Hub straight into dest_root. Returns bytes transferred."""
    url = hf_hub_url(repo_id=repo, filename=path, repo_type="dataset")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = get_session().get(url, stream=True, headers=headers, timeout=60)
    resp.raise_for_status()
    counter = _Counting(resp.raw)
    tree = path.split("/", 1)[0]
    out_root = dest_root / tree
    with tarfile.open(fileobj=counter, mode="r|") as tar:
        for info, dest in _safe_members(tar, out_root):
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            src = tar.extractfile(info)
            with open(tmp, "wb") as fh:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
    # Drain any trailing padding so the hash covers the whole object.
    while True:
        if not counter.read(1 << 20):
            break
    if expect_sha and counter.h.hexdigest() != expect_sha:
        raise ValueError(f"sha256 mismatch for {path}")
    return counter.n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download GEOID-Flood from the Hugging Face Hub and unpack it to the "
                    "canonical layout. Shards are streamed straight into the output tree, so "
                    "peak disk usage equals the final dataset size. Re-running resumes. "
                    "--sample is the exception: it downloads via the Hub cache and then copies "
                    "into --dest, so it needs roughly twice the sample size while it runs."
    )
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dest", default="data", type=Path, help="parent dir; trees are created under it")
    ap.add_argument("--tree", nargs="+", choices=TREES, default=list(TREES))
    ap.add_argument("--split", nargs="+", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--layer", nargs="+", choices=LAYERS, default=list(LAYERS))
    ap.add_argument("--list", action="store_true", help="print the selection and exit")
    ap.add_argument("--sample", action="store_true",
                    help="fetch only the two-AoI EMSR712 sample: 47 tiles, all nine layers, "
                         "~2.9 GB (needs ~5.8 GB free, see below) and exit")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    if args.sample:
        from huggingface_hub import snapshot_download

        tmp = snapshot_download(
            repo_id=args.repo, repo_type="dataset", allow_patterns=["sample/*"], token=args.token
        )
        for src in (Path(tmp) / "sample").rglob("*"):
            if src.is_file():
                dst = args.dest / src.relative_to(Path(tmp) / "sample")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
        print(f"sample -> {args.dest}/geoid-flood/")
        print("run a config against it with:")
        print("  --data.init_args.metadata_filename data_tiles_s256_st128_sample.csv")
        return

    index_path = hf_hub_download(
        repo_id=args.repo, filename="shard_index.json.gz", repo_type="dataset", token=args.token
    )
    with gzip.open(index_path, "rt") as fh:
        index = json.load(fh)

    selected = []
    for shard in index["shards"]:
        m = SHARD_RE.match(shard["path"])
        if m and m["tree"] in args.tree and m["split"] in args.split and m["layer"] in args.layer:
            selected.append(shard)

    total = sum(s["size"] for s in selected)
    print(f"{len(selected)} shards, {total/1e9:.1f} GB")
    if args.list:
        by = {}
        for s in selected:
            m = SHARD_RE.match(s["path"])
            k = (m["tree"], m["split"], m["layer"])
            a, b = by.get(k, (0, 0))
            by[k] = (a + 1, b + s["size"])
        for k in sorted(by):
            n, b = by[k]
            print(f"  {'/'.join(k):45} {n:4d} shards  {b/1e9:8.2f} GB")
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    for tree in args.tree:
        for name in METADATA:
            p = hf_hub_download(
                repo_id=args.repo, filename=f"{tree}/{name}", repo_type="dataset",
                local_dir=args.dest, token=args.token,
            )
            print(f"metadata {p}")

    state = args.dest / ".geoid_flood_done"
    done = set(state.read_text().split()) if state.exists() else set()

    got = 0
    for i, shard in enumerate(selected, 1):
        if shard["path"] in done:
            got += shard["size"]
            continue
        n = stream_shard(args.repo, shard["path"], args.dest, shard.get("sha256"), args.token)
        got += n
        with state.open("a") as fh:
            fh.write(shard["path"] + "\n")
        print(f"[{i}/{len(selected)}] {shard['path']}  {got/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

    print(f"\ndone -> {args.dest}/")
    print("configs expect data/geoid-flood and data/geoid-flood-heldout; symlink if --dest differs.")


if __name__ == "__main__":
    main()
