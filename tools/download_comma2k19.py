"""Download a small subset of the comma2k19 dataset.

comma2k19 (https://github.com/commaai/comma2k19) is only distributed as a ~100GB
BitTorrent (academictorrents infohash 65a2fbc964078aff62076ff4e103f18b951c5ddb).
Despite the per-segment folder layout documented in the upstream README, the
torrent itself contains only 10 whole-chunk zip files (comma2k19/Chunk_1.zip ..
Chunk_10.zip, ~9-10GB each) -- BitTorrent can only select whole files, so there
is no way to fetch individual segments over the wire. This script instead:

  1. Downloads exactly one chunk zip (whichever --chunks you ask for).
  2. Extracts only the requested segment folders from inside that zip (without
     unpacking the other ~200 segments in the same chunk).
  3. Deletes the zip afterward, unless --keep-zip is passed.

Usage:
    python tools/download_comma2k19.py --out data/comma2k19/raw --chunks 1 --max-segments 15

route_ids inside the dataset contain a literal "|" (dongle_id|start_time), which
is not a legal character in a Windows path -- extraction sanitizes it (and any
other reserved characters) to "_" when writing to disk.

Tries, in order:
  1. `libtorrent` (pure local download) -- often unavailable on Windows since
     prebuilt wheels lag behind new Python releases.
  2. A running qBittorrent's Web UI, driven via `qbittorrent-api` (pure-Python,
     installs anywhere). Enable it once in qBittorrent under
     Tools > Options > Web UI, then pass --qbt-username/--qbt-password (or set
     QBT_USERNAME/QBT_PASSWORD env vars).
  3. Manual instructions, printed to stdout, and the script exits non-zero so a
     calling notebook cell doesn't silently continue with an empty directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

INFOHASH = "65a2fbc964078aff62076ff4e103f18b951c5ddb"
TRACKERS = [
    "udp://tracker.openbittorrent.com:80/announce",
    "udp://tracker.opentrackr.org:1337/announce",
    "http://academictorrents.com/announce.php",
]
MAGNET = f"magnet:?xt=urn:btih:{INFOHASH}&dn=comma2k19" + "".join(f"&tr={t}" for t in TRACKERS)

CHUNK_ZIP_RE = re.compile(r"Chunk_(\d+)\.zip$", re.IGNORECASE)
INVALID_WIN_CHARS_RE = re.compile(r'[<>:"|?*]')


def select_chunk_zip_paths(file_paths: list[str], chunks: list[int]) -> list[str]:
    wanted = []
    for path in file_paths:
        m = CHUNK_ZIP_RE.search(path)
        if m and int(m.group(1)) in chunks:
            wanted.append(path)
    return wanted


def sanitize_for_windows(rel_path: str) -> str:
    return "/".join(INVALID_WIN_CHARS_RE.sub("_", part) for part in rel_path.split("/"))


def safe_unlink(path: Path, retries: int = 10, delay_s: float = 2.0) -> None:
    """Deletes `path`, retrying on Windows' transient WinError 32 ("being used by another
    process"). The torrent client (qBittorrent/libtorrent) can hold the just-finished zip open
    a few seconds longer for its own post-download hashing/bookkeeping even after our download
    loop sees 100% progress -- deleting immediately races that. The extraction itself (reading
    the zip, writing segment files elsewhere) already succeeded by the time this runs; losing
    this race must not raise, since `keep_zip=False` is just disk cleanup, not a real failure."""
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == retries - 1:
                print(f"  warning: could not delete {path} after {retries} attempts "
                      f"(still locked by the torrent client) -- delete it manually later "
                      f"to reclaim disk space.", file=sys.stderr)
                return
            time.sleep(delay_s)


def extract_segments_from_zip(zip_path: Path, out_dir: Path, segments: list[dict] | None, max_segments: int) -> int:
    """Pulls only a handful of segment folders out of a whole-chunk zip."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        video_members = [n for n in names if n.replace("\\", "/").endswith("video.hevc")]

        if segments:
            wanted_segment_dirs = []
            for s in segments:
                match = next(
                    (n for n in video_members if f"_{s['chunk']}" in n.split("/")[0]
                     and f"/{s['segment']}/" in n and s["route_id"] in n),
                    None,
                )
                if match:
                    wanted_segment_dirs.append(match.rsplit("/", 1)[0] + "/")
        else:
            wanted_segment_dirs = [n.rsplit("/", 1)[0] + "/" for n in video_members[:max_segments]]

        if not wanted_segment_dirs:
            print(f"  no matching segments found inside {zip_path.name}; first few entries:", file=sys.stderr)
            for n in names[:10]:
                print(f"    {n}", file=sys.stderr)
            return 0

        print(f"  extracting {len(wanted_segment_dirs)} segment(s) from {zip_path.name}:")
        n_files = 0
        for seg_dir in wanted_segment_dirs:
            print(f"    {seg_dir}")
            for member in names:
                if member.startswith(seg_dir) and not member.endswith("/"):
                    dest = out_dir / sanitize_for_windows(member)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    n_files += 1
        return len(wanted_segment_dirs)


def print_manual_instructions(out_dir: Path, chunks: list[int], max_segments: int):
    print(
        "\nNo automated download method worked in this environment.\n"
        "To download a small subset of comma2k19 manually:\n"
        f"  1. Install a BitTorrent client (qBittorrent or Transmission).\n"
        f"  2. Open this magnet link:\n     {MAGNET}\n"
        f"  3. The torrent only contains whole chunk zips (Chunk_1.zip .. Chunk_10.zip, "
        f"~9-10GB each) -- select only Chunk_{chunks[0]}.zip.\n"
        f"  4. Set the download location to: {out_dir.resolve()}\n"
        f"  5. Once it finishes, extract just a few segment folders from the zip yourself, "
        f"or re-run this script with --keep-zip pointed at the downloaded file via --out, and "
        f"it will do the extraction step for you (skip straight to that by placing the zip "
        f"at {out_dir.resolve()}/comma2k19/Chunk_{chunks[0]}.zip and re-running with the same "
        f"--chunks/--max-segments).\n"
    )


def download_with_libtorrent(out_dir: Path, chunks: list[int], segments: list[dict] | None, max_segments: int, keep_zip: bool):
    import libtorrent as lt

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ses = lt.session()
    ses.listen_on(6881, 6891)

    params = lt.parse_magnet_uri(MAGNET)
    params.save_path = str(out_dir)
    params.flags |= lt.torrent_flags.upload_mode  # don't seed while fetching metadata
    handle = ses.add_torrent(params)

    print("Fetching torrent metadata...")
    while not handle.status().has_metadata:
        time.sleep(1)
    handle.unset_flags(lt.torrent_flags.upload_mode)
    info = handle.torrent_file()
    print(f"Metadata received: {info.num_files()} files in the torrent.")

    file_paths = [info.files().file_path(i) for i in range(info.num_files())]
    wanted_zips = select_chunk_zip_paths(file_paths, chunks)
    if not wanted_zips:
        raise RuntimeError(f"Could not find a chunk zip for chunks={chunks} in the torrent.")

    print(f"Downloading whole chunk zip(s): {wanted_zips}")
    priorities = [0] * info.num_files()
    for i, path in enumerate(file_paths):
        if path in wanted_zips:
            priorities[i] = 4
    handle.prioritize_files(priorities)

    while True:
        s = handle.status()
        wanted_done = sum(handle.file_progress()[i] for i in range(info.num_files()) if priorities[i] > 0)
        wanted_total = sum(info.files().file_size(i) for i in range(info.num_files()) if priorities[i] > 0)
        pct = 100.0 * wanted_done / wanted_total if wanted_total else 100.0
        print(f"\r  {pct:5.1f}% | down {s.download_rate / 1000:.1f} kB/s | peers {s.num_peers}", end="")
        if wanted_done >= wanted_total:
            break
        time.sleep(2)
    print()

    n_segments = 0
    for zip_rel_path in wanted_zips:
        zip_path = out_dir / zip_rel_path
        n_segments += extract_segments_from_zip(zip_path, out_dir, segments, max_segments)
        if not keep_zip:
            safe_unlink(zip_path)
    print(f"Done. {n_segments} segment(s) extracted under {out_dir.resolve()}")


def download_with_qbittorrent(
    out_dir: Path,
    chunks: list[int],
    segments: list[dict] | None,
    max_segments: int,
    keep_zip: bool,
    host: str,
    port: int,
    username: str,
    password: str,
):
    import qbittorrentapi

    client = qbittorrentapi.Client(host=f"{host}:{port}", username=username, password=password)
    client.auth_log_in()  # raises qbittorrentapi.LoginFailed on bad credentials

    out_dir = out_dir.resolve()  # qBittorrent resolves a relative save_path against its own
    out_dir.mkdir(parents=True, exist_ok=True)  # default download dir, not this script's cwd
    torrent_hash = INFOHASH.lower()
    if not client.torrents_info(torrent_hashes=torrent_hash):
        client.torrents_add(urls=MAGNET, save_path=str(out_dir), category="comma2k19")

    print("Fetching torrent metadata via qBittorrent...")
    files = []
    for _ in range(180):
        try:
            files = client.torrents_files(torrent_hash=torrent_hash)
        except Exception:  # noqa: BLE001 -- 404 until qBittorrent has resolved magnet metadata
            files = []
        if files:
            break
        time.sleep(1)
    if not files:
        raise RuntimeError(
            "Torrent metadata never arrived after 180s (no reachable peers/seeds for this magnet?)."
        )
    print(f"Metadata received: {len(files)} files in the torrent.")

    file_paths = [f.name for f in files]
    wanted_zips = select_chunk_zip_paths(file_paths, chunks)
    if not wanted_zips:
        raise RuntimeError(f"Could not find a chunk zip for chunks={chunks} in the torrent.")

    print(f"Downloading whole chunk zip(s): {wanted_zips}")
    want_ids = [f.id for f in files if f.name in wanted_zips]
    skip_ids = [f.id for f in files if f.name not in wanted_zips]
    if skip_ids:
        client.torrents_file_priority(torrent_hash=torrent_hash, file_ids=skip_ids, priority=0)
    client.torrents_file_priority(torrent_hash=torrent_hash, file_ids=want_ids, priority=1)

    # qBittorrent's own progress/piece-completion bookkeeping is independent of the actual
    # filesystem: if a previous run (or our own --keep-zip=False cleanup) deleted a zip that
    # qBittorrent had already verified, it still reports progress=1.0 for that file -- it only
    # notices the file is gone after an explicit recheck. Compare on-disk size to the expected
    # size up front and force a recheck if anything's missing/short, so the polling loop below
    # reflects what's actually on disk, not stale in-memory state.
    expected_size = {f.name: f.size for f in [f for f in files if f.id in want_ids]}
    on_disk_matches = all(
        (out_dir / rel_path).exists() and (out_dir / rel_path).stat().st_size == expected_size[rel_path]
        for rel_path in wanted_zips
    )
    if not on_disk_matches:
        print("  on-disk file(s) missing/incomplete relative to qBittorrent's last known state "
              "-- forcing a recheck...")
        client.torrents_recheck(torrent_hashes=torrent_hash)
        for _ in range(60):
            torrent = client.torrents_info(torrent_hashes=torrent_hash)[0]
            if torrent.state not in ("checkingDL", "checkingUP", "checkingResumeData", "allocating"):
                break
            time.sleep(2)

    while True:
        files = client.torrents_files(torrent_hash=torrent_hash)
        wanted = [f for f in files if f.id in want_ids]
        done = sum(f.size * f.progress for f in wanted)
        total = sum(f.size for f in wanted)
        torrent = client.torrents_info(torrent_hashes=torrent_hash)[0]
        pct = 100.0 * done / total if total else 100.0
        print(f"\r  {pct:5.1f}% | down {torrent.dlspeed / 1000:.1f} kB/s | peers {torrent.num_leechs + torrent.num_seeds}", end="")
        if all(f.progress >= 1.0 for f in wanted):
            break
        time.sleep(2)
    print()

    # qBittorrent reporting progress==1.0 (piece-hash verified) doesn't guarantee the file has
    # finished being flushed/moved into its final on-disk location yet -- reading the zip too
    # early can land mid-move and corrupt the read (seen in practice: a zlib "invalid stored
    # block lengths" error while extracting, on a torrent that qBittorrent considered 100%
    # done). Wait for the on-disk file size to actually match the expected size before opening
    # the zip, not just the API's progress flag.
    expected_size = {f.name: f.size for f in wanted}
    for rel_path in wanted_zips:
        zip_path = out_dir / rel_path
        for attempt in range(30):
            actual_size = zip_path.stat().st_size if zip_path.exists() else -1
            if actual_size == expected_size[rel_path]:
                break
            time.sleep(2)
        else:
            raise RuntimeError(
                f"{zip_path} never reached its expected size "
                f"({expected_size[rel_path]} bytes) on disk after waiting -- download likely "
                f"incomplete or still being moved by qBittorrent."
            )

    n_segments = 0
    for rel_path in wanted_zips:
        zip_path = out_dir / rel_path
        n_segments += extract_segments_from_zip(zip_path, out_dir, segments, max_segments)
        if not keep_zip:
            safe_unlink(zip_path)
    print(f"Done. {n_segments} segment(s) extracted under {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/comma2k19/raw"))
    parser.add_argument("--chunks", type=int, nargs="+", default=[1])
    parser.add_argument("--max-segments", type=int, default=15)
    parser.add_argument("--segments", type=Path, default=None, help="JSON list of {chunk, route_id, segment}")
    parser.add_argument("--keep-zip", action="store_true", help="Don't delete the chunk zip after extracting the subset")
    parser.add_argument("--qbt-host", default=os.environ.get("QBT_HOST", "localhost"))
    parser.add_argument("--qbt-port", type=int, default=int(os.environ.get("QBT_PORT", "8080")))
    parser.add_argument("--qbt-username", default=os.environ.get("QBT_USERNAME", "admin"))
    parser.add_argument("--qbt-password", default=os.environ.get("QBT_PASSWORD"))
    args = parser.parse_args()

    segments = json.loads(args.segments.read_text()) if args.segments else None

    try:
        download_with_libtorrent(args.out, args.chunks, segments, args.max_segments, args.keep_zip)
        return
    except ImportError:
        print("libtorrent not installed, trying qBittorrent Web UI...", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"libtorrent download failed ({exc}), trying qBittorrent Web UI...", file=sys.stderr)

    if args.qbt_password:
        try:
            download_with_qbittorrent(
                args.out, args.chunks, segments, args.max_segments, args.keep_zip,
                args.qbt_host, args.qbt_port, args.qbt_username, args.qbt_password,
            )
            return
        except Exception as exc:  # noqa: BLE001
            print(f"qBittorrent Web UI download failed: {exc}", file=sys.stderr)
    else:
        print(
            "qBittorrent Web UI not configured (pass --qbt-password or set QBT_PASSWORD).",
            file=sys.stderr,
        )

    print_manual_instructions(args.out, args.chunks, args.max_segments)
    sys.exit(1)


if __name__ == "__main__":
    main()
