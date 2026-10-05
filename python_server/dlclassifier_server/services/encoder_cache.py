"""Pre-download pretrained encoder weights, and tell when they go stale.

Every pretrained encoder is fetched from the HuggingFace Hub on first use and
cached in ``~/.cache/huggingface/hub`` -- outside the Appose environment, so
rebuilding that environment does not carry the weights with it. A first
training run therefore needs the network, which at a workshop venue is a live
failure mode.

Pre-warming removes that. The cost of pre-warming anything is that it can
silently go out of date, so this module answers the other half too: which
cached weights no longer match the Hub, and how to replace them.

Two decisions worth knowing about, both made after the obvious thing failed:

**Staleness is judged on the weight file, not the repository.** A repository's
commit sha moves when its README changes. Treating that as stale reports every
encoder as out of date whenever an author edits a model card -- measured on
this cache, three repositories whose weights were untouched. The LFS sha256 of
the weights is the thing that matters, and the cache stores each blob under
that same sha256, so the two compare directly without downloading anything.

**Everything in the cache is checked, rather than a recorded
encoder-to-repository mapping.** Mapping an encoder name to its repository
cannot be done reliably: ``timm-mobilenetv3_large_100`` resolves by name to
``timm/mobilenetv3_large_100.ra_in1k`` but actually loads
``timm/tf_mobilenetv3_large_100.in1k``. Watching the cache grow does not work
either, because an encoder that is already cached adds nothing, and the
recorded access time does not advance on a cache hit. Checking every cached
repository needs none of that, and it covers the histology and foundation
encoders as well as the small ImageNet ones.
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Mirrors FastPretrainedHandler.BACKBONES on the Java side. The two are halves
# of one contract: an encoder the dialog offers but nothing pre-warms re-opens
# the hole this module exists to close. FastPretrainedBackbonesTest checks the
# lists agree.
OFFERED_ENCODERS = [
    "timm-tf_efficientnet_lite0",
    "tu-repghostnet_050",
    "tu-efficientvit_b0",
    "timm-mobilenetv3_small_100",
    "tu-mobilenetv4_conv_small",
    "timm-mobilenetv3_large_100",
]

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".ckpt")

CURRENT = "current"
STALE = "stale"
UNKNOWN = "unknown"


@dataclass
class RepoStatus:
    """Whether one cached repository's weights still match the Hub."""

    repo_id: str
    state: str = UNKNOWN
    detail: str = ""
    size_on_disk: int = 0
    revisions: List[str] = field(default_factory=list)


def prewarm(encoders: Optional[List[str]] = None) -> List[str]:
    """Download each encoder's ImageNet weights into the shared cache.

    Building the model is what populates the cache, and it is the only step
    that needs the network. Running it twice is cheap: the second run resolves
    from cache.

    :param encoders: encoder names, defaulting to the ones the dialog offers
    :return: the encoders that are now usable offline
    """
    import segmentation_models_pytorch as smp

    encoders = encoders or list(OFFERED_ENCODERS)
    done = []
    for name in encoders:
        try:
            smp.Unet(
                encoder_name=name, encoder_weights="imagenet", in_channels=3, classes=2
            )
            done.append(name)
            logger.info("Ready offline: %s", name)
        except Exception as e:
            logger.warning("Could not pre-warm encoder %s: %s", name, e)
    return done


def classify_weights(cached: Set[str], remote: Set[str]) -> str:
    """Decide one repository's state from its cached and remote weight hashes.

    Separated from the cache walk so the decision can be tested without a
    network or a populated cache, which is where the subtle cases live: an
    overlap means the weights we hold are still published, and an empty remote
    set means the Hub told us nothing rather than that the weights are gone.

    :param cached: content hashes of the weight files on disk
    :param remote: content hashes of the weight files the Hub lists
    :return: one of CURRENT, STALE, UNKNOWN
    """
    if not cached or not remote:
        return UNKNOWN
    return CURRENT if cached & remote else STALE


def _remote_weight_hashes(repo_id: str) -> Set[str]:
    """Content hashes of a repository's weight files, without downloading them."""
    from huggingface_hub import HfApi

    api = HfApi()
    weights = [f for f in api.list_repo_files(repo_id) if f.endswith(WEIGHT_SUFFIXES)]
    if not weights:
        return set()
    hashes = set()
    for info in api.get_paths_info(repo_id, paths=weights):
        lfs = getattr(info, "lfs", None)
        # A large weight file lives in LFS and the cache names its blob after
        # the LFS sha256. A small one is a plain git blob, named after its git
        # object id instead.
        sha = getattr(lfs, "sha256", None) if lfs else None
        hashes.add(sha or getattr(info, "blob_id", None))
    return {h for h in hashes if h}


def status(include_all: bool = True) -> List[RepoStatus]:
    """Report every cached repository's weight freshness.

    :param include_all: kept for callers that want only weight-bearing repos;
        repositories with no weight file are reported as unknown either way
    :return: one status per cached repository, worst first
    """
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
    except Exception as e:
        logger.warning("Could not scan the HuggingFace cache: %s", e)
        return []

    out = []
    for repo in sorted(cache.repos, key=lambda r: r.repo_id):
        cached = set()
        for revision in repo.revisions:
            for f in revision.files:
                if f.file_name.endswith(WEIGHT_SUFFIXES):
                    cached.add(Path(str(f.blob_path)).name)
        row = RepoStatus(
            repo_id=repo.repo_id,
            size_on_disk=repo.size_on_disk,
            revisions=[r.commit_hash[:8] for r in repo.revisions],
        )
        if not cached:
            row.state = UNKNOWN
            row.detail = "no weight file cached"
            out.append(row)
            continue
        try:
            remote = _remote_weight_hashes(repo.repo_id)
        except Exception as e:
            row.state = UNKNOWN
            row.detail = "could not reach the Hub: %s" % str(e)[:80]
            out.append(row)
            continue
        row.state = classify_weights(cached, remote)
        if row.state == STALE:
            row.detail = "newer weights published"
        elif row.state == UNKNOWN:
            row.detail = "the Hub lists no weight file"
        out.append(row)
    order = {STALE: 0, UNKNOWN: 1, CURRENT: 2}
    return sorted(out, key=lambda r: (order.get(r.state, 3), r.repo_id))


def refresh(encoders: Optional[List[str]] = None, dry_run: bool = False) -> List[str]:
    """Drop stale cached revisions so the next use downloads current weights.

    Deleting rather than overwriting is deliberate: the Hub client keys the
    cache by revision, so a stale revision would sit there forever otherwise,
    and a model built offline would keep loading it.

    :param encoders: encoders to pre-warm again afterwards; defaults to the
        offered set. Pass an empty list to only delete.
    :param dry_run: report what would be deleted and delete nothing
    :return: the repository ids that were stale
    """
    from huggingface_hub import scan_cache_dir

    stale = [r for r in status() if r.state == STALE]
    if not stale:
        logger.info("Every cached encoder is current")
        return []
    names = [r.repo_id for r in stale]
    logger.info("Stale: %s", ", ".join(names))
    if dry_run:
        return names

    cache = scan_cache_dir()
    revisions = [
        rev.commit_hash
        for repo in cache.repos
        if repo.repo_id in set(names)
        for rev in repo.revisions
    ]
    strategy = cache.delete_revisions(*revisions)
    logger.info("Freeing %s", strategy.expected_freed_size_str)
    strategy.execute()
    prewarm(encoders if encoders is not None else None)
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "action", choices=["prewarm", "status", "refresh"], help="what to do"
    )
    parser.add_argument(
        "--encoder",
        action="append",
        dest="encoders",
        help="limit pre-warming to this encoder; repeatable",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="refresh: report, delete nothing"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.action == "prewarm":
        done = prewarm(args.encoders)
        print("%d encoder(s) ready offline" % len(done))
        return 0
    if args.action == "refresh":
        names = refresh(args.encoders, args.dry_run)
        print("%d stale repositor%s" % (len(names), "y" if len(names) == 1 else "ies"))
        return 0
    rows = status()
    if not rows:
        print("nothing cached")
        return 0
    width = max(len(r.repo_id) for r in rows)
    for r in rows:
        print(
            "%-*s  %-8s %6.0f MB  %s"
            % (width, r.repo_id, r.state, r.size_on_disk / 1e6, r.detail)
        )
    stale = sum(1 for r in rows if r.state == STALE)
    print(
        "\n%d repositor%s cached, %d stale"
        % (len(rows), "y" if len(rows) == 1 else "ies", stale)
    )
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
