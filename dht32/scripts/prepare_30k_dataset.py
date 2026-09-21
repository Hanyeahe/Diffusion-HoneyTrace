#!/usr/bin/env python
import os
import shutil
import time
from pathlib import Path


def main():
    src_root = Path("/private/projects/DHT-32/outputs/dataset_protected_15000_sigmoid_K8_s003_L50_1000step")
    dst_root = Path("/private/projects/DHT-32/outputs/dataset_protected_30000_sigmoid_K8_s003_L50_1000step")
    src = src_root / "images_protected"
    dst = dst_root / "images_protected"
    dst.mkdir(parents=True, exist_ok=True)

    for name in ["watermark_state.pt", "config.json", "metrics.json", "scores_and_projection.csv"]:
        sp = src_root / name
        if sp.exists():
            dp = dst_root / ("source15k_" + name if name != "watermark_state.pt" else name)
            if not dp.exists():
                shutil.copy2(sp, dp)

    count = 0
    linked = 0
    copied = 0
    started = time.time()
    for sp in sorted(src.glob("*.png")):
        dp = dst / sp.name
        if dp.exists():
            count += 1
            continue
        try:
            os.link(sp, dp)
            linked += 1
        except OSError:
            shutil.copy2(sp, dp)
            copied += 1
        count += 1
        if count % 2000 == 0:
            print(
                f"prepared {count}/15000 linked={linked} copied={copied} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    print(f"done_prepare count={sum(1 for _ in dst.glob('*.png'))} linked={linked} copied={copied}")


if __name__ == "__main__":
    main()
