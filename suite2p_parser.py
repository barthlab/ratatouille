import os
from glob import glob
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from distinctipy import distinctipy


def search_pattern_file(pattern: str, search_dir: str) -> List[str]:
    recursive_path = os.path.join(search_dir, '**', pattern)
    return list(glob(recursive_path, recursive=True))


def parse_suite2p(p: Path):
    print(f"Parsing {p}...")
    ops = np.load(os.path.join(p, "ops.npy"), allow_pickle=True).item()
    stat = np.load(os.path.join(p, "stat.npy"), allow_pickle=True)
    iscell = np.load(os.path.join(p, "iscell.npy"), allow_pickle=True)

    # Pick which mean image to use
    mean_img = ops.get("meanImg", None)
    if mean_img is None:
        mean_img = ops.get("meanImgE", None)
    if mean_img is None:
        raise ValueError("Could not find meanImg or meanImgE in ops.npy")

    # Use only accepted cells   
    cell_inds = np.where(iscell[:, 0] == 1)[0]
    
    # Generate a color palette for ROIs
    colors = distinctipy.get_colors(len(cell_inds), pastel_factor=0.5)
    colors_inverted = distinctipy.invert_colors(colors)


    # Build ROI centers and a colored overlay
    centers = []
    overlay = np.zeros((*mean_img.shape, 4), dtype=float)
    for order, i in enumerate(cell_inds):
        ypix = stat[i]["ypix"]
        xpix = stat[i]["xpix"]
        centers.append((float(np.mean(xpix)), float(np.mean(ypix)), order))        
        overlay[ypix, xpix, :3] = colors[order]
        overlay[ypix, xpix, 3] = 0.8


    # Plotting
    fig_width = mean_img.shape[1] / 50.0
    fig_height = mean_img.shape[0] / 50.0
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height*2.1), constrained_layout=True)

    axes[0].imshow(mean_img, cmap="gray", interpolation="nearest")
    axes[0].axis("off")

    axes[1].imshow(mean_img, cmap="gray", interpolation="nearest")
    axes[1].imshow(overlay, interpolation="nearest", zorder=10)
    for (x, y, idx), color in zip(centers, colors_inverted):
        axes[1].text(x, y, str(idx), color=color, fontsize=16, ha="center", va="center", zorder=20)
    axes[1].axis("off")

    out_png = os.path.join(p, "..", "..", "ROIs_overlay.png")
    fig.savefig(out_png, dpi=200, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


def parse_all_suite2p_under_dir(dir_path: str):
    pattern = os.path.join("suite2p", "plane0")
    all_suite2p_ops = search_pattern_file(pattern, dir_path)
    print(f"Found {len(all_suite2p_ops)} suite2p ops files under {dir_path}, start parsing...")
    for p in Path(dir_path).rglob(pattern):
        parse_suite2p(p)
    print("Parsing complete!")


if __name__ == "__main__":
    parse_all_suite2p_under_dir(r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\PassivePuff_HighFreqImaging\HighFreqImaging_202602")
    
