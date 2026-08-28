from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

WORKING_DIR = Path(
    r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\PassivePuff_HighFreqImaging\HighFreqImaging_202608"
)

# This is the ops file that your Suite2p GUI is already using:
OPS_PATH = Path(
    r"C:\Users\maxyc\.suite2p\ops\ops.npy"
)

# False = skip datasets that already contain suite2p/plane0/stat.npy
# True  = run everything again
RERUN_COMPLETED = False


# Folder names allowed:
#   20260724_FOV1
#   20260724_FOV2
#   20260724_ROI1
#   20260724_ROI4
FOLDER_PATTERN = re.compile(
    r"^\d{8}_(?:FOV|ROI)\d+$",
    re.IGNORECASE,
)


# ============================================================
# FIND DATASETS
# ============================================================

datasets = sorted(
    p for p in WORKING_DIR.rglob("*")
    if p.is_dir() and FOLDER_PATTERN.fullmatch(p.name)
)

print("=" * 80)
print("Suite2p batch runner")
print("=" * 80)
print(f"Working directory:")
print(f"  {WORKING_DIR}")
print()
print(f"Found {len(datasets)} dataset(s):")

for i, folder in enumerate(datasets, start=1):
    print(f"  {i:02d}. {folder}")

print("=" * 80)


if not datasets:
    print("No matching folders found.")
    sys.exit(0)


# ============================================================
# RUN SUITE2P
# ============================================================

success = []
failed = []
skipped = []


for i, folder in enumerate(datasets, start=1):

    print()
    print("=" * 80)
    print(f"[{i}/{len(datasets)}] {folder.name}")
    print(f"Path: {folder}")
    print("=" * 80)

    stat_path = folder / "suite2p" / "plane0" / "stat.npy"

    # --------------------------------------------------------
    # Skip completed datasets
    # --------------------------------------------------------

    if stat_path.exists() and not RERUN_COMPLETED:
        print("Already processed:")
        print(f"  {stat_path}")
        print("Skipping.")
        skipped.append(folder)
        continue

    # --------------------------------------------------------
    # Create dataset-specific db.npy
    # --------------------------------------------------------

    db = {
        "data_path": [str(folder)],
        "subfolders": [],
        "save_path0": str(folder),
        "fast_disk": str(folder),
        "input_format": "tif",
    }

    db_path = folder / "_suite2p_batch_db.npy"
    np.save(db_path, db)

    log_path = folder / "suite2p_batch.log"

    command = [
        sys.executable,
        "-u",
        "-W",
        "ignore",
        "-m",
        "suite2p",
        "--ops",
        str(OPS_PATH),
        "--db",
        str(db_path),
    ]

    print("Starting Suite2p...")
    print(f"Log:")
    print(f"  {log_path}")
    print()

    start_time = datetime.now()

    # --------------------------------------------------------
    # Run Suite2p and save ALL console output
    # --------------------------------------------------------

    with open(log_path, "a", encoding="utf-8") as log:

        log.write("\n")
        log.write("=" * 80 + "\n")
        log.write(f"Batch started: {start_time}\n")
        log.write(f"Dataset: {folder}\n")
        log.write(f"Command: {' '.join(command)}\n")
        log.write("=" * 80 + "\n")
        log.flush()

        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(folder),
        )

        end_time = datetime.now()

        log.write("\n")
        log.write(f"Finished: {end_time}\n")
        log.write(f"Elapsed: {end_time - start_time}\n")
        log.write(f"Return code: {result.returncode}\n")
        log.flush()

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if result.returncode == 0 and stat_path.exists():
        print(f"SUCCESS: {folder.name}")
        print(f"Elapsed: {datetime.now() - start_time}")
        success.append(folder)

    else:
        print(f"FAILED: {folder.name}")
        print(f"Return code: {result.returncode}")
        print(f"Check:")
        print(f"  {log_path}")
        failed.append(folder)

        # IMPORTANT:
        # continue to the next dataset rather than killing
        # the entire overnight batch
        continue


# ============================================================
# SUMMARY
# ============================================================

print()
print()
print("=" * 80)
print("BATCH FINISHED")
print("=" * 80)

print(f"Successful : {len(success)}")
print(f"Skipped    : {len(skipped)}")
print(f"Failed     : {len(failed)}")

if success:
    print("\nSuccessful:")
    for p in success:
        print(f"  {p.name}")

if skipped:
    print("\nSkipped:")
    for p in skipped:
        print(f"  {p.name}")

if failed:
    print("\nFAILED:")
    for p in failed:
        print(f"  {p.name}")
        print(f"    {p / 'suite2p_batch.log'}")

print("=" * 80)