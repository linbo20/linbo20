"""只需修改 INPUT_FOLDER，然后运行 python main.py。"""
from pathlib import Path
from uwb_vitals.pipeline import run

INPUT_FOLDER = Path(r"D:\WIFI雷达感知\cir\radar_2026-11-05-19-36-34")

if __name__ == "__main__":
    run(INPUT_FOLDER)
