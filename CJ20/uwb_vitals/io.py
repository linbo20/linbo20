import re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np


def load_recording(folder, cfg):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"BIN 文件夹不存在: {folder}")
    stamp = next((re.fullmatch(r"radar_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", p.name)
                  for p in (folder, *folder.parents)
                  if re.fullmatch(r"radar_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", p.name)), None)
    if stamp is None:
        raise ValueError("输入目录或其父目录必须包含 radar_YYYY-MM-DD-HH-MM-SS")
    start = datetime.strptime(stamp[1], "%Y-%m-%d-%H-%M-%S").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    paths = list(folder.glob("COM*_radar_*.bin"))
    if not paths:
        raise FileNotFoundError("没有找到 COM*_radar_*.bin")
    matches = [re.fullmatch(r"(COM\d+)_radar_(\d+)", p.stem) for p in paths]
    if any(m is None for m in matches):
        raise ValueError("文件名必须为 COM数字_radar_数字.bin")
    if len({m[1] for m in matches}) != 1:
        raise ValueError("发现多个 COM 数据流，请将需要的单路数据放入独立目录")
    entries = sorted(zip(paths, matches), key=lambda item: int(item[1][2]))
    numbers = [int(m[2]) for _, m in entries]
    if any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
        raise ValueError("BIN 编号重复或缺号，无法可靠推算时间，请先检查采集完整性")
    if numbers[0] not in (0, 1):
        raise ValueError("首文件编号必须为 0 或 1，以确认目录开始时间对应首帧")
    sizes = []
    for i, (p, _) in enumerate(entries):
        size = p.stat().st_size
        frame_bytes = cfg.rx * cfg.taps * 4
        if size == 0 or size % frame_bytes:
            raise ValueError(f"{p.name}: 字节数 {size} 不符合无帧头 Q/I int16 结构")
        n = size // frame_bytes
        if n > cfg.frames_per_file or (i < len(entries)-1 and n != cfg.frames_per_file):
            raise ValueError(f"{p.name}: {n} 帧，非末文件必须恰好 100 帧，末文件允许不足 100 帧")
        sizes.append(n)
    cir = np.empty((sum(sizes), cfg.rx, cfg.taps), dtype=np.complex64)
    manifest = []
    offset = 0
    for (p, _), n in zip(entries, sizes):
        iq = np.fromfile(p, dtype="<i2").reshape(n, cfg.rx, cfg.taps, 2)
        cir[offset:offset+n] = iq[..., 1].astype(np.float32) + 1j * iq[..., 0].astype(np.float32)
        manifest.append(dict(file=p.name, first_frame=offset, frames=n))
        offset += n
    return cir, start, manifest
