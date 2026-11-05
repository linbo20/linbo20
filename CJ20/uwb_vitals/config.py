from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    fs: float = 100.0
    rx: int = 8
    taps: int = 32
    first_tap: int = 65
    frames_per_file: int = 100
    window_s: float = 30.0
    step_s: float = 1.0
    respiration_band: tuple = (0.1, 0.6)
    heart_band: tuple = (0.8, 2.5)
    motion_threshold: float = 0.08
    motion_mad_factor: float = 8.0
    guard_s: float = 3.0
    edge_s: float = 5.0
    min_spectral_fraction: float = 0.35
