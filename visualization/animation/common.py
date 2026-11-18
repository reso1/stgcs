"""Shared playback and GIF encoding for saved trajectories."""
from pathlib import Path

import numpy as np
from PIL import Image


class GifAnimation:
    @staticmethod
    def position(segments, time):
        index = min(int(np.searchsorted(segments[:, 1, -1], time, side="left")), len(segments) - 1)
        start, end = segments[index]
        duration = end[-1] - start[-1]
        alpha = np.clip((time - start[-1]) / duration, 0, 1) if duration > 1e-12 else 1.0
        return start[:-1] + alpha * (end[:-1] - start[:-1])

    @staticmethod
    def times(start, end, fps, speed):
        if not np.isfinite(fps) or fps <= 0 or not np.isfinite(speed) or speed <= 0:
            raise ValueError("Playback speed and frame rate must be finite and positive")
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            raise ValueError("Animation times must be finite and ordered")
        return [float(t) for t in np.arange(start, end, speed / fps) if t < end - 1e-9] + [end]

    @staticmethod
    def save(frames, output, fps):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        durations = [round(1000 / fps)] * len(frames)
        durations[0], durations[-1] = 500, 1000
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, disposal=1, optimize=False)
        with Image.open(output) as gif:
            if gif.n_frames != len(frames) or gif.info["loop"] != 0:
                raise ValueError("Saved GIF does not contain the expected animation")
            for index in range(gif.n_frames):
                gif.seek(index)
                gif.load()
