"""mp4 녹화 모듈 - imageio + ffmpeg 백엔드"""
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import pygame


class VideoRecorder:
    """pygame Surface를 한 프레임씩 mp4로 누적 저장.

    사용 예:
        rec = VideoRecorder('out.mp4', fps=30)
        rec.capture(screen)  # 매 프레임
        rec.close()
    """

    def __init__(self, output_path: str, fps: int = 30, codec: str = 'libx264',
                 quality: int = 8):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: Optional[imageio.core.format.Format.Writer] = imageio.get_writer(
            str(self.output_path),
            fps=fps,
            codec=codec,
            quality=quality,
            macro_block_size=1,
        )
        self.frame_count = 0

    def capture(self, surface: pygame.Surface):
        if self._writer is None:
            return
        # pygame surfarray는 (W, H, 3), imageio는 (H, W, 3) 기대
        frame = pygame.surfarray.array3d(surface).swapaxes(0, 1).astype(np.uint8)
        self._writer.append_data(frame)
        self.frame_count += 1

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0
