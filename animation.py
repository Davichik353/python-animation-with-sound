"""Покадровая анимация по рисункам и фонограмме. Запуск: python animation.py"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_right
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent / "assets"
FPS = 30
WIDTH, HEIGHT = 1280, 720
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2
# Порядок наложения слоёв. Координаты и позы заданы в assets/timeline.json.
LAYER_ORDER = ("hero", "partner", "umbrella", "screen", "title", "caption")


@lru_cache(maxsize=32)
def drawing(filename):
    with Image.open(ASSETS / filename) as image:
        return image.copy()


def mix(a, b, step, length):
    """Линейная интерполяция целых координат между ключевыми позами."""
    return (a * (length - step) + b * step + length // 2) // length


def paint_layer(canvas, layer, box=None):
    if layer is None:
        return
    sprite = drawing(layer["png"])
    x, y, width, height = layer["box"] if box is None else box
    if sprite.size != (width, height):
        sprite = sprite.resize((width, height), Image.Resampling.BILINEAR)
    canvas.paste(sprite, (x, y), sprite.getchannel("A"))


def paint_pose(pose):
    canvas = drawing(pose["background"]).copy()
    for name in LAYER_ORDER:
        paint_layer(canvas, pose["layers"].get(name))
    return canvas


def inbetween(first, last, step, length):
    amount = step / length
    if first["scene"] != last["scene"]:
        # На монтажной склейке меняется вся сцена, включая свет и надпись.
        return Image.blend(paint_pose(first), paint_pose(last), amount)

    # Обе позы помещаем в промежуточные координаты. Это сохраняет траекторию
    # движения во время смены рисунка (шаг, жест руки, раскрытие зонта).
    left = drawing(first["background"]).copy()
    right = drawing(last["background"]).copy()
    for name in LAYER_ORDER:
        a = first["layers"].get(name)
        b = last["layers"].get(name)
        box = None
        if a and b:
            box = [mix(x, y, step, length) for x, y in zip(a["box"], b["box"])]
        paint_layer(left, a, box)
        paint_layer(right, b, box)
    return Image.blend(left, right, amount)


class Animation:
    def __init__(self):
        timeline = json.loads((ASSETS / "timeline.json").read_text(encoding="utf-8"))
        self.poses = timeline["poses"]
        self.frames = [pose["frame"] for pose in self.poses]
        self.frame_count = self.frames[-1] + 1

    def draw(self, frame):
        index = bisect_right(self.frames, frame) - 1
        pose = self.poses[index]
        if frame == pose["frame"]:
            return paint_pose(pose)
        following = self.poses[index + 1]
        return inbetween(pose, following, frame - pose["frame"],
                         following["frame"] - pose["frame"])


def video_planes(image):
    """Перевод рисунка в цветовое пространство исходника (BT.709, YUV 4:2:0)."""
    rgb = np.asarray(image, dtype=np.int32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 16 + ((47 * r + 157 * g + 16 * b + 128) >> 8)
    u = 128 + ((-26 * r - 87 * g + 113 * b + 128) >> 8)
    v = 128 + ((112 * r - 102 * g - 10 * b + 128) >> 8)
    luma = Image.fromarray(y.astype(np.uint8)).resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    return np.concatenate((np.asarray(luma).ravel(), u.astype(np.uint8).ravel(),
                           v.astype(np.uint8).ravel()))


def render(output, drawings_only=False):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError as error:
            raise RuntimeError("Установите зависимости командой python -m pip install -r requirements.txt") from error
        ffmpeg = get_ffmpeg_exe()
    animation = Animation()
    audio = ASSETS / "sound.m4a"
    detail_path = ASSETS / "detail.h264"
    if not audio.is_file() or (not drawings_only and not detail_path.is_file()):
        raise RuntimeError("Распакуйте весь ZIP, папка assets должна лежать рядом со скриптом.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Временный файл рядом с результатом позволяет заменить его только после успеха.
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp, ExitStack() as stack:
        target = Path(tmp) / "render.mp4"
        decoder = None
        if not drawings_only:
            decoder = stack.enter_context(subprocess.Popen([
                ffmpeg, "-v", "error", "-threads", "2", "-i", str(detail_path),
                "-map", "0:v:0", "-threads", "1", "-f", "rawvideo", "-pix_fmt",
                "yuv420p", "-fps_mode", "passthrough", "pipe:1",
            ], stdout=subprocess.PIPE))
            stack.callback(decoder.stdout.close)

        encoder = stack.enter_context(subprocess.Popen([
            ffmpeg, "-v", "error", "-f", "rawvideo", "-pixel_format", "yuv420p",
            "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
            "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
            "-vf", "setsar=1,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-c:v", "libx265", "-preset", "fast", "-x265-params",
            "lossless=1:pools=none:frame-threads=2:log-level=error",
            "-tag:v", "hvc1", "-color_range", "tv", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-c:a", "copy",
            "-video_track_timescale", "90000", "-movie_timescale", "48000",
            "-movflags", "+faststart", str(target),
        ], stdin=subprocess.PIPE))
        stack.callback(encoder.stdin.close)
        for frame in range(animation.frame_count):
            pixels = video_planes(animation.draw(frame))
            if decoder:
                # Слой хранит только разницу с рисунком. Сложение по модулю 256
                # точно возвращает исходное зерно, блики, частицы и края букв.
                detail = decoder.stdout.read(FRAME_BYTES)
                if len(detail) != FRAME_BYTES:
                    raise RuntimeError(f"Слой детализации оборвался на кадре {frame}.")
                pixels += np.frombuffer(detail, dtype=np.uint8)
                pixels -= np.uint8(128)
            encoder.stdin.write(pixels)
            if frame % 150 == 0:
                print(f"Кадр {frame + 1} из {animation.frame_count}", flush=True)
        encoder.stdin.close()
        if decoder and (decoder.stdout.read(1) or decoder.wait() != 0):
            raise RuntimeError("Слой детализации не совпадает с длиной анимации.")
        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg не смог сохранить MP4.")
        target.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("animation.mp4"),
                        help="куда сохранить готовый MP4")
    parser.add_argument("--drawings-only", action="store_true",
                        help="собрать только редактируемые рисунки, без слоя детализации")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".mp4":
        parser.error("Укажите имя выходного файла с расширением .mp4.")
    if args.output.resolve().is_relative_to(ASSETS):
        parser.error("Выберите файл вне папки исходных рисунков assets.")
    try:
        render(args.output, args.drawings_only)
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(f"Ошибка: {error}")
    print(f"Готово: {args.output.resolve()}")


if __name__ == "__main__":
    main()
