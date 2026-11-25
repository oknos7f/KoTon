from pathlib import Path
from PIL import Image, ImageDraw


def create_process_icon(out_path: Path) -> None:
    """Render a 256x256 cyber-neon circular gear/CPU icon for Process entities."""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((16, 16, 240, 240), fill=(10, 24, 48, 235), outline=(0, 245, 255, 255), width=8)
    draw.ellipse((48, 48, 208, 208), outline=(56, 189, 248, 180), width=4)
    draw.rectangle((88, 88, 168, 168), fill=(14, 116, 144, 220), outline=(0, 245, 255, 255), width=5)
    for offset in (104, 128, 152):
        draw.line((offset, 64, offset, 88), fill=(0, 245, 255, 255), width=5)
        draw.line((offset, 168, offset, 192), fill=(0, 245, 255, 255), width=5)
        draw.line((64, offset, 88, offset), fill=(0, 245, 255, 255), width=5)
        draw.line((168, offset, 192, offset), fill=(0, 245, 255, 255), width=5)
    draw.ellipse((114, 114, 142, 142), fill=(224, 242, 254, 255))
    img.save(out_path, format="PNG")


def create_file_icon(out_path: Path) -> None:
    """Render a 256x256 cyber-neon square document icon for File entities."""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((24, 24, 232, 232), radius=20, fill=(18, 28, 45, 235), outline=(52, 211, 153, 255), width=8)
    poly = [(72, 56), (156, 56), (188, 88), (188, 200), (72, 200)]
    draw.polygon(poly, fill=(6, 78, 59, 210), outline=(110, 231, 183, 255))
    draw.polygon([(156, 56), (156, 88), (188, 88)], fill=(52, 211, 153, 255))
    for y in (116, 142, 168):
        draw.line((92, y, 168, y), fill=(209, 250, 229, 240), width=6)
    img.save(out_path, format="PNG")


def create_socket_icon(out_path: Path) -> None:
    """Render a 256x256 cyber-neon diamond network node icon for Socket entities."""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    diamond = [(128, 14), (242, 128), (128, 242), (14, 128)]
    draw.polygon(diamond, fill=(30, 18, 52, 235), outline=(192, 132, 252, 255))
    inner = [(128, 46), (210, 128), (128, 210), (46, 128)]
    draw.polygon(inner, outline=(232, 121, 249, 200))
    nodes = [(128, 76), (80, 148), (176, 148)]
    draw.line((nodes[0], nodes[1]), fill=(240, 171, 252, 255), width=5)
    draw.line((nodes[0], nodes[2]), fill=(240, 171, 252, 255), width=5)
    draw.line((nodes[1], nodes[2]), fill=(240, 171, 252, 255), width=5)
    for cx, cy in nodes:
        draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=(250, 232, 255, 255), outline=(192, 132, 252, 255), width=3)
    img.save(out_path, format="PNG")


def main() -> None:
    """Generate fallback PNG HUD icons under web/images/ only when missing, preserving user assets."""
    out_dir = Path(__file__).resolve().parents[1] / "web" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "process.png").exists():
        create_process_icon(out_dir / "process.png")
    if not (out_dir / "file.png").exists():
        create_file_icon(out_dir / "file.png")
    if not (out_dir / "socket.png").exists():
        create_socket_icon(out_dir / "socket.png")


if __name__ == "__main__":
    main()
