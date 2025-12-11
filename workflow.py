import os
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logging.error(f"Échec de la commande: {' '.join(cmd)}")
        raise SystemExit(1)
    logging.info(f"Commande terminée: {' '.join(cmd)}")

class Config:
    project_name: str
    video_input_path: str
    fps: int
    convert_resize: bool
    iterations: int

    def __init__(
        self,
        project_name: str,
        video_input_path: str,
        fps: int = 2,
        convert_resize: bool = False,
        iterations: int = 30000
    ) -> None:
        self.project_name = project_name
        self.video_input_path = video_input_path
        self.fps = fps
        self.convert_resize = convert_resize
        self.iterations = iterations

def load_config() -> Config:
    load_dotenv()
    project_name = os.getenv("PROJECT_NAME")
    video_input = os.getenv("VIDEO_INPUT_PATH")

    if not project_name or not video_input:
        logging.error("PROJECT_NAME et VIDEO_INPUT_PATH doivent être définis dans le .env")
        raise SystemExit(1)

    fps = int(os.getenv("FPS", 2))
    convert_resize = os.getenv("CONVERT_RESIZE", "False").lower() == "true"
    iterations = int(os.getenv("ITERATIONS", 30000))

    return Config(
        project_name=project_name,
        video_input_path=video_input,
        fps=fps,
        convert_resize=convert_resize,
        iterations=iterations
    )

def main() -> None:
    config = load_config()

    output_dir = Path("output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    run([
        "ffmpeg",
        "-i", config.video_input_path,
        "-qscale:v", "1",
        "-vf", f"fps={config.fps}",
        str(output_dir / "frame_%04d.png")
    ])

    gs_data_path = Path("gaussian-splatting") / "Data" / config.project_name
    gs_data_path.mkdir(parents=True, exist_ok=True)

    dest_input = gs_data_path / "input"
    shutil.move(str(output_dir), dest_input)

    os.chdir("gaussian-splatting")

    run(["python", "convert.py", "-s", str(Path("Data") / config.project_name)])

    os.chdir("..")

    run([
        "python",
        "Depth-Anything-V2/run.py",
        "--encoder", "vitl",
        "--pred-only",
        "--grayscale",
        "--img-path", str(gs_data_path / "input"),
        "--outdir", str(gs_data_path / "depths")
    ])

    os.chdir("gaussian-splatting")

    run([
        "python",
        "utils/make_depth_scale.py",
        "--base_dir", str(Path("Data") / config.project_name),
        "--depths_dir", str(Path("Data") / config.project_name / "depths")
    ])

    run([
        "python",
        "train.py",
        "-s", str(Path("Data") / config.project_name),
        "-m", str(Path("Data") / config.project_name / "output"),
        "--iterations", str(config.iterations)
    ])

if __name__ == "__main__":
    main()
