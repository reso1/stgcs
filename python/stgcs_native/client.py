"""Executable discovery, builds, and subprocess management for the native tools."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import threading
import time


class NativeClient:
    ROOT = Path(__file__).resolve().parents[2]
    SOURCE_ROOT = ROOT / "stgcs_cplusplus"
    BUILD_ROOT = SOURCE_ROOT / "build"
    PROCESSES = set()
    LOCK = threading.Lock()
    STOPPED = threading.Event()

    @classmethod
    def executable(cls, name: str) -> Path:
        variables = {
            "stgcs_planner": "STGCS_NATIVE_EXECUTABLE",
            "stgcs_trajopt": "STGCS_TRAJOPT_EXECUTABLE",
        }
        variable = variables[name]
        path = Path(os.environ.get(variable, cls.BUILD_ROOT / "bin" / name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing native executable: {path}. Build with "
                "cmake -S stgcs_cplusplus -B stgcs_cplusplus/build "
                "-DCMAKE_PREFIX_PATH=/path/to/drake -DCMAKE_BUILD_TYPE=Release, then "
                f"cmake --build stgcs_cplusplus/build -j4; or set {variable}.")
        return path

    @classmethod
    def rebuild(cls, executable: Path) -> Path:
        build_dir = executable.resolve().parent.parent
        cache = (build_dir / "CMakeCache.txt").read_text().splitlines()
        drake_dir = next(line.split("=", 1)[1] for line in cache if line.startswith("drake_DIR:"))
        return cls.build(build_dir, Path(drake_dir))

    @classmethod
    def build(cls, build_dir: Path, drake_dir: Path) -> Path:
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("CMake is required to build the native ST-GCS tools")
        build_dir.mkdir(parents=True, exist_ok=True)
        with (build_dir / "build.log").open("w") as log:
            subprocess.run([cmake, "-S", str(cls.SOURCE_ROOT), "-B", str(build_dir),
                            f"-Ddrake_DIR={drake_dir}", "-DCMAKE_BUILD_TYPE=Release"],
                           check=True, stdout=log, stderr=log)
            subprocess.run([cmake, "--build", str(build_dir), "--config", "Release", "-j", "4"],
                           check=True, stdout=log, stderr=log)
        return build_dir / "bin/stgcs_planner"

    @classmethod
    def stop(cls):
        cls.STOPPED.set()
        with cls.LOCK:
            for process in cls.PROCESSES:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, 15)
                    except ProcessLookupError:
                        pass

    @classmethod
    def active_processes(cls):
        with cls.LOCK:
            return [{"pid": process.pid, "command": process.args} for process in cls.PROCESSES
                    if process.poll() is None]

    @classmethod
    def run_process(cls, command, timeout: float | None, log_path: Path, expect_json=True, progress=False) -> dict:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("w") as log:
            with cls.LOCK:
                if cls.STOPPED.is_set():
                    raise InterruptedError("Native execution stopped")
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, start_new_session=True)
                cls.PROCESSES.add(process)
            reader = threading.Thread(target=cls._capture_output, args=(process, log, progress))
            reader.start()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, 15)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                    process.wait()
                return {"event": "result", "success": False, "status": "hard_timeout",
                        "runtime": timeout, "cost": None, "expanded": 0, "generated": 0,
                        "wall_time": time.monotonic() - started}
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, 15)
                    process.wait()
                raise
            finally:
                reader.join()
                process.stdout.close()
                with cls.LOCK:
                    cls.PROCESSES.discard(process)
        if process.returncode:
            raise RuntimeError(
                f"Worker exited {process.returncode}; see {log_path}: "
                f"{log_path.read_text()[-4000:].strip()}")
        if not expect_json:
            return {"exit_code": process.returncode}
        events = [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith("{")]
        if not events:
            raise RuntimeError(f"Worker returned no result: {log_path}")
        return events[-1]

    @classmethod
    def request(cls, executable: str, mode: str, words, progress: bool = False) -> dict:
        binary = cls.executable(executable)
        with TemporaryDirectory(prefix="stgcs-request-") as work:
            source = Path(work) / "input.txt"
            source.write_text(" ".join(map(str, words)) + "\n")
            return cls.run_process([str(binary), mode, str(source)], None,
                                   Path(work) / "output.log", progress=progress)

    @staticmethod
    def _capture_output(process, log, progress):
        for line in process.stdout:
            log.write(line)
            log.flush()
            if progress and not line.startswith("{"):
                print(line, end="", flush=True)
