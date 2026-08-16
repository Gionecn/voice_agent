"""VoiceAgent 本地开发环境的一键启动器。

脚本直接创建两个子进程：项目虚拟环境中的 Uvicorn 后端，以及 ``ui`` 目录内
安装的 Vite 前端。它负责检查依赖、等待端口就绪、监控异常退出，并在 Ctrl+C
或启动失败时统一清理两个服务，避免开发机残留占用 8000/5173 端口的进程。

``--smoke-test`` 模式只验证进程和端口能否启动，成功后立即关闭；它不访问外部
模型，也不能代替真实麦克风和语音链路测试。
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
UI_ROOT = PROJECT_ROOT / "ui"
# 当前脚本面向本机开发环境，前后端仅监听回环地址。
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 5173
STARTUP_TIMEOUT_SECONDS = 20.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class Service:
    """描述一个由启动脚本托管的子服务。

    Attributes:
        name: 用于控制台日志的可读服务名。
        command: 传给 ``subprocess.Popen`` 的参数列表，不经过 shell 拼接。
        cwd: 子进程工作目录；后端在项目根目录，前端在 ``ui`` 目录。
        process: 启动后保存的 Popen 对象，启动前为 ``None``。
    """

    name: str
    command: list[str]
    cwd: Path
    process: subprocess.Popen[bytes] | None = None


def backend_python() -> Path:
    """定位项目虚拟环境中的 Python 解释器。

    Returns:
        Windows 下的 ``.venv/Scripts/python.exe`` 或类 Unix 系统下的
        ``.venv/bin/python``。

    Raises:
        FileNotFoundError: 项目尚未创建虚拟环境或解释器路径无效。
    """

    executable = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else PROJECT_ROOT / ".venv" / "bin" / "python"
    )
    if not executable.is_file():
        raise FileNotFoundError(f"未找到项目 Python 环境：{executable}")
    return executable


def frontend_command() -> list[str]:
    """定位 Node 和项目内 Vite，构造前端启动命令。

    Returns:
        可直接传给 Popen 的参数列表，监听 ``127.0.0.1:5173``。

    Raises:
        FileNotFoundError: 系统 PATH 中找不到 Node，或 ``ui/node_modules`` 尚未
            安装 Vite。

    直接执行 Vite 的 JS 入口而不是 ``yarn dev``，可以避免额外 shell 窗口和
    Windows 命令包装差异，并让父进程更准确地掌握实际服务进程。
    """

    node = shutil.which("node.exe" if os.name == "nt" else "node")
    if not node:
        node = shutil.which("node")
    if not node:
        raise FileNotFoundError("未找到 Node.js，请先安装 Node.js。")

    # 直接调用项目内 Vite，避免 shell 差异和全局依赖版本不一致。
    vite = UI_ROOT / "node_modules" / "vite" / "bin" / "vite.js"
    if not vite.is_file():
        raise FileNotFoundError(
            "未安装前端依赖，请先在 ui 目录执行 corepack yarn install。"
        )
    return [
        node,
        str(vite),
        "--configLoader",
        "runner",
        "--host",
        FRONTEND_HOST,
        "--port",
        str(FRONTEND_PORT),
    ]


def build_services() -> list[Service]:
    """验证工程结构并生成后端、前端服务定义。

    Returns:
        固定顺序为 ``[backend, frontend]`` 的两个 Service；后续端口检查依赖
        这个顺序。

    Raises:
        FileNotFoundError: 前端 package.json、虚拟环境或前端依赖缺失。
    """

    package_json = UI_ROOT / "package.json"
    if not package_json.is_file():
        raise FileNotFoundError(f"未找到前端工程：{package_json}")

    return [
        Service(
            name="backend",
            command=[
                str(backend_python()),
                "-m",
                "uvicorn",
                "agent.main:app",
                "--host",
                BACKEND_HOST,
                "--port",
                str(BACKEND_PORT),
            ],
            cwd=PROJECT_ROOT,
        ),
        Service(
            name="frontend",
            command=frontend_command(),
            cwd=UI_ROOT,
        ),
    ]


def start_service(service: Service) -> None:
    """启动一个子服务并保存其 Popen 对象。

    Args:
        service: 已包含命令和工作目录、尚未启动的 Service。

    子进程不经过 ``shell=True``，降低转义问题并确保参数边界明确。Windows 下
    创建新进程组，使后续能够发送 CTRL_BREAK_EVENT 进行优雅关闭。
    """

    creationflags = 0
    if os.name == "nt":
        # 独立进程组使 Ctrl+Break 能传递给对应服务及其子进程。
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    # 子进程继承用户环境，只补充 UTF-8 默认值，不覆盖用户显式配置。
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("PYTHONUTF8", "1")
    service.process = subprocess.Popen(
        service.command,
        cwd=service.cwd,
        env=environment,
        creationflags=creationflags,
    )
    print(f"[{service.name}] 已启动，PID={service.process.pid}", flush=True)


def wait_for_port(service: Service, host: str, port: int) -> None:
    """在限定时间内等待服务端口开始监听。

    Args:
        service: 正在启动、需要同时监控退出状态的服务。
        host: 探测地址。
        port: 探测端口。

    Raises:
        RuntimeError: 子进程在端口就绪前已经退出。
        TimeoutError: 超过 ``STARTUP_TIMEOUT_SECONDS`` 仍无法连接端口。

    每次连接超时只有 0.4 秒，并在失败后短暂轮询，既能快速发现服务就绪，也
    不会用长时间阻塞掩盖子进程提前退出。
    """

    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if service.process and service.process.poll() is not None:
            raise RuntimeError(
                f"{service.name} 启动失败，退出码={service.process.returncode}"
            )
        try:
            # 短连接只用于就绪探测，不会占用应用的长连接资源。
            with socket.create_connection((host, port), timeout=0.4):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"等待 {service.name} 端口 {host}:{port} 超时")


def stop_service(service: Service) -> None:
    """优先优雅停止服务，超时后再强制清理。

    Args:
        service: 可能尚未启动、已经退出或仍在运行的服务。

    未启动和已退出服务直接返回。正常路径给进程最多五秒释放端口和连接；超时
    后 Windows 使用 ``taskkill /T /F`` 清理整个子进程树，其他平台调用 kill。
    该强制路径只针对本脚本明确创建并保存 PID 的进程。
    """

    process = service.process
    if process is None or process.poll() is not None:
        return

    print(f"[{service.name}] 正在停止……", flush=True)
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass

    # 优雅关闭失败时，Windows 使用 taskkill 同时清理 Vite 等后代进程。
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"[{service.name}] 进程未能在限定时间内退出。", file=sys.stderr)


def monitor(services: list[Service]) -> int:
    """持续监控服务列表，任一退出时结束托管。

    Returns:
        非零子进程退出码；如果服务意外以 0 退出，也返回 1，表示整个一键启动
        会话未能继续保持两个服务同时运行。

    正常情况下循环直到用户按 Ctrl+C。函数每 0.3 秒轮询一次，避免忙等占用 CPU。
    """

    while True:
        for service in services:
            process = service.process
            if process and process.poll() is not None:
                code = process.returncode or 0
                print(f"[{service.name}] 已退出，退出码={code}", flush=True)
                return code if code != 0 else 1
        time.sleep(0.3)


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回 argparse 命名空间。

    当前仅支持 ``--smoke-test``，后续如果增加端口或 Host 参数，应同步调整
    Service 构建和前端 WebSocket 地址策略。
    """

    parser = argparse.ArgumentParser(description="同时启动 Voice Agent 前后端服务")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="确认两个端口启动成功后自动停止服务",
    )
    return parser.parse_args()


def main() -> int:
    """启动、检查并托管两个服务。

    Returns:
        ``0`` 表示用户正常中断或冒烟检查通过，``1`` 表示启动/运行失败，其他
        非零值可能来自提前退出的子服务。

    ``finally`` 是整个脚本的资源保障：无论依赖检查、端口等待、监控还是用户
    中断在哪一步结束，已经启动的服务都会按相反顺序尝试停止。
    """

    # Windows 控制台默认编码可能不是 UTF-8，显式设置可避免中文日志乱码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    services: list[Service] = []
    try:
        services = build_services()
        for service in services:
            start_service(service)

        # 只有两个端口均可连接后，才向用户报告整个项目启动成功。
        wait_for_port(services[0], BACKEND_HOST, BACKEND_PORT)
        wait_for_port(services[1], FRONTEND_HOST, FRONTEND_PORT)
        print("\nVoice Agent 已启动：", flush=True)
        print(f"  前端：http://{FRONTEND_HOST}:{FRONTEND_PORT}", flush=True)
        print(f"  后端：http://{BACKEND_HOST}:{BACKEND_PORT}", flush=True)

        if args.smoke_test:
            print("自检通过，准备停止两个服务。", flush=True)
            return 0

        print("按 Ctrl+C 同时停止两个服务。\n", flush=True)
        return monitor(services)
    except KeyboardInterrupt:
        print("\n收到停止指令。", flush=True)
        return 0
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        # reversed 让后启动的前端先停止，再关闭后端。
        for service in reversed(services):
            stop_service(service)


if __name__ == "__main__":
    raise SystemExit(main())
