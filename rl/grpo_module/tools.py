from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
from typing import Dict, Optional

import ray

from open_instruct.tool_utils.tool_actor import ToolActor, TOOL_CLASS_REGISTRY
from open_instruct.tool_utils.tool_proxy import ToolProxy


def launch_mcp_subprocess(run_mcp_command: str, output_dir: str) -> Optional[subprocess.Popen]:
    print(f"Launching MCP server subprocess: {run_mcp_command}")
    mcp_logs_dir = os.path.join(output_dir, "mcp_logs")
    os.makedirs(mcp_logs_dir, exist_ok=True)
    mcp_stdout = open(os.path.join(mcp_logs_dir, "mcp_server_stdout.log"), "w")
    mcp_stderr = open(os.path.join(mcp_logs_dir, "mcp_server_stderr.log"), "w")
    mcp_process = subprocess.Popen(
        [run_mcp_command],
        shell=True,
        stdout=mcp_stdout,
        stderr=mcp_stderr,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    try:
        import re
        import requests

        port_match = re.search(r"--port\s+(\d+)", run_mcp_command)
        port = int(port_match.group(1)) if port_match else 8003
        start_time = time.time()
        max_wait_time = 600
        server_ready = False
        while time.time() - start_time < max_wait_time:
            if mcp_process.poll() is not None:
                break
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    server_ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        if mcp_process.poll() is None and server_ready:
            def cleanup_mcp() -> None:
                if mcp_process.poll() is None:
                    try:
                        os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                        time.sleep(2)
                        if mcp_process.poll() is None:
                            os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                mcp_stdout.close()
                mcp_stderr.close()
            atexit.register(cleanup_mcp)
            return mcp_process
    except Exception as exc:
        print(f"Error launching MCP server: {exc}")
    mcp_stdout.close()
    mcp_stderr.close()
    return None


def register_tools(args) -> Dict[str, ToolProxy]:
    tool_objects: Dict[str, ToolProxy] = {}
    if not args.tools:
        return tool_objects

    def _register_actor_backed_tool(class_path: str, init_kwargs: dict) -> None:
        actor = ToolActor.options(max_concurrency=args.tool_max_concurrency).remote(
            class_path=class_path,
            init_kwargs=init_kwargs,
        )
        start = ray.get(actor.get_start_str.remote())
        stop_strings = ray.get(actor.get_stop_strings.remote())
        for end_str in stop_strings:
            tool_objects[end_str] = ToolProxy(actor_handle=actor, start_str=start, end_str=end_str)

    for tool in args.tools:
        class_path = TOOL_CLASS_REGISTRY.get(tool.lower(), None)
        if class_path is None:
            raise ValueError(f"Unknown tool: {tool}")
        _register_actor_backed_tool(class_path=class_path, init_kwargs=vars(args))
    return tool_objects
