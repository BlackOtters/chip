#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_FILE = "submodule_cfg.yaml"


def run(cmd, cwd=None):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")


def load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def install_submodules(selected=None, index_url=None):
    config = load_config()
    for name, info in config.items():
        print(f"\n----- Installing submodule: {name} -----")
        install = info.get("install", False)
        if (selected is None and not install) or (selected is not None and name not in selected):
            print(f"Skipping submodule: {name}")
            continue

        path = Path(info["path"])
        patches = info.get("patches", [])
        addons = info.get("addons", [])

        print(f"Initializing submodule '{name}'...")
        # If submodule already initialized, clean it first
        if (path / ".git").exists():
            print(f"Cleaning existing submodule '{name}'...")
            run(f"cd {path} && git reset --hard && git clean -fd")
        run(f"git submodule update --init {path}")

        if not path.exists():
            print(f"Path {path} does not exist. Skipping {name}.")
            continue

        for patch in patches:
            patch_path = path / patch
            if patch_path.exists():
                print(f"Applying patch {patch} for '{name}'...")
                run(f"cd {path} && git apply {patch}")
            else:
                print(f"Patch {patch} not found for '{name}', skipping.")

        for addon in addons:
            addon_path = path / addon
            if addon_path.exists():
                print(f"Adding addon '{addon_path}' to '{name}'...")
                shutil.copytree(addon_path, path, dirs_exist_ok=True)
            else:
                print(f"Addon path {addon} does not exist, skipping.")

        # install Python package
        packages = [path]  # main package
        if (extra_packages := info.get("extra_packages", None)) is not None:
            packages += extra_packages
        for pkg in packages:
            pip_cmd = f"pip install -e {pkg}"
            if index_url:
                pip_cmd += f" -i {index_url}"
            run(pip_cmd)


def parse_args():
    parser = argparse.ArgumentParser(description="Install optional RoboJuDo submodules and Python packages.")
    parser.add_argument(
        "modules",
        nargs="*",
        help="Optional list of module names (e.g. mujoco_viewer unitree_cpp). If omitted, uses install=true in submodule_cfg.yaml.",
    )
    parser.add_argument(
        "-i",
        "--index-url",
        type=str,
        default=None,
        help="Custom pip index URL (e.g. https://pypi.tuna.tsinghua.edu.cn/simple).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_modules = args.modules if len(args.modules) > 0 else None
    # selected_modules will override install cfg if provided
    install_submodules(selected_modules, index_url=args.index_url)

    # Usage example:
    # python submodule_install.py mujoco_viewer unitree_cpp
    # python submodule_install.py -i https://pypi.tuna.tsinghua.edu.cn/simple
