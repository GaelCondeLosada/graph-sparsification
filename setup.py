import os
import sys
import shutil
from setuptools import setup, find_packages

# ── Cross-platform C++ extension build ────────────────────────────────
# pybind11's auto_cpp_level probes versioned compilers (g++-12, g++-11, …)
# which may not exist even when g++ does. We detect this and set CXX
# automatically so that `pip install -e .` works on any platform without
# the user needing to know about CXX.

def _ensure_cxx_env():
    """Set CXX if not already set and the default g++ is available."""
    if "CXX" in os.environ:
        return
    if sys.platform == "win32":
        return  # MSVC is found automatically on Windows
    # On Linux/macOS, check if g++ or c++ is available
    for candidate in ["g++", "c++", "clang++"]:
        if shutil.which(candidate):
            os.environ["CXX"] = candidate
            return

_ensure_cxx_env()

ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    compile_args = []
    link_args = []
    if sys.platform == "win32":
        compile_args = ["/O2", "/std:c++17"]
    elif sys.platform == "darwin":
        compile_args = ["-O3", "-std=c++17", "-stdlib=libc++"]
        link_args = ["-stdlib=libc++"]
    else:
        compile_args = ["-O3", "-std=c++17"]

    ext_modules = [
        Pybind11Extension(
            "graph_sparsification._sir_cpp",
            ["src/cpp/sir.cpp"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    print("WARNING: pybind11 not found — C++ extension will not be built.\n"
          "  The package will fall back to a pure-Python SIR implementation.\n"
          "  To enable the fast C++ backend: pip install pybind11")

setup(
    name="graph_sparsification",
    version="0.1.0",
    description="Graph sparsification research: MBB, EffR, SIR simulations",
    package_dir={"": "src/python"},
    packages=find_packages(where="src/python"),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "networkx>=3.0",
        "pybind11>=2.11",
    ],
)
