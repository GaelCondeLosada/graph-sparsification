from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

ext_modules = [
    Pybind11Extension(
        "graph_sparsification._sir_cpp",
        ["src/cpp/sir.cpp"],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="graph_sparsification",
    version="0.1.0",
    description="Graph sparsification research: MBB, EffR, SIR simulations",
    package_dir={"": "src/python"},
    packages=find_packages(where="src/python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "networkx>=3.0",
    ],
)
