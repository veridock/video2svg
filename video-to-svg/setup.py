from setuptools import setup, find_packages

setup(
    name="video2svg",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "opencv-python",
        "numpy",
        "svgwrite",
        "PyYAML"
    ],
    entry_points={
        "console_scripts": [
            "video2svg=video2svg.cli:main",
        ],
    },
    python_requires=">=3.6",
    description="Convert videos to SVG animations",
    author="",
    author_email="",
    url="",
)
