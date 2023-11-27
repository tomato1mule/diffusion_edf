from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="diffusion_edf",
    version="0.0.1",
    author="Hyunwoo Ryu",
    author_email="tomato1mule@gmail.com",
    description="Diffusion EDF.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomato1mule/diffusion_edf",
    project_urls={
        "Bug Tracker": "https://github.com/tomato1mule/diffusion_edf/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 22.04",
    ],
    packages=['diffusion_edf'],
    python_requires="<3.9",
    install_requires=[
        'torch',
        'torchvision',
        'e3nn==0.4.4',
        # 'open3d==0.16.0',
        'xitorch==0.5.1',
        'pyyaml',        # 6.0
        'tqdm',          # 4.64.1
        'jupyter',       # 1.0.0
        'pandas',
        'plotly',
        'dash',
        'dash_vtk',
        'dash_daq',
        'beartype',
        'tensorboard',
        # 'einops==0.7.0'
    ]
)