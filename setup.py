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
    python_requires="<3.9",
    install_requires=[
        'torch==1.11.0',
        'torchvision==0.12.0',
        'torchaudio==0.11.0',
        'torch-geometric==2.1.0.post1',
        'pytorch3d==0.7.2',
        'e3nn==0.4.4',
        'pyyaml',        # 6.0
        'tqdm',          # 4.64.1
        'jupyter',       # 1.0.0
        'pandas',
        'plotly',
    ]
)