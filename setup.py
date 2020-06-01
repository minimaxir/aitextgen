from setuptools import setup

setup(
    name="aitextgen",
    packages=["aitextgen"],  # this must be the same as the name above
    version="0.1.1",
    description="A robust Python tool for text-based AI training and generation using GPT-2.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Max Woolf",
    author_email="max@minimaxir.com",
    url="https://github.com/minimaxir/aitextgen",
    keywords=["gpt-2", "gpt2", "text generation", "ai"],
    classifiers=[],
    license="MIT",
    entry_points={"console_scripts": ["aitextgen=aitextgen.cli:aitextgen_cli"]},
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=[
        "transformers>=2.9.1",
        "fire>=0.3.0",
        "pytorch-lightning>=0.7.6",
    ],
)
