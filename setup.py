from setuptools import setup, find_packages

long_description = """
A robust tool for advanced AI text generation.
"""


setup(
    name="aitextgen",
    packages=["aitextgen"],  # this must be the same as the name above
    version="0.1",
    description="A robust tool for advanced AI text generation using Transformers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Max Woolf",
    author_email="max@minimaxir.com",
    url="https://github.com/minimaxir/aitextgen",
    keywords=["wordcloud", "data visualization", "text cool stuff"],
    classifiers=[],
    license="MIT",
    entry_points={"console_scripts": ["aitextgen=aitextgen.cli:aitextgen_cli"]},
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=[
        "transformers>=2.9.0",
        "fire",
        "msgpack",
        "pytorch-lightning>=0.7.5",
    ],
)
