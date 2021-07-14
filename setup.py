import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PYTHON-ML",
    version="0.0.1",
    author="shyang",
    author_email="disroway@chungbuk.ac.kr",
    description="머신러닝교과서 with 파이썬을 공부하기 위한 프로젝트",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shyang1012/python-ML",
    project_urls={
        "Bug Tracker": "https://github.com/shyang1012/python-ML/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)