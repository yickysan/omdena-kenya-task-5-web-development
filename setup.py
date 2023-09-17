import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"
REPO_NAME = "omdena-kenya-task-5-web-development"
ORGANIZATION = "OmdenaAI"
AUTHOR_NAME = "Abomaye Victor"
SRC_NAME = "OmdenaKenyaRoadAccidents"


setuptools.setup(
    name = SRC_NAME,
    version = __version__,
    author = AUTHOR_NAME,
    description = "A python package for predicting road accidents severity in Kenya",
    long_description = long_description,
    long_description_content = "text/markdown",
    url = f"https://www.github.com/yickysan/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://www.github.com/yickysan/{REPO_NAME}/issues"
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where = "src")
    )
