import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "HippoTrainer"
copyright = "2025, Intelligent Systems"
author = "Daniil Dorin, Igor Ignashin, Nikita Kiselev, Andrey Veprikov"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "myst_parser",
]

html_theme = "furo"
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
