import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "HippoTrainer"
copyright = "2025, Intelligent Systems"
author = "Daniil Dorin, Igor Ignashin, Nikita Kiselev, Andrey Veprikov"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
