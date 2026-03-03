# Configuration file for the Sphinx documentation builder.

import sys
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------
with open(ROOT / "pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

project = pyproject["project"]["name"]
copyright = f"{datetime.now().year}, MWS AI"
author = "MWS AI Team"
version = pyproject["project"]["version"]
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
    "sphinx.ext.viewcode",  # (5) Source code links
]

# -- AutoAPI configuration ---------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../open_autonlu"]
autoapi_root = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "inherited-members",  # Filtered conditionally in skip_member
]
autoapi_ignore = [
    "*/__pycache__/*",
    "*/.env*",
    # Excluded modules
    "*/auto_classes/constants.py",
    "*/data_quality/data_types.py",
    "*/data_quality/utils.py",
    "*/data_types.py",
    "*/llm_pipelines/processor.py",
    "*/methods/utils/*",
    "*/progress_reporter.py",
]
autoapi_python_class_content = "both"

# Classes that should show inherited members
SHOW_INHERITED_FOR = [
    "open_autonlu.auto_classes",
    "open_autonlu.methods.finetuner_ood",
    "open_autonlu.methods.set_fit_ood",
    "open_autonlu.methods.anc_set_fit_ood",
]

# Attributes to always skip (like logger instances)
SKIP_ATTRIBUTES = ["log", "logger", "logging"]


# (2) Hide private members, (3) Hide abstract methods
def skip_member(app, what, name, obj, skip, options):
    # Get short name (last part after dot)
    short_name = name.split(".")[-1]
    # Skip private members (but keep __init__ and other dunder if needed)
    if short_name.startswith("_") and not short_name.startswith("__"):
        return True
    # Skip logger attributes
    if short_name in SKIP_ATTRIBUTES:
        return True
    # Skip abstract methods
    if (
        hasattr(obj, "properties")
        and obj.properties
        and "abstractmethod" in obj.properties
    ):
        return True
    # Skip inherited members unless in allowed modules
    if hasattr(obj, "inherited") and obj.inherited:
        # Check if this belongs to a module that should show inherited
        should_show = any(name.startswith(prefix) for prefix in SHOW_INHERITED_FOR)
        if not should_show:
            return True
    # Skip attributes without docstrings (empty attributes)
    if what == "attribute" and hasattr(obj, "docstring") and not obj.docstring:
        return True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", skip_member)


# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True

templates_path = ["_templates"]
exclude_patterns = []
language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
