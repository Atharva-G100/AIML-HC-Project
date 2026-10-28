"""Locations shared by the application, independent of the launch directory."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ENV_FILE = PROJECT_ROOT / '.env'
ASSETS_DIR = PACKAGE_DIR / 'assets'
