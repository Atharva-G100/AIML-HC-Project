"""Exit successfully only when the launcher can import its dependencies."""
import importlib
import sys

for module in ('streamlit', 'pandas', 'plotly', 'google.genai', 'dotenv', 'PIL', 'PyPDF2', 'pdfplumber'):
    try:
        importlib.import_module(module)
    except ImportError:
        print(f'Missing dependency: {module}')
        sys.exit(1)
