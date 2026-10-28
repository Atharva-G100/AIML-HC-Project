"""Streamlit entry point. Run from the repository root with streamlit run app.py."""
from labnote.scanner import MedicalReportScanner
from labnote.ui import render_app


def main():
    render_app(MedicalReportScanner)


if __name__ == '__main__':
    main()
