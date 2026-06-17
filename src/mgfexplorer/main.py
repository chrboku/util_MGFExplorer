import sys
from importlib.resources import files
from PyQt6.QtWidgets import QApplication
from .app import MGFExplorerApp


def _load_stylesheet() -> str:
    """Load the bundled Qt stylesheet from style.css."""
    try:
        return files(__package__).joinpath("style.css").read_text(encoding="utf-8")
    except Exception:
        return ""


def main():
    app = QApplication(sys.argv)
    stylesheet = _load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)
    window = MGFExplorerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
