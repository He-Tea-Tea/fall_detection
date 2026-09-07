"""v0.96 launcher.

The production entry point lives in ``fall_detection.app.main``.  Keeping
this small launcher at the release root preserves the familiar command::

    python main.py --self-test
"""

from fall_detection.app.main import main


if __name__ == "__main__":
    main()
