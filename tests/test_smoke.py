"""Small package-level smoke test.

The release currently keeps detailed synthetic self-tests next to each
algorithm module.  This test provides one standard unittest/pytest entry
point while those tests are migrated into this directory incrementally.
"""

import unittest
from pathlib import Path

from fall_detection.app.main import load_config, run_self_test


class CompletePipelineSmokeTest(unittest.TestCase):
    def test_complete_pipeline_self_test(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(str(project_root / "config.yaml"))
        run_self_test(config)


if __name__ == "__main__":
    unittest.main()
