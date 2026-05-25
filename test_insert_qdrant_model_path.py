import os
import tempfile
import unittest

from core import config


class SelectModelPathTest(unittest.TestCase):
    def test_selects_newest_existing_boost_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, "boost_best_vlm_v5"))
            os.mkdir(os.path.join(tmpdir, "boost_best_vlm_v7"))

            self.assertEqual(
                config.select_model_path(tmpdir),
                os.path.join(tmpdir, "boost_best_vlm_v7"),
            )


if __name__ == "__main__":
    unittest.main()
