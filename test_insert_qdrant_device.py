import os
import unittest

from insert_qdrant import select_model_device


class SelectModelDeviceTest(unittest.TestCase):
    def test_defaults_to_cpu_for_update_stability(self):
        old_value = os.environ.pop("INNOFLOW_EMBED_DEVICE", None)
        try:
            self.assertEqual(select_model_device(), "cpu")
        finally:
            if old_value is not None:
                os.environ["INNOFLOW_EMBED_DEVICE"] = old_value

    def test_env_can_override_device(self):
        old_value = os.environ.get("INNOFLOW_EMBED_DEVICE")
        os.environ["INNOFLOW_EMBED_DEVICE"] = "cuda:0"
        try:
            self.assertEqual(select_model_device(), "cuda:0")
        finally:
            if old_value is None:
                os.environ.pop("INNOFLOW_EMBED_DEVICE", None)
            else:
                os.environ["INNOFLOW_EMBED_DEVICE"] = old_value


if __name__ == "__main__":
    unittest.main()
