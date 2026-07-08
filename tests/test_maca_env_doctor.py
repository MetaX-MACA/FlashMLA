import importlib.util
import unittest
from pathlib import Path


DOCTOR_PATH = Path(__file__).resolve().parents[1] / "tools" / "maca_env_doctor.py"
spec = importlib.util.spec_from_file_location("maca_env_doctor", DOCTOR_PATH)
maca_env_doctor = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(maca_env_doctor)


class MacaEnvDoctorTest(unittest.TestCase):
    def test_collect_report_marks_missing_environment(self):
        report = maca_env_doctor.collect_report({})

        self.assertFalse(report["ok"])
        self.assertIn("environment", report)
        self.assertTrue(any(item["name"] == "MACA_PATH" for item in report["checks"]))


if __name__ == "__main__":
    unittest.main()
