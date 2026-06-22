import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_tools.maca_env import (
    format_maca_build_env_errors,
    resolve_maca_build_env,
    validate_maca_build_env,
)


class MacaEnvTest(unittest.TestCase):
    def test_resolve_maca_build_env_defaults_to_cu_bridge(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            maca_path = tmp_path / "maca"
            cuda_path = maca_path / "tools" / "cu-bridge"
            clang_path = maca_path / "mxgpu_llvm" / "bin"
            for path in (cuda_path / "bin", clang_path, maca_path / "lib"):
                path.mkdir(parents=True)
            cucc = cuda_path / "bin" / "cucc"
            cucc.write_text("#!/bin/sh\n", encoding="utf-8")

            build_env = resolve_maca_build_env({"MACA_PATH": str(maca_path)}, which=lambda _: None)

            self.assertEqual(build_env.maca_path, maca_path)
            self.assertEqual(build_env.cuda_path, cuda_path)
            self.assertEqual(build_env.maca_clang_path, clang_path)
            self.assertEqual(build_env.maca_lib_path, maca_path / "lib")
            self.assertEqual(build_env.cucc_path, cucc)
            self.assertEqual(validate_maca_build_env(build_env), [])

    def test_validate_maca_build_env_reports_missing_paths(self):
        build_env = resolve_maca_build_env({}, which=lambda _: None)

        errors = validate_maca_build_env(build_env)

        self.assertIn("MACA_PATH is not configured", errors)
        self.assertTrue(any("cucc compiler was not found" in error for error in errors))
        self.assertIn("Invalid MACA build environment", format_maca_build_env_errors(errors))

    def test_resolve_maca_build_env_uses_path_cucc(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            maca_path = tmp_path / "maca"
            cuda_path = tmp_path / "cu-bridge"
            clang_path = tmp_path / "clang"
            path_cucc = tmp_path / "bin" / "cucc"
            for path in (maca_path / "lib", cuda_path, clang_path, path_cucc.parent):
                path.mkdir(parents=True)
            path_cucc.write_text("#!/bin/sh\n", encoding="utf-8")

            build_env = resolve_maca_build_env(
                {
                    "MACA_PATH": str(maca_path),
                    "CUDA_PATH": str(cuda_path),
                    "MACA_CLANG_PATH": str(clang_path),
                },
                which=lambda _: str(path_cucc),
            )

            self.assertEqual(build_env.cucc_path, path_cucc)
            self.assertEqual(validate_maca_build_env(build_env), [])

    def test_resolve_maca_build_env_strips_whitespace(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            maca_path = tmp_path / "maca"
            for path in (maca_path / "tools" / "cu-bridge" / "bin", maca_path / "mxgpu_llvm" / "bin", maca_path / "lib"):
                path.mkdir(parents=True)
            cucc = maca_path / "tools" / "cu-bridge" / "bin" / "cucc"
            cucc.write_text("#!/bin/sh\n", encoding="utf-8")

            build_env = resolve_maca_build_env(
                {"MACA_PATH": f"  {maca_path}  "},
                which=lambda _: None,
            )

            self.assertEqual(build_env.maca_path, maca_path)
            self.assertEqual(build_env.cucc_path, cucc)


if __name__ == "__main__":
    unittest.main()
