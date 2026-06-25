import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.version import Version

from build_tools.compiler_version import (
    get_cuda_bare_metal_version,
    parse_cuda_release_version,
)


class CompilerVersionTest(unittest.TestCase):
    def test_parse_nvcc_release_version(self):
        output = "Cuda compilation tools, release 12.1, V12.1.105"

        self.assertEqual(parse_cuda_release_version(output), Version("12.1"))

    def test_parse_cucc_version_without_release_token(self):
        output = "cucc compiler driver V12.2.91"

        self.assertEqual(parse_cuda_release_version(output), Version("12.2"))

    def test_get_cuda_bare_metal_version_prefers_cucc(self):
        with TemporaryDirectory() as tmp_dir:
            bin_dir = Path(tmp_dir) / "bin"
            bin_dir.mkdir()
            cucc = bin_dir / "cucc"
            nvcc = bin_dir / "nvcc"
            cucc.write_text("#!/bin/sh\n", encoding="utf-8")
            nvcc.write_text("#!/bin/sh\n", encoding="utf-8")

            commands: list[list[str]] = []

            def fake_check_output(command, **_kwargs):
                commands.append(command)
                return "cucc compiler driver V12.3.0"

            raw_output, version = get_cuda_bare_metal_version(
                tmp_dir,
                check_output=fake_check_output,
            )

            self.assertEqual(commands, [[str(cucc), "-V"]])
            self.assertIn("cucc", raw_output)
            self.assertEqual(version, Version("12.3"))


if __name__ == "__main__":
    unittest.main()
