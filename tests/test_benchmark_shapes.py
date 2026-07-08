import ast
import unittest
from pathlib import Path


BENCHMARK = Path(__file__).parents[1] / "benchmark" / "bench_flash_mla.py"


class BenchmarkShapeConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(BENCHMARK.read_text(encoding="utf-8"))

    def test_shape_configs_are_not_created_at_import_time(self):
        top_level_assigns = [
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "shape_configs"
                for target in node.targets
            )
        ]

        self.assertEqual(top_level_assigns, [])

    def test_shape_builder_accepts_device_argument(self):
        builder = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_shape_configs"
        )

        self.assertEqual(builder.args.args[0].arg, "device")
        self.assertEqual(ast.literal_eval(builder.args.defaults[0]), "cuda")

    def test_main_creates_shape_configs_lazily(self):
        main_block = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
        calls = [node for node in ast.walk(main_block) if isinstance(node, ast.Call)]

        self.assertTrue(
            any(
                isinstance(call.func, ast.Name) and call.func.id == "build_shape_configs"
                for call in calls
            )
        )


if __name__ == "__main__":
    unittest.main()
