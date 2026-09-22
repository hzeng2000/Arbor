import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.regenerate_train_data import load_completed_ids

from tests.utils import (
    execute_shell_command,
    get_available_port,
    terminate_process_trees,
    wait_for_server,
)

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")


class TestRegenerateTrainData(unittest.TestCase):
    def test_resume_loads_completed_ids_across_output_files(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output.jsonl"
            error = root / "output_error.jsonl"
            skipped = root / "output_skipped.jsonl"
            output.write_text(json.dumps({"id": "success"}) + "\n")
            error.write_text(json.dumps({"id": "error"}) + "\n")

            completed_ids, counts = load_completed_ids(
                [str(output), str(error), str(skipped)]
            )

            self.assertEqual({"success": 1, "error": 1}, dict(completed_ids))
            self.assertEqual(1, counts[str(output)])
            self.assertEqual(1, counts[str(error)])
            self.assertEqual(0, counts[str(skipped)])

    def test_regenerate_sharegpt(self):
        port = get_available_port()
        data_process = execute_shell_command(
            "python scripts/prepare_data.py --dataset sharegpt"
        )
        data_process.wait()

        sglang_process = execute_shell_command(
            f"""python3 -m sglang.launch_server \
    --model unsloth/Llama-3.2-1B-Instruct \
    --tp 1 \
    --cuda-graph-bs 4 \
    --dtype bfloat16 \
    --mem-frac=0.8 \
    --port {port}
        """,
            disable_proxy=True,
            enable_hf_mirror=False,
            sglang_use_modelscope=True,
            start_new_session=True,
        )
        try:
            wait_for_server(
                f"http://localhost:{port}",
                timeout=300,
                disable_proxy=True,
                process=sglang_process,
            )
            regeneration_process = execute_shell_command(
                f"""python scripts/regenerate_train_data.py \
    --model unsloth/Llama-3.2-1B-Instruct \
    --concurrency 128 \
    --max-tokens 128 \
    --server-address localhost:{port} \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl \
    --num-samples 10
        """,
                disable_proxy=True,
                enable_hf_mirror=False,
            )
            regeneration_process.wait()
            self.assertEqual(regeneration_process.returncode, 0)
            self.assertTrue(
                CACHE_DIR.joinpath("dataset", "sharegpt_train_regen.jsonl").exists()
            )
        finally:
            terminate_process_trees(sglang_process, grace_s=30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
