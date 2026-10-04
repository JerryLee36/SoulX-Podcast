import unittest

import numpy as np

from soulxpodcast.utils.segmented_workflow import (
    assemble_audio_and_manifest,
    normalize_sentence_table_rows,
    replace_segment_audio,
    split_script_into_sentence_units,
    table_to_units,
)


class SegmentedWorkflowTests(unittest.TestCase):
    def test_normalize_rows_from_gradio_payload_dict(self):
        payload = {
            "headers": ["sentence_id", "chunk_id", "speaker", "text", "estimate_seconds"],
            "data": [
                [0, 0, "S1", "Hello world.", 1.2],
                [None, None, None, None, None],
            ],
        }

        rows = normalize_sentence_table_rows(payload)
        units = table_to_units(rows)

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].text, "Hello world.")
        self.assertEqual(units[0].speaker, "S1")

    def test_normalize_rows_from_array_with_header_row(self):
        payload = [
            ["sentence_id", "chunk_id", "speaker", "text", "estimate_seconds"],
            [0, 0, "S1", "First.", 1.0],
            [1, 0, "S2", "Second.", 1.1],
        ]

        rows = normalize_sentence_table_rows(payload)
        units = table_to_units(rows)

        self.assertEqual([u.text for u in units], ["First.", "Second."])
        self.assertEqual([u.speaker for u in units], ["S1", "S2"])

    def test_normalize_rows_rejects_partially_populated_row(self):
        payload = [[0, 0, "S1", None, 1.0]]
        with self.assertRaisesRegex(ValueError, "Partially populated row 0: missing text"):
            normalize_sentence_table_rows(payload)

    def test_normalize_rows_supports_dataframe_like_records(self):
        class FakeDataframe:
            def to_dict(self, orient):
                if orient != "records":
                    raise ValueError("unexpected orient")
                return [
                    {"sentence_id": 0, "chunk_id": 0, "speaker": "S1", "text": "A.", "estimate_seconds": 1.0},
                    {"sentence_id": 1, "chunk_id": 0, "speaker": "S2", "text": "B.", "estimate_seconds": 1.0},
                ]

        rows = normalize_sentence_table_rows(FakeDataframe())
        units = table_to_units(rows)
        self.assertEqual([u.text for u in units], ["A.", "B."])

    def test_sentence_split_preserves_order_and_speaker(self):
        script = "[S1]Dr. Lee arrived at 3.14. Are you ready? Yes! [S2]Great."  # noqa: E501
        units = split_script_into_sentence_units(script, max_chunk_seconds=120.0)

        self.assertEqual([u.speaker for u in units], ["S1", "S1", "S1", "S2"])
        self.assertEqual(
            [u.text for u in units],
            [
                "Dr. Lee arrived at 3.14.",
                "Are you ready?",
                "Yes!",
                "Great.",
            ],
        )

    def test_chunking_respects_limit(self):
        script = "[S1]One short sentence. Another short sentence. Third short sentence."
        units = split_script_into_sentence_units(script, max_chunk_seconds=2.0)

        # Small limit should force multiple chunks while preserving sentence order
        self.assertEqual([u.sentence_id for u in units], [0, 1, 2])
        self.assertGreaterEqual(len({u.chunk_id for u in units}), 2)

    def test_replace_and_reassemble_updates_timing(self):
        segments = [
            {"sentence_id": 0, "chunk_id": 0, "speaker": "S1", "text": "A.", "audio": np.ones(24000, dtype=np.float32)},
            {"sentence_id": 1, "chunk_id": 0, "speaker": "S1", "text": "B.", "audio": np.ones(12000, dtype=np.float32)},
            {"sentence_id": 2, "chunk_id": 1, "speaker": "S2", "text": "C.", "audio": np.ones(6000, dtype=np.float32)},
        ]

        _, manifest_before = assemble_audio_and_manifest(segments, sample_rate=24000)
        self.assertAlmostEqual(manifest_before["segments"][1]["start_seconds"], 1.0)

        updated = replace_segment_audio(segments, sentence_id=1, new_audio=np.ones(24000, dtype=np.float32))
        _, manifest_after = assemble_audio_and_manifest(updated, sample_rate=24000)

        self.assertAlmostEqual(manifest_after["segments"][1]["duration_seconds"], 1.0)
        self.assertAlmostEqual(manifest_after["segments"][2]["start_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
