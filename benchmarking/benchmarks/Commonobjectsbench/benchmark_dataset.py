"""CommonObjectsBench benchmark dataset implementation."""
from imsearch_eval.adapters.huggingface import HuggingFaceDataset


class CommonObjectsBench(HuggingFaceDataset):
    """Benchmark dataset class for CommonObjectsBench (general object image retrieval)."""

    def get_query_column(self) -> str:
        """Get the name of the column containing the query text."""
        return "query_text"

    def get_query_id_column(self) -> str:
        """Get the name of the column containing the query ID."""
        return "query_id"

    def get_relevance_column(self) -> str:
        """Get the name of the column containing relevance labels (0=not relevant, 1=relevant)."""
        return "relevance_label"

    def get_metadata_columns(self) -> list:
        """Get optional metadata columns to include in evaluation stats."""
        return [
            "viewpoint",
            "lighting",
            "environment_type",
            "urban_scene",
            "rural_scene",
            "outdoor_scene",
            "person_present",
            "animal_present",
            "food_present",
            "vehicle_present",
            "multiple_objects",
            "artificial_lighting",
            "occlusion_present",
            "text_visible",
        ]