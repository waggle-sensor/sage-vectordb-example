"""CloudBench benchmark dataset implementation."""
from imsearch_eval.adapters.huggingface import HuggingFaceDataset


class CloudBench(HuggingFaceDataset):
    """Benchmark dataset class for CloudBench (cloud/atmospheric image retrieval)."""

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
            "cloud_coverage",
            "viewpoint",
            "lighting",
            "confounder_type",
            "occlusion_present",
            "multiple_cloud_types",
            "horizon_visible",
            "ground_visible",
            "sun_visible",
            "precipitation_visible",
            "overcast",
            "multiple_layers",
            "storm_visible",
        ]
