import pandas as pd
from .embedding_service import EmbeddingService
from .config import CONFIG
import json
import os
import numpy as np

class DataLoader:
    def __init__(self, csv_path: str = CONFIG.KNOWLEDGE_BASE_PATH, output_json_path: str = CONFIG.KNOWLEDGE_BASE_EMBEDDINGS_PATH) -> None:
        self.csv_path = csv_path
        self.output_json_path = output_json_path
        self.embedding_service = EmbeddingService()

    def load_and_generate_embeddings(self) -> pd.DataFrame:
        """
        Loads the knowledge base. If the JSON file with embeddings exists, it loads that.
        Otherwise, it loads the original CSV, generates embeddings for missing values in the "Vector" column,
        saves the new data to JSON, and returns the DataFrame.
        """
        if os.path.exists(self.output_json_path):
            print(f"Loading existing file: {self.output_json_path}")
            df = pd.read_json(self.output_json_path)

            def fix_vector_format(val):
                """Ensure each value in 'Vector' is valid JSON."""
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, list) and all(isinstance(x, (float, int)) for x in parsed):
                            return parsed
                        print(f"Warning: Skipping corrupted JSON in 'Vector' column: {val[:100]}...")
                        return None
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping corrupted JSON in 'Vector' column: {val[:100]}...")
                        return None
                elif isinstance(val, list) or isinstance(val, np.ndarray):
                    return list(val)
                return None

            df["Vector"] = df["Vector"].apply(fix_vector_format)

            missing_embeddings = df["Vector"].isna() | df["Vector"].apply(lambda x: x is None or len(x) == 0)

            if missing_embeddings.any():
                print(f"Regenerating {missing_embeddings.sum()} missing embeddings...")
                df.loc[missing_embeddings, "Vector"] = df.loc[missing_embeddings, "Text"].apply(
                    lambda text: self._generate_and_store_embedding(text)
                )

                df.to_json(self.output_json_path, orient="records", indent=4)
                print(f"Updated embeddings saved in: {self.output_json_path}")

            return df

        df = pd.read_csv(self.csv_path)

        if "Vector" not in df.columns:
            df["Vector"] = ""

        missing_embeddings = df["Vector"].isna() | (df["Vector"].astype(str).str.strip() == "")
        if missing_embeddings.any():
            print(f"Generating embeddings for {missing_embeddings.sum()} missing entries...")
            df.loc[missing_embeddings, "Vector"] = df.loc[missing_embeddings, "Text"].apply(
                lambda text: self._generate_and_store_embedding(text)
            )

        df.to_json(self.output_json_path, orient="records", indent=4)
        print(f"Embeddings saved in: {self.output_json_path}")
        return df

    def _generate_and_store_embedding(self, text):
        """Helper function to generate embeddings and store them as lists (JSON serializable)."""
        vector = self.embedding_service.get_embedding(text)
        if isinstance(vector, np.ndarray):
            return vector.tolist()
        return vector
