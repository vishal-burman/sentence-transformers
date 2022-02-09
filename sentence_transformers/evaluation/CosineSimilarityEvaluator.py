from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics.pairwise import paired_cosine_distances
import numpy as np
import logging
import os
import csv
from typing import List

logger = logging.getLogger(__name__)

class CosineSimilarityEvaluator(SentenceEvaluator):
    """
    Computes the Cosine-Similarity between the computed sentence embedding
    and some target sentence embedding.
    The CosineSimilarity is computed between ||teacher.encode(source_sentences) - student.encode(target_sentences)||.
    :param source_sentences: Source sentences are embedded with the teacher model
    :param target_sentences: Target sentences are ambedding with the student model.
    :param show_progress_bar: Show progress bar when computing embeddings
    :param batch_size: Batch size to compute sentence embeddings
    :param name: Name of the evaluator
    :param write_csv: Write results to CSV file
    """

    def __init__(self, source_sentences: List[str], target_sentences: List[str], teacher_model = None, show_progress_bar: bool = False, batch_size: int = 32, name: str = '', write_csv: bool = True):

        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.csv_file = "cosine_similarity_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Cosine Similarity"]
        self.write_csv = write_csv

    def __call__(self, model, output_path, epoch  = -1, steps = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        source_embeddings = model.encode(self.source_sentences, show_progress_bar=self.show_progress_bar, batch_size=batch_size, convert_to_numpy=True)
        target_embeddings = model.encode(self.target_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=True)

        #cs = cosine_similarity(self.source_embeddings, target_embeddings).diagonal().mean()
        cs = (1 - paired_cosine_distances(source_embeddings, target_embeddings)).mean()

        logger.info("Cosine Similarity evaluation (lower = better) on "+self.name+" dataset"+out_txt)
        logger.info(f"Cosine Similarity: {cs: .4f}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, cs])

        return cs
