from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_dataset(num_docs=100):
    try:
        # Use the fixed URL directly for the Natural Questions dataset
        data_path = util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip", "nq")
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        # Check if corpus is loaded properly
        if corpus is None or len(corpus) == 0:
            raise ValueError("Corpus not loaded properly or is empty.")

        # Limit the size of the corpus for memory efficiency
        small_corpus = {k: corpus[k] for k in list(corpus.keys())[:num_docs]}
        return small_corpus, queries, qrels

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

