from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_dataset(num_docs=100):
    # Use the fixed URL directly for the Natural Questions dataset
    data_path = util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip", "nq")
    
    # Load the dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    # Limit the size of the corpus for memory efficiency
    small_corpus = {k: corpus[k] for k in list(corpus.keys())[:num_docs]}
    
    return small_corpus, queries, qrels
