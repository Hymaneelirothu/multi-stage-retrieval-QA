from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_dataset(dataset_name="nq"):
    data_path = util.download_and_unzip(f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip", dataset_name)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels
