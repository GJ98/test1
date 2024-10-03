import pytorch_lightning as pl
from module.data import *
from module.model_retriever import DenseRetriever, BertEncoder
import hydra


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):

    p_encoder = BertEncoder.from_pretrained(config.model.plm_name)
    q_encoder = BertEncoder.from_pretrained(config.model.plm_name)
    retriever = DenseRetriever(config, q_encoder, p_encoder)
    retriever.setup()
    retriever.fit()
    retriever.make_dense_embedding_matrix()

    print(retriever.search("우리나라 대통령은 누구야?"))


if __name__ == "__main__":
    main()
