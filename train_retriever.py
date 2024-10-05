import pytorch_lightning as pl
import module.data as module_data
from module.retriever import DenseRetriever
import module.encoder as module_encoder
import hydra


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):

    data_module = getattr(module_data, config.data.retriever_data_module)(config)

    p_encoder = getattr(module_encoder, config.model.encoder).from_pretrained(config.model.encoder_plm_name)
    q_encoder = getattr(module_encoder, config.model.encoder).from_pretrained(config.model.encoder_plm_name)
    retriever = DenseRetriever(config, q_encoder, p_encoder)

    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(num_sanity_val_steps=0, accelerator="gpu", devices=1, max_epochs=config.train.num_train_epochs, log_every_n_steps=1)
    trainer.fit(model=retriever, datamodule=data_module)

    retriever.create_embedding_vector()
    print(retriever.search("우리나라 대통령은 누구야?"))

if __name__ == "__main__":
    main()
