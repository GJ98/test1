import pytorch_lightning as pl
from module.data import *
from module.mrc import *
from module.retrieval import *
from datasets import load_from_disk
import hydra

@hydra.main(config_path="./config", config_name="combine", version_base=None)
def main(config):

    mode = 'validation'
    mode = 'test'

    if mode == 'validation':
        dataset_list = get_dataset_list(config.mrc.data.dataset_name)
        dataset = concatenate_datasets([ds['validation'] for ds in dataset_list]).select(range(100))

        retrieval = TfIdfRetrieval(config.retrieval)
        retrieval.fit()
        retrieval.create_embedding_vector()
        doc_ids, docs = retrieval.search(dataset['question'])

        dataset = dataset.remove_columns(['context'])
        dataset = dataset.add_column('context', docs)
        print(dataset['context'][0])
        print(len(dataset['context'][1]))

        data_module = MrcDataModule(config.mrc)
        data_module.eval_examples = dataset
        data_module.eval_dataset = data_module.get_dataset(dataset,
                                                           data_module.prepare_validation_features)
        val_dataloader = data_module.val_dataloader()

        mrc = MrcLightningModule.load_from_checkpoint("/data/ephemeral/home/gj/mrc-template/test/kxcrgozn/checkpoints/epoch=1-step=26.ckpt")
        mrc.eval_examples = dataset
        mrc.eval_dataset = data_module.eval_dataset
        trainer = pl.Trainer()
        trainer.validate(model=mrc, dataloaders=val_dataloader)

    else:
        dataset = load_from_disk("./data/default/test_dataset/")['validation'].select(range(100))
        retrieval = TfIdfRetrieval(config.retrieval)
        retrieval.fit()
        retrieval.create_embedding_vector()
        doc_ids, docs = retrieval.search(dataset['question'])

        dataset = dataset.add_column('context', docs)
        print(dataset['context'][0])
        print(len(dataset['context'][1]))

        data_module = MrcDataModule(config.mrc)
        data_module.eval_examples = dataset
        data_module.eval_dataset = data_module.get_dataset(dataset,
                                                           data_module.prepare_validation_features)
        val_dataloader = data_module.val_dataloader()

        mrc = MrcLightningModule.load_from_checkpoint("/data/ephemeral/home/gj/mrc-template/test/kxcrgozn/checkpoints/epoch=1-step=26.ckpt")
        mrc.eval_examples = dataset
        mrc.eval_dataset = data_module.eval_dataset
        trainer = pl.Trainer()
        trainer.test(model=mrc, dataloaders=val_dataloader)



if __name__=="__main__":
    main()