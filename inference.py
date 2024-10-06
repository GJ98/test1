import pytorch_lightning as pl
from module.data import *
from module.mrc import *
from module.retrieval import *
from datasets import load_from_disk
import hydra

@hydra.main(config_path="./config", config_name="combine", version_base=None)
def main(config):

    mode = 'validation'
    only_mrc = False
    model_checkpoint = '/data/ephemeral/home/gj/mrc-template/test/2pl0fmh1/checkpoints/epoch=1-step=26.ckpt'
    mode = 'test'

    if mode == 'validation':
        # 1. load eval examples
        dataset_list = get_dataset_list(config.mrc.data.dataset_name)
        eval_examples = concatenate_datasets([ds['validation'] for ds in dataset_list]).select(range(100))

        if not only_mrc:
            # 2. retrieve context
            retrieval = TfIdfRetrieval(config.retrieval)
            retrieval.fit()
            retrieval.create_embedding_vector()
            doc_ids, docs = retrieval.search(eval_examples['question'])

            # 3. change original context to retrieved context
            eval_examples = eval_examples.remove_columns(['context'])
            eval_examples = eval_examples.add_column('context', docs)

        # 4. make eval_dataset & eval_dataloader
        data_module = MrcDataModule(config.mrc)
        data_module.eval_dataset = data_module.get_dataset(eval_examples)
        val_dataloader = data_module.val_dataloader()

        # 5. load mrc model 
        mrc = MrcLightningModule.load_from_checkpoint(model_checkpoint)
        # 5.1. put eval examples and eval dataset to inference
        mrc.eval_examples = eval_examples
        mrc.eval_dataset = data_module.eval_dataset

        # 6. inference eval dataset
        trainer = pl.Trainer()
        trainer.validate(model=mrc, dataloaders=val_dataloader)

    else:
        # 1. load test examples
        test_examples = load_from_disk("./data/default/test_dataset/")['validation'].select(range(100))

        # 2. retrieve context
        retrieval = TfIdfRetrieval(config.retrieval)
        retrieval.fit()
        retrieval.create_embedding_vector()
        doc_ids, docs = retrieval.search(test_examples['question'])

        # 3. insert retrieved context column
        test_examples = test_examples.add_column('context', docs)

        # 4. make eval_dataset & eval_dataloader
        data_module = MrcDataModule(config.mrc)
        data_module.test_dataset = data_module.get_dataset(test_examples)
        test_dataloader = data_module.test_dataloader()

        # 5. load mrc model 
        mrc = MrcLightningModule.load_from_checkpoint(model_checkpoint)
        # 5.1. put eval examples and eval dataset to inference
        mrc.test_examples = test_examples
        mrc.test_dataset = data_module.test_dataset

        # 6. inference eval dataset
        trainer = pl.Trainer()
        trainer.test(model=mrc, dataloaders=test_dataloader)



if __name__=="__main__":
    main()