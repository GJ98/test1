import pytorch_lightning as pl
from module.data import *
from module.mrc import *
import hydra


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    data_module = MrcDataModule(config)
    data_module.setup()

    eval_dataset = data_module.eval_dataset
    test_dataset = data_module.test_dataset
    eval_examples = data_module.eval_examples
    test_examples = data_module.test_examples
    print('test_examples')
 
    model_module = MrcLightningModule(config, eval_dataset, test_dataset, eval_examples, test_examples)

   # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(num_sanity_val_steps=0, accelerator="gpu", devices=1, max_epochs=config.train.num_train_epochs, log_every_n_steps=1)

    # Train part
    trainer.fit(model=model_module, datamodule=data_module)

if __name__ == "__main__":
    main()
