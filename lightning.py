from LightningDataset import LightningOpenImages
from default_config import mse_lpips_args
from LightningTrainer import HiFiC 
from src.helpers import utils 
import os
import pytorch_lightning as pl

args = mse_lpips_args
args = utils.setup_generic_signature(args, special_info=args.model_type)

logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath("train.py"))

dm = LightningOpenImages(args)
model = HiFiC(args)

trainer = pl.Trainer(gpus=[0], amp_level='O2', auto_scale_batch_size=True, max_epochs=20, limit_val_batches=2,
                     val_check_interval=0.5, progress_bar_refresh_rate=20, automatic_optimization=False, precision=16,
                     benchmark=True, resume_from_checkpoint=None, amp_backend='apex', num_sanity_val_steps=0)

trainer.fit(model, dm)
trainer.save_checkpoint("example.ckpt")