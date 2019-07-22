""" Train or Test DNN

Usage:
```
    python main.py {--train, --test={seen, unseen}}
                   [--room_train ROOM_TRAIN]
                   [--room_test ROOM_TEST]
                   [--logdir LOGDIR]
                   [--n_epochs MAX_EPOCH]
                   [--from START_EPOCH]
                   [--device DEVICES] [--out_device OUT_DEVICE]
                   [--batch_size B]
                   [--learning_rate LR]
                   [--weight_decay WD]
                   [--model_name MODEL]
```

More parameters are in `hparams.py`.
- specify `--train` or `--test {seen, unseen}`.
- ROOM_TRAIN: room used to train
- ROOM_TEST: room used to test
- LOGDIR: log directory
- MAX_EPOCH: maximum epoch
- START_EPOCH: start epoch (Default: -1)
- DEVICES, OUT_DEVICE, B, LR, WD, MODEL: read `hparams.py`.
"""
# noinspection PyUnresolvedReferences
import matlab.engine

import os
import shutil
from argparse import ArgumentError, ArgumentParser

from torch.utils.data import DataLoader

from hparams import hp
from dataset import MulchWavDataset
from train import Trainer

tfevents_fname = 'events.out.tfevents.*'
form_overwrite_msg = 'The folder "{}" already has tfevent files. Continue? [y/n]\n'

parser = ArgumentParser()

parser.add_argument('--train', action='store_true', )
parser.add_argument('--test', choices=('seen', 'unseen'), metavar='DATASET')
parser.add_argument('--from', type=int, default=-1, dest='epoch', metavar='EPOCH')

args = hp.parse_argument(parser)
del parser
if not (args.train ^ (args.test is not None)) or args.epoch < -1:
    raise ArgumentError

# directory
logdir_train = hp.logdir / 'train'
if (args.train and args.epoch == -1 and
        logdir_train.exists() and list(logdir_train.glob(tfevents_fname))):
    ans = input(form_overwrite_msg.format(logdir_train))
    if ans.lower() == 'y':
        shutil.rmtree(logdir_train)
        try:
            os.remove(hp.logdir / 'summary.txt')
        except FileNotFoundError:
            pass
        try:
            os.remove(hp.logdir / 'hparams.txt')
        except FileNotFoundError:
            pass
    else:
        exit()
os.makedirs(logdir_train, exist_ok=True)

if args.test:
    logdir_test = hp.logdir
    if hp.room_test == hp.room_train:
        logdir_test /= args.test
    else:
        logdir_test /= f'{args.test}_{hp.room_test}'
    if logdir_test.exists() and list(logdir_test.glob(tfevents_fname)):
        ans = input(form_overwrite_msg.format(logdir_test))
        if ans.lower().startswith('y'):
            shutil.rmtree(logdir_test)
            os.makedirs(logdir_test)
        else:
            exit()
    os.makedirs(logdir_test, exist_ok=True)

# epoch, state dict
first_epoch = args.epoch + 1
if first_epoch > 0:
    path_state_dict = logdir_train / f'{hp.model_name}_{args.epoch}.pt'
    if not path_state_dict.exists():
        raise FileNotFoundError(path_state_dict)
else:
    path_state_dict = None

# Training + Validation Set
dataset_temp = MulchWavDataset('train', n_file=hp.n_file)
dataset_train, dataset_valid = MulchWavDataset.split(dataset_temp, (hp.train_ratio, -1))

# run
trainer = Trainer(path_state_dict)
if args.train:
    loader_train = DataLoader(dataset_train,
                              batch_size=hp.batch_size,
                              num_workers=hp.num_disk_workers,
                              collate_fn=dataset_train.pad_collate,
                              pin_memory=True,
                              shuffle=True,
                              )
    loader_valid = DataLoader(dataset_valid,
                              batch_size=hp.batch_size,
                              num_workers=hp.num_disk_workers,
                              collate_fn=dataset_valid.pad_collate,
                              pin_memory=True,
                              shuffle=False,
                              )

    trainer.train(loader_train, loader_valid, logdir_train, first_epoch)
else:  # args.test
    # Test Set
    dataset_test = MulchWavDataset(args.test,
                                   n_file=hp.n_file // 4,
                                   random_by_utterance=False,
                                   normalization_in=dataset_temp.norm_in,
                                   normalization_out=dataset_temp.norm_out
                                   )
    loader = DataLoader(dataset_test,
                        batch_size=1,
                        num_workers=hp.num_disk_workers,
                        collate_fn=dataset_test.pad_collate,
                        pin_memory=True,
                        shuffle=False,
                        )

    # noinspection PyUnboundLocalVariable
    trainer.test(loader, logdir_test)
