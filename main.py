""" The main function of rPPG deep learning pipeline."""
# python main.py --config_file --config ./configs/EMOconfigs/Arsl_BC_PHYSMAMBA.yaml
import time
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import get_config
from trainer import trainer
from unsupervised_encoders.unsupervised_predictor import unsupervised_predict

from dataset import data_loader
from dataset.data_loader.collate_fn import custom_collate_fn

num_workers = 2

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # default = False
torch.backends.cudnn.enabled = True


# Create generators for reproducibility
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def add_args(parser):
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/MAHNOB-HCI_PhysMamba.yaml",
                        type=str, help="Path to config file")
    """
    Neural Method Sample YAML LIST:
      Arsl_BC_PHYSMAMBA.yaml
      Vlnc_BC_PHYSMAMBA.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      MAHNOB-HCI_UNSUPERVISED.yaml
    """
    return parser

def select_trainer(config, data_loader_dict):
    """
    Returns the Trainer object based on config.ENCODER.NAME and config.TEST.LABEL_COLUMN.
    """
    if config.TEST.DATA.LABEL_COLUMN in ['BC_Arsl', 'BC_Vlnc']:
        '''
        Not Supported Yet !
        Will be added in the future.
        
        if config.ENCODER.NAME == "Physnet":
            return trainer.PhysnetTrainer_BC.PhysnetTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "iBVPNet":
            return trainer.iBVPNetTrainer_BC.iBVPNetTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "FactorizePhys":
            return trainer.FactorizePhysTrainer_BC.FactorizePhysTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "Tscan":
            return trainer.TscanTrainer_BC.TscanTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "EfficientPhys":
            return trainer.EfficientPhysTrainer_BC.EfficientPhysTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "DeepPhys":
            return trainer.DeepPhysTrainer_BC.DeepPhysTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "BigSmall":
            return trainer.BigSmallTrainer_BC.BigSmallTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "PhysFormer":
            return trainer.PhysFormerTrainer_BC.PhysFormerTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "RhythmFormer":
            return trainer.RhythmFormerTrainer_BC.RhythmFormerTrainer(config, data_loader_dict)
        '''
        if config.ENCODER.NAME == 'TemporalBranch':
            return trainer.TemporalBranchTrainer_BC.TemporalBranchTrainer_BC(config, data_loader_dict)
        elif config.ENCODER.NAME == "PhysMamba":
            return trainer.PhysMambaTrainer_BC.PhysMambaTrainer(config, data_loader_dict)
        else:
            raise ValueError("Your Model is Not Supported Yet!")
    
    elif config.TEST.DATA.LABEL_COLUMN in ['3C_Arsl', '3C_Vlnc']:
        '''
        Not Supported Yet !
        Will be added in the future.
        
        if config.ENCODER.NAME == "Physnet":
            return trainer.PhysnetTrainer_3C.PhysnetTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "iBVPNet":
            return trainer.iBVPNetTrainer_3C.iBVPNetTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "FactorizePhys":
            return trainer.FactorizePhysTrainer_3C.FactorizePhysTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "Tscan":
            return trainer.TscanTrainer_3C.TscanTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "EfficientPhys":
            return trainer.EfficientPhysTrainer_3C.EfficientPhysTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "DeepPhys":
            return trainer.DeepPhysTrainer_3C.DeepPhysTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "BigSmall":
            return trainer.BigSmallTrainer_3C.BigSmallTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "PhysFormer":
            return trainer.PhysFormerTrainer_3C.PhysFormerTrainer(config, data_loader_dict)
        elif config.ENCODER.NAME == "RhythmFormer":
            return trainer.RhythmFormerTrainer_3C.RhythmFormerTrainer(config, data_loader_dict)
        '''
        if config.ENCODER.NAME == "PhysMamba":
            return trainer.PhysMambaTrainer_3C.PhysMambaTrainer(config, data_loader_dict)
        else:
            raise ValueError("Your Model is Not Supported Yet!")
    else:
        raise ValueError("Set your LABEL_COLUMN!")

def unsupervised_method_inference(config, data_loader):
    """Runs unsupervised method inference based on the method specified in the config."""
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        elif unsupervised_method == "OMIT":
            unsupervised_predict(config, data_loader, "OMIT")
        else:
            raise ValueError("Not supported unsupervised method!")

def create_data_loaders(loader_cls, config):
    """
    Creates and returns the train, valid, and test data loaders.
    """
    if config.ReMOTION_MODE == "train_and_test":
        train_data = loader_cls(name="train",
                                data_path=config.TRAIN.DATA.DATA_PATH,
                                config_data=config.TRAIN.DATA,
                                device=config.DEVICE)
        valid_data = loader_cls(name="valid",
                                data_path=config.VALID.DATA.DATA_PATH,
                                config_data=config.VALID.DATA,
                                device=config.DEVICE)
    else:
        train_data = None
        valid_data = None
    test_data = loader_cls(name="test",
                           data_path=config.TEST.DATA.DATA_PATH,
                           config_data=config.TEST.DATA,
                           device=config.DEVICE)
    
    if config.ReMOTION_MODE == "train_and_test":
        train_loader = DataLoader(dataset=train_data,
                                num_workers=num_workers,
                                batch_size=config.TRAIN.BATCH_SIZE,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=train_generator,
                                collate_fn=custom_collate_fn)
        valid_loader = DataLoader(dataset=valid_data,
                                num_workers=num_workers,
                                batch_size=config.VALID.BATCH_SIZE,
                                shuffle=False,
                                worker_init_fn=seed_worker,
                                generator=general_generator,
                                collate_fn=custom_collate_fn)
    else:
        train_loader = None
        valid_loader = None
    test_loader = DataLoader(dataset=test_data,
                             num_workers=num_workers,
                             batch_size=config.INFERENCE.BATCH_SIZE,
                             shuffle=False,
                             worker_init_fn=seed_worker,
                             generator=general_generator,
                             collate_fn=custom_collate_fn)
    return {"train": train_loader, "valid": valid_loader, "test": test_loader}

        
def run_LOSO_fold(config, subject_id):
    """
    Runs one LOSO fold with the given subject_id as the test subject.
    Updates config.TRAIN.DATA.current_subject.
    Returns validation metric and best epoch.
    """
    print(f"\n=== LOSO: Leaving subject {subject_id} out ===")
    config.defrost()
    config.TRAIN.DATA.current_subject = subject_id
    config.VALID.DATA.current_subject = subject_id
    config.TEST.DATA.current_subject = subject_id
    config.freeze()

    loader_cls = data_loader.MAHNOBHCILoader.MAHNOBHCILoader
    data_loader_dict = create_data_loaders(loader_cls, config)

    trainer_obj = select_trainer(config, data_loader_dict)
    trainer_obj.train(data_loader_dict)
    trainer_obj.test(data_loader_dict)
    return trainer_obj.min_valid_loss, trainer_obj.best_epoch

def run_kfold_fold(config, fold_index):
    """
    Runs one k_fold fold based on the given fold index.
    Updates config.TRAIN.DATA.FOLD_INDEX.
    Returns validation metric and best epoch.
    """
    print(f"\n=== k_fold: Fold {fold_index} ===")
    config.defrost()
    config.TRAIN.DATA.FOLD_INDEX = fold_index
    config.VALID.DATA.FOLD_INDEX = fold_index
    config.TEST.DATA.FOLD_INDEX = fold_index
    config.freeze()

    loader_cls = data_loader.MAHNOBHCILoader.MAHNOBHCILoader
    data_loader_dict = create_data_loaders(loader_cls, config)

    trainer_obj = select_trainer(config, data_loader_dict)
    trainer_obj.train(data_loader_dict)
    trainer_obj.test(data_loader_dict)
    return trainer_obj.min_valid_loss, trainer_obj.best_epoch

def train_and_test(config, data_loader_dict):
    """Trains and tests the model."""
    model_trainer = select_trainer(config, data_loader_dict)
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)

def test(config, data_loader_dict):
    """Tests the model."""
    model_trainer = select_trainer(config, data_loader_dict)
    model_trainer.test(data_loader_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    config = get_config(args)
    print("Configuration:")
    print(config, end="\n\n")

    # The overall pipeline is controlled by ReMOTION_MODE.
    # SPLIT_METHOD only defines how the data is partitioned.
    if config.ReMOTION_MODE in ["train_and_test", "only_test"]:
        # If cross-validation is set in the data config, perform it.
        if config.TRAIN.DATA.SPLIT_METHOD == "LOSO":
            loader_cls = data_loader.MAHNOBHCILoader.MAHNOBHCILoader
            temp_loader = loader_cls(name="temp",
                                     data_path=config.TRAIN.DATA.DATA_PATH,
                                     config_data=config.TRAIN.DATA,
                                     device=config.DEVICE)
            subject_list = temp_loader.get_all_subject_ids()
            print("Subject IDs:", subject_list)
            fold_results = {}
            for subj in subject_list:
                val_loss, best_epoch = run_LOSO_fold(config, subj)
                fold_results[subj] = {"val_loss": val_loss, "best_epoch": best_epoch}
            print("\n=== LOSO Cross-Validation Results ===")
            for subj, metrics in fold_results.items():
                print(f"Subject {subj}: Val Loss = {metrics['val_loss']}, Best Epoch = {metrics['best_epoch']}")
        elif config.TRAIN.DATA.SPLIT_METHOD == "k_fold":
            num_folds = config.TRAIN.DATA.NUM_FOLDS
            fold_results = {}
            for fold_idx in range(num_folds):
                val_loss, best_epoch = run_kfold_fold(config, fold_idx)
                fold_results[fold_idx] = {"val_loss": val_loss, "best_epoch": best_epoch}
            print("\n=== k_fold Cross-Validation Results ===")
            for fold_idx, metrics in fold_results.items():
                print(f"Fold {fold_idx}: Val Loss = {metrics['val_loss']}, Best Epoch = {metrics['best_epoch']}")
        else:
            # Normal supervised mode (without cross-validation)
            loader_cls = data_loader.MAHNOBHCILoader.MAHNOBHCILoader
            if config.ReMOTION_MODE == "train_and_test":
                data_loader_dict = create_data_loaders(loader_cls, config)
                train_and_test(config, data_loader_dict)
            elif config.ReMOTION_MODE == "only_test":
                data_loader_dict = create_data_loaders(loader_cls, config)
                test(config, data_loader_dict)
    elif config.ReMOTION_MODE == "unsupervised_method":
        loader_cls = data_loader.MAHNOBHCILoader.MAHNOBHCILoader
        unsupervised_data = loader_cls(name="unsupervised",
                                       data_path=config.UNSUPERVISED.DATA.DATA_PATH,
                                       config_data=config.UNSUPERVISED.DATA,
                                       config_mode=config.ReMOTION_MODE,
                                       device=config.DEVICE)
        data_loader_dict = {
            "unsupervised": DataLoader(dataset=unsupervised_data, num_workers=num_workers,
                                       batch_size=1, shuffle=False,
                                       worker_init_fn=seed_worker, generator=general_generator)
        }
        unsupervised_method_inference(config, data_loader_dict)
    else:
        raise ValueError("Unsupported ReMOTION_MODE!")