import logging
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

import torch.utils.data as data

import pandas as pd
import numpy as np


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    print('model',model)
    print(model.parameters())

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

class Tor_Dataset(data.Dataset):
    def __init__(self, csv_file, data_dir):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
        """
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir

        self.fail_state = [0,0,0,0,0,0,0]

    def __len__(self):
        return len(self.data)

    def clear_clusters(self):
        self.clusters_epoch = [];

    def __getitem__(self,idx):
        """
        Arguments:
            idx (int): CSV file index for training/testing
        Returns:
            poly (polymer): carbohydrate polymer class instance
        """

        if torch.is_tensor(idx):
            idx = idx.tolist();

        pdb_name = self.data.iloc[idx,0]
        chain_name = self.data.iloc[idx,1]

        coor, resid = self.get_coor(pdb_name)
        polymer = self.chain_to_poly(chain_name,coor,resid)
        
        return polymer

        def get_coor(self,pdb):
            """
            Arguments:
                pdb (str): pdb file to read in
            Returns:
                coor (arr): 3D cartesian coordinates
                resid (arr): Residue atom names [atom_name, residue_num, chain_name, res_name]
            """
            structure=parser.get_structure("carb", self.data_dir + pdb)
            coor = []
            resid = []
            
            models = structure.get_models()
            models = list(models)
            if len(models) == 0:
                return [],[];
            
            for m in range(len(models)):
                chains = list(models[m].get_chains())
                for c in range(len(chains)):
                    residues = list(chains[c].get_residues())
                    for r in range(len(residues)):

                        res = residues[r].get_resname()
                        if res == 'HOH':
                            continue;

                        atoms = list(residues[r].get_atoms())

                        for a in range(len(atoms)):
                            at = atoms[a]

                            if 'H' in at.get_name():
                                continue;

                            #print(str(residues[r].get_parent().id).strip())

                            coor.append( at.get_coord() )
                            resid.append( [ str(at.get_name()), str(residues[r].id[1]).strip(), str(chains[c].id).strip(), str(residues[r].get_resname()) ] )
                            
                #print(len(coor))
                return np.array(coor), resid


        #enter coordinates and chain and get the polymer object instance
        def chain_to_poly(self, my_chain,coor,res):
            """
            params:
                chain (str): chain identifier
                coor (arr n x 3): coordinates
                res (arr n x 4): residue information of each atom (aname, resnum, chain, resname)
            return:
                polymer
            """


            polymer = []
            c_resnum = -1;
            c_resname = ''
            n = []
            c = []
            
            
            for i in range(len(res)):
                ii = res[i]
                aname = ii[0]
                resnum = ii[1]
                chain = ii[2]
                resname = ii[3]
                
                

                if c_resnum != resnum:
                    if c_resnum != -1 and len(n) > 1:
                        #print(str(c_resnum),c_resname,np.array(c),n)
                        m = mono(str(c_resnum),c_resname,np.array(c),n)
                        polymer.append(m)
                    #reset
                    c_resnum = resnum;
                    c_resname = resname
                    n = []
                    c = []

                if chain == my_chain:

                    n.append(aname)
                    c.append(coor[i])

            
            if len(n) > 1:
                m = mono(str(c_resnum),c_resname,np.array(c),n)
                polymer.append(m)

            return poly(polymer)


class TorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../pdb_pre", full_list: str = '../../pdb_pre/tor_bois.txt',
                  batch_size: int = 32, num_workers: int = 8, pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self):
        full_dataset = Tor_Dataset(self.data_dir)
        self.train, self.val = random_split(
            full_dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size=self.batch_size)





def get_loaders(train_file, test_file, root_dir="../",
                batch_size=1, num_workers=0, train_cluster=True, val_cluster=False,
                knn=[6,12,18,24], pin_memory=True, use_pad=True, pad_size=1750):

    train_ds = CSV_Dataset( csv_file=train_file, root_dir=root_dir, train=1, use_clusters=train_cluster,nn=knn,
                            use_pad=use_pad, pad_size=pad_size)
    train_loader = DataLoader( train_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=True )

    val_ds = CSV_Dataset( csv_file=test_file, root_dir=root_dir, train=1, use_clusters=val_cluster,nn=knn, val=True, use_pad=False)
    val_loader = DataLoader( val_ds, batch_size=1, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False )

    return train_loader, val_loader



def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()