import pytorch_lightning as pl
from attr import evolve
from pytorch_lightning.loggers import TensorBoardLogger

from moco import MoCoMethod
from moco import MoCoMethodParams


def main():
    # base_config = MoCoMethodParams(
        # lr=0.8,
        # batch_size=256,
        # gather_keys_for_queue=False,
        # loss_type="ip",
        # use_negative_examples_from_queue=False,
        # use_both_augmentations_as_queries=True,
        # mlp_normalization="bn",
        # prediction_mlp_layers=2,
        # projection_mlp_layers=2,
        # m=0.996,
        # use_momentum_schedule=True,
    # )
    two_resnet50s = MoCoMethodParams(
        use_eqco_margin=True, 
        eqco_alpha=65536, 
        K=0,
        use_negative_examples_from_batch=True,
        use_negative_examples_from_queue=False,
        max_epochs=200,
        encoder_arch="resnet50",
        embedding_dim=2048,
        use_both_augmentations_as_queries=True,
        prediction_mlp_layers=0
    )
    two_botnets = MoCoMethodParams(
        use_eqco_margin=True, 
        eqco_alpha=65536, 
        K=0,
        use_negative_examples_from_batch=True,
        use_negative_examples_from_queue=False,
        max_epochs=200,
        encoder_arch="BoTNet",
        embedding_dim=2048,
        fmap_size=24,
        dim=256,
        batch_size=64,
        use_both_augmentations_as_queries=True,
        prediction_mlp_layers=0
    )
    BoTRes50_eqco = MoCoMethodParams(
        use_eqco_margin=True, 
        eqco_alpha=65536, 
        K=0,
        use_negative_examples_from_batch=True,
        use_negative_examples_from_queue=False,
        max_epochs=200,
        encoder_arch="BoTRes",
        embedding_dim=2048,
        fmap_size=24,
        dim=256,
        batch_size=64,
        use_both_augmentations_as_queries=True,
        prediction_mlp_layers=0
    )
    BoTRes50_byol = MoCoMethodParams(
        max_epochs=200,
        encoder_arch="BoTRes",
        embedding_dim=2048,
        fmap_size=24,
        dim=256,
        batch_size=64,
        prediction_mlp_layers = 2,
        mlp_normalization = "bn",
        loss_type = "ip",
        use_negative_examples_from_queue = False,
        # use_both_augmentations_as_queries = True,
        use_momentum_schedule = True,
        optimizer_name = "lars",
        exclude_matching_parameters_from_lars = [".bias", ".bn"],
        loss_constant_factor = 2
    )
    base_config = BoTRes50_byol
    configs = {
        "base": base_config,
        # "pred_only": evolve(base_config, mlp_normalization=None, prediction_mlp_normalization="bn"),
        # "proj_only": evolve(base_config, mlp_normalization="bn", prediction_mlp_normalization=None),
        # "no_norm": evolve(base_config, mlp_normalization=None),
        # "layer_norm": evolve(base_config, mlp_normalization="ln"),
        # "xent": evolve(base_config, use_negative_examples_from_queue=True, loss_type="ce", mlp_normalization=None, lr=0.02),
    }
    for seed in range(1):
        for name, config in configs.items():
            method = MoCoMethod(config)
            logger = TensorBoardLogger("tb_logs", name=f"{name}_{seed}")

            trainer = pl.Trainer(gpus=1, max_epochs=base_config.max_epochs, logger=logger)

            trainer.fit(method)


if __name__ == "__main__":
    main()
