import torch

from model.complex_tdcr_net import ComplexTDCRNet
from model.real_tdcr_net import RealTDCRNet
from model.tcn_conformer_net import TCNConformerNet


class ModelFactory:
    @staticmethod
    def create_model(
        model_type: str, model_params: dict, dtype: torch.dtype, device: torch.device
    ):
        """Instantiate model"""
        models: dict = {
            "complextdcr": ComplexTDCRNet,
            "tdcr": RealTDCRNet,
            "tcnconformer": TCNConformerNet,
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = models[model_type]
        if model_class not in [ComplexTDCRNet] and dtype.is_complex:
            raise ValueError(f"Real models cannot use complex dtype {dtype}")

        if model_class in [TCNConformerNet]:
            model = model_class(**model_params)
        else:
            model = model_class(**model_params, dtype=dtype)

        print(f"Using model: {model.__class__.__name__}")
        return model.to(device)
