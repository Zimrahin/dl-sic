import torch
from inspect import signature

from model.complex_tdcr_net import ComplexTDCRNet
from model.tdcr_net import RealTDCRNet
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
        model_signature = signature(model_class.__init__)
        valid_params = {
            key: value
            for key, value in model_params.items()
            if key in model_signature.parameters
        }

        if model_class not in [ComplexTDCRNet] and dtype.is_complex:
            raise ValueError(f"Real models cannot use complex dtype {dtype}")
        if "dtype" in model_signature.parameters:
            model = model_class(**valid_params, dtype=dtype)
        else:
            model = model_class(**valid_params)

        print(f"Using model: {model.__class__.__name__}")
        print("Parameters: ")
        for key, value in valid_params.items():
            print(f"  - {key}: {value}")

        return model.to(device)
