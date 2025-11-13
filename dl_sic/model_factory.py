import torch

from model.complex_tdcr_net import ComplexTDCRNet
from model.real_tdcr_net import RealTDCRNet


class ModelFactory:
    @staticmethod
    def create_model(
        model_type: str, model_params: dict, dtype: torch.dtype, device: torch.device
    ):
        """Instantiate model"""

        if model_type == "complex":
            model = ComplexTDCRNet(**model_params, dtype=dtype)
        elif model_type == "real":
            if dtype.is_complex:
                raise ValueError(f"Real model cannot use complex dtype {dtype}")
            model = RealTDCRNet(**model_params, dtype=dtype)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"Using model: {model.__class__.__name__}")
        return model.to(device)
