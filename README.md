<div align="center">

# DL-based SIC for BLE and IEEE 802.15.4

</div>

Deep Learning-based Successive Interference Cancellation for BLE and IEEE 802.15.4 signals.


---
## Unit Testing
To execute unit tests, run them as Python modules from the project root. For example:

```bash
python -m dl_sic.model.ctdcr_net
```
---
## Checkpoints and Logging

When running `train.py`, the following files are written to `./checkpoints/` (relative to the folder where `train.py` is executed):

- `best_model_weights.pth` – stores the weights of the best model so far
- `last_checkpoint.pth` – stores the most recent model checkpoint
- `training_log.json` – records training history

⚠️ These files are **overwritten** each time training is started.

If the `--resume` flag is provided, training will continue from the last stored checkpoint instead of starting fresh.

---
## Installation

1. **Install PyTorch**
   Choose the appropriate command for your system from [pytorch.org](https://pytorch.org/get-started/locally/).
   Example for CPU-only on Linux:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Install the remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```
