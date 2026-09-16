# Running on the GPU VM

The training code is identical to what runs locally on Apple Silicon --
`saffron_cut.device.get_device()` picks CUDA automatically whenever
`torch.cuda.is_available()`. Nothing in the model, dataloader or training
loop is Mac- or MPS-specific.

```bash
ssh <gpu-vm>
git clone git@github.com:mralinp/RefineDet-detecting-saffron-flowers-endpoint.git
cd RefineDet-detecting-saffron-flowers-endpoint

# CUDA build of torch first (match the VM's driver / CUDA version -- cu124
# below works for most recent drivers; check `nvidia-smi` if unsure)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`data/` is tracked in `origin`, so the `git clone` above already brought
`Labeled/`, `Unlabeled/` and `Test/` with it -- no separate copy step.

Confirm CUDA is actually selected, then train:

```bash
python -c "from saffron_cut.device import device_report; print(device_report())"
python main.py 2>&1 | tee train.log          # full run: train (+ self-train) then predict Test/
```

`checkpoints/` and the filled-in `data/Test/*.csv` are what you need back
(neither is tracked in git -- checkpoints are gitignored, Test's
predictions are generated, not source data); grab them with rsync:

```bash
rsync -avz <gpu-vm>:~/RefineDet-detecting-saffron-flowers-endpoint/checkpoints/ ./checkpoints/
rsync -avz <gpu-vm>:~/RefineDet-detecting-saffron-flowers-endpoint/data/Test/ ./data/Test/
```

Run in a `tmux`/`screen` session (or with `nohup ... &`) if the training
run will outlast your SSH session.
