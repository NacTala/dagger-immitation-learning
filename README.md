# FetchReach Imitation Learning and RL


Repository for experiments on `FetchReach-v4` with `gymnasium-robotics`.


```text
.

├── scripts/
│   ├── collect_data.py
│   ├── plot_metrics.py
│   ├── train_bc.py
│   ├── train_dagger.py
│   └── train_sac.py
├── src/
│   └── fetchreach_il/
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── experts.py
│       ├── models.py
│       ├── plotting.py
│       └── training.py
├── bc.py
├── collect_data.py
├── dagger.py
├── rl_classique.py
└── script.sh
```


## What the repo does

- manual demonstration collection for FetchReach
- behavioral cloning from collected data
- DAgger with a human, PID, or SAC expert
- standard SAC training
- metric plotting from CSV files

## Environment

Create a virtual environment and install the runtime dependencies using the requirements.txt file:



## Entry scripts

Collect manual demonstrations:

```bash
python scripts/collect_data.py
```

Train behavioral cloning:

```bash
python scripts/train_bc.py
```

Train DAgger with the default PID expert:

```bash
python scripts/train_dagger.py
```

Select another expert:

```bash
python scripts/train_dagger.py --expert human
python scripts/train_dagger.py --expert pid
python scripts/train_dagger.py --expert sac --expert-model-path artifacts/models/sac_fetchreach_100000steps.zip
```

Train SAC over a schedule of timesteps:

```bash
python scripts/train_sac.py
```

Generate plots from CSV metrics:

```bash
python scripts/plot_metrics.py
```

## Common outputs

- demonstration data is stored in `data_collect/fetchreach_manual_dataset.pkl`
- training metrics are written to `artifacts/metrics/`
- trained models are saved in `artifacts/models/`
- plots are written to `artifacts/plots/`

## Useful options

- `scripts/train_bc.py`: `--dataset-path`, `--epochs`, `--metrics-path`, `--model-path`
- `scripts/train_dagger.py`: `--expert`, `--expert-model-path`, `--iterations`, `--bc-epochs`, `--finetune-epochs`, `--metrics-path`, `--model-path`
- `scripts/train_sac.py`: `--timesteps`, `--metrics-path`, `--model-dir`, `--eval-episodes`
- `scripts/plot_metrics.py`: positional CSV paths or directories, plus `--output-dir`

## Notes

- The human expert mode requires an active graphical session.
- SAC models are saved by Stable-Baselines3 as `.zip` archives.

## Academic Context

This repository was developed as part of an academic project for the **Master MSR** at **Sorbonne Université**, within the **Social Robotics** course taught by **Prof. Mohamed Chetouani**.

The project was carried out by:
- Nacim TALAOUBRID
- Romane COUEDEL
- Oualid BOUDEMAGH