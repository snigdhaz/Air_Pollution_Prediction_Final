import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def remove_unwanted_columns(df):
    """
    Removes specific unwanted columns if they exist in the DataFrame.
    """
    unwanted = [
        'Unnamed: 0', 'Month', 'day_of_week', 
        'Day_of_week', 'pm25_lag1', 'pm25_lag7', 'to_date',
    ]
    
    # Only drop columns that are actually present in the dataframe
    cols_to_drop = [col for col in unwanted if col in df.columns]
    
    return df.drop(columns=cols_to_drop)



def fix_temporal_structure(df, datetime_col='from_date', freq='h'):
    """
    Ensures the dataframe has a continuous datetime index with no gaps.
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Remove duplicates by taking the mean of entries with the same timestamp
    df = df.groupby(datetime_col).mean(numeric_only=True).reset_index()

    # Set index and fill missing hours
    df = df.set_index(datetime_col)
    df = df.asfreq(freq)
    
    return df



def apply_physical_bounds(df):
    """
    Clips sensor data to realistic physical limits.
    """
    # Pollutants & Wind Speed: Cannot be negative
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    
    # Humidity: Cannot exceed 100%
    if 'humidity' in df.columns:
        df['humidity'] = df['humidity'].clip(upper=100)
        
    return df



def handle_missing_values(df, max_gap=3):
    """
    Fills gaps using linear interpolation. 
    'max_gap' limits interpolation so we don't 'guess' too much data.
    """
    # Interpolate small gaps linearly
    df = df.interpolate(method='linear', limit=max_gap)
    
    # For larger gaps, use a backfill/forward fill to catch edges
    df = df.ffill().bfill()
    
    return df

def load_and_preprocess_split(train_path, val_path, test_path):
    """
    Loads three CSVs, cleans them using the predefined pipeline, 
    and returns scaled DataFrames + the fitted scaler.
    """
    
    # 1. Load raw data
    datasets = {
        'train': pd.read_csv(train_path),
        'val': pd.read_csv(val_path),
        'test': pd.read_csv(test_path)
    }
    
    processed_dfs = {}
    
    for name, df in datasets.items():
        # Apply the pipeline we built earlier
        # Note: 'city' and 'to_date' are usually dropped for modeling
        df_clean = (df.pipe(remove_unwanted_columns)  # 1. Strip junk first
                      .pipe(fix_temporal_structure)    # 2. Fix time index
                      .pipe(apply_physical_bounds)    # 3. Clip sensor errors
                      .pipe(handle_missing_values))    # 4. Fill NaNs
        
        if 'time_idx' in df_clean.columns:
            df_clean['time_idx'] = df_clean['time_idx'].astype(int)
        processed_dfs[name] = df_clean

    # 2. Feature Scaling (Standardization)
    # We fit ONLY on train to prevent data leakage
        
    return processed_dfs['train'], processed_dfs['val'], processed_dfs['test']

train_file = "/share/ftrscape/lmiddha/hw/data/Train_data.csv"
val_file = "/share/ftrscape/lmiddha/hw/data/Validation_data.csv"
test_file = "/share/ftrscape/lmiddha/hw/data/Test_data.csv"

train_data, val_data, test_data = load_and_preprocess_split(train_file, val_file, test_file)


feature_cols=[ 'pm25', 'pm10', 'no', 'nh3', 'no2', 'nox', 'so2', 'co', 'ozone', 'bp',
       'wind_speed', 'air_temp', 'humidity', 'rainfall']
scaler = MinMaxScaler()
scaler.fit(train_data[feature_cols])
train_data[feature_cols] = scaler.transform(train_data[feature_cols])
val_data[feature_cols] = scaler.transform(val_data[feature_cols])
test_data[feature_cols] = scaler.transform(test_data[feature_cols])

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

train_data["city"] = "Delhi"
test_data["city"] = "Delhi"
val_data["city"] = "Delhi"

print(train_data.head())
print(test_data.head())
print(val_data.head())

import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
from pytorch_forecasting import Baseline
from pytorch_forecasting.metrics import MAE, RMSE
from lightning.pytorch.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_forecasting.data.encoders import TorchNormalizer

max_prediction_length = 24
max_encoder_length = 360

training = TimeSeriesDataSet(
    train_data,
    time_idx="time_idx",
    target="ozone",
    group_ids=["city"],
    min_encoder_length=168,
    max_encoder_length=max_encoder_length,
    min_prediction_length=12,
    max_prediction_length=max_prediction_length,
    static_categoricals=["city"],
    static_reals=[],
    time_varying_known_categoricals=[],
    variable_groups={},
    time_varying_known_reals=[
        "time_idx", "wind_speed", "air_temp", "humidity", "bp", "rainfall",
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "pm10", "pm25", "no", "nh3", "no2", "so2", "co", "ozone", "nox",
    ],
    # identity normalizer because we already scaled with MinMaxScaler
    target_normalizer=TorchNormalizer(method="identity", center=False),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

history_for_val = train_data.iloc[-max_encoder_length:]
val_combined = pd.concat([history_for_val, val_data]).reset_index(drop=True)

validation = TimeSeriesDataSet.from_dataset(
    training,
    val_combined,
    predict=False,
    stop_randomization=True
)

train_dataloader = training.to_dataloader(
    train=True, batch_size=128, num_workers=1
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=64, num_workers=1
)

class LossHistory(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses   = []
 
    def on_train_epoch_end(self, trainer, pl_module):
        # TFT logs "train_loss_epoch" after aggregation; fall back to "train_loss"
        loss = trainer.callback_metrics.get("train_loss_epoch") or \
               trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())
 
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())


early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor() 
tqdm_logger = TQDMProgressBar(refresh_rate =20)
logger = TensorBoardLogger("lightning_logs")
loss_history= LossHistory()


checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints-ozone-15-days",
    filename="tft-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    save_top_k=3,        # keep best 3 models
    mode="min",
    save_last=True       # also save last epoch
)

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback, tqdm_logger, checkpoint_callback, loss_history],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    output_size=1,
    hidden_continuous_size=32,
    loss=RMSE(),
    log_interval=10, 
    optimizer="adamw",
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

history = val_data.iloc[-max_encoder_length:].copy()
step_size   = max_prediction_length
total_steps = len(test_data)
targets     = ["ozone"]

all_preds   = []
all_actuals = []

target_idx = feature_cols.index("ozone")

for start_idx in range(0, total_steps, step_size):
    true_chunk = test_data.iloc[start_idx:start_idx + step_size]
    if len(true_chunk) == 0:
        break

    encoder_df = history.iloc[-max_encoder_length:].copy()
    window_df  = pd.concat([encoder_df, true_chunk]).reset_index(drop=True)

    global_time_offset        = int(encoder_df["time_idx"].iloc[0])
    window_df["time_idx"]     = range(global_time_offset, global_time_offset + len(window_df))
    window_df["time_idx"]     = window_df["time_idx"].astype(int)

    window_dataset = TimeSeriesDataSet.from_dataset(
        training,
        window_df,
        predict=True,
        stop_randomization=True
    )
    window_dataloader = window_dataset.to_dataloader(
        train=False, batch_size=64, num_workers=1
    )

    preds = tft.predict(
        window_dataloader,
        mode="prediction",
        trainer_kwargs=dict(accelerator="gpu")
    )

    chunk_len  = len(true_chunk)
    pred_chunk = preds.reshape(-1).cpu().numpy()[:chunk_len]   # (chunk_len,)

    dummy_pred = np.zeros((len(pred_chunk), len(feature_cols)))
    dummy_pred[:, target_idx] = pred_chunk
    pred_original = scaler.inverse_transform(dummy_pred)[:, target_idx]  # (chunk_len,)

    actual_chunk = true_chunk["ozone"].values                   # scaled [0,1]
    dummy_act    = np.zeros((chunk_len, len(feature_cols)))
    dummy_act[:, target_idx] = actual_chunk
    act_original = scaler.inverse_transform(dummy_act)[:, target_idx]    # (chunk_len,)

    all_preds.append(pred_original)
    all_actuals.append(act_original)

    # Update history with true observed values
    history = pd.concat([history, true_chunk])

from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
mae  = mean_absolute_error(all_actuals, all_preds)

print(f"\n Metrics over full 30-day test set (720 points) ")
print(f"Ozone  RMSE: {rmse:.4f} µg/m³  |  MAE: {mae:.4f} µg/m³")


import matplotlib.pyplot as plt


plt.figure(figsize=(15,5))
plt.plot(all_actuals, label="Actual Ozone", color="blue")
plt.plot(all_preds, label="Predicted Ozone", color="red", alpha=0.7)
plt.xlabel("Time Index")
plt.ylabel("Ozone")
plt.title("Ozone Predictions vs Actuals - 15 days window")
plt.legend()
plt.savefig("ozone_predictions_15_days.png", dpi=300, bbox_inches="tight")


import matplotlib.pyplot as plt

def plot_loss_history(loss_history):
    train_losses = loss_history.train_losses
    val_losses   = loss_history.val_losses

    # Align lengths in case val runs one epoch behind
    min_len = min(len(train_losses), len(val_losses))
    train_losses = train_losses[:min_len]
    val_losses   = val_losses[:min_len]

    epochs = range(1, min_len + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, val_losses,   label="Val Loss",   marker="o", markersize=3)

    # Mark best val loss
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val   = min(val_losses)
    plt.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best Val Epoch {best_epoch}")
    plt.scatter([best_epoch], [best_val], color="red", zorder=5)
    plt.annotate(f"{best_val:.4f}", xy=(best_epoch, best_val),
                 xytext=(best_epoch + 0.3, best_val), fontsize=9, color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (Scaled RMSE)")
    plt.title("Training vs Validation Loss - Ozone")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve_ozone_15_days.png", dpi=300)

plot_loss_history(loss_history)
