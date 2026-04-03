import uuid
import numpy as np
from zoology.config import TrainConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.config import ModelConfig, ModuleConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "o2b-deltanet-mqar-" + sweep_id

VOCAB_SIZE = 8_192

# 1. Data configuration
train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]

input_seq_len = max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(batch_size, batch_size // 8),
    cache_dir="./cache"
)

# 2. Model configurations
models = []

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# Short convolution mixer (used before the main sequence mixer)
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)

MIXERS = {
    "attention": "zoology.mixers.attention.MHA",
    "deltanet": "zoology.mixers.ogd.deltanet.DeltaNetLayer",
    "o2b_deltanet": "zoology.mixers.ogd.o2b_deltanet.O2BDeltaNetLayer",
}


def add_model(models, model_name, conv_mixer, input_seq_len, model_factory_kwargs,
               num_layers=2, model_kwargs=None):
    """Add model configurations to the models list."""
    if model_kwargs is None:
        model_kwargs = {}

    for d_model in [64]:
        seq_mixer = dict(
            name=MIXERS[model_name],
            kwargs={**model_kwargs}
        )
        mixers = [conv_mixer, seq_mixer] if conv_mixer is not None else [seq_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name=f"{model_name}-{'-'.join([f'{k}={v}' for k, v in model_kwargs.items()])}-d{d_model}",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# Add models with different configurations
for eta in [0.01, 0.1, 1]:
    # O2B DeltaNet default
    models = add_model(
        models, "o2b_deltanet", conv_mixer, input_seq_len, model_factory_kwargs,
        model_kwargs={
            "eta": eta,
            "use_qk_activation": True,
            "sync_kv_scale": False,
            "ogd_mode": "deltanet",
            "use_RoPE": False,
        }
    )

    # use RoPE
    models = add_model(
        models, "o2b_deltanet", conv_mixer, input_seq_len, model_factory_kwargs,
        model_kwargs={
            "eta": eta,
            "use_qk_activation": True,
            "sync_kv_scale": False,
            "ogd_mode": "deltanet",
            "use_RoPE": False,
        }
    )

# 3. Create train configs
configs = []
for model in models:
    for lr in np.logspace(-3, -3, 1):  # Single learning rate for testing
        run_id = f"{model.name}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"./{run_id}",
            collect_predictions=True,
            # Optional: enable WandB logging
            # logger=LoggerConfig(
            #     project_name="zoology_o2b_deltanet",
            #     entity="your-entity",
            # ),
        )
        configs.append(config)
