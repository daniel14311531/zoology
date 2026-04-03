"""
GLA experiment configuration for Zoology.

This config tests the Gated Linear Attention (GLA) mixer on MQAR tasks.
"""

import uuid
import numpy as np
from zoology.config import TrainConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.config import ModelConfig, ModuleConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "gla-mqar-test" + sweep_id

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

# 2. Model configuration
models = []

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# Short convolution mixer (used in hybrid with GLA)
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)

MIXERS = {
    "gla_naive": "zoology.mixers.my_gla.gla_naive.GLANaive",
}


def add_model(models, model_name, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2, model_kwargs=None):
    """Add model configuration to the models list."""
    if model_kwargs is None:
        model_kwargs = {}
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]:
        seq_mixer = dict(
            name=MIXERS[model_name],
            kwargs={
                **model_kwargs
            }
        )
        mixers = [conv_mixer, seq_mixer] if conv_mixer is not None else [seq_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type=block_type,
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name=f"{model_name}-{'-'.join([f'{k}={v}' for k, v in model_kwargs.items()])}-d{d_model}",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# GLA configurations
# Test different chunk sizes
for chunk_size in [32, 64]:
    models = add_model(
        models, "gla_naive", conv_mixer, input_seq_len, model_factory_kwargs,
        model_kwargs={"chunk_size": chunk_size, "num_heads": 4, "gate_act": "sigmoid"}
    )

# Test different gate activations
for gate_act in ["sigmoid", "softmax"]:
    models = add_model(
        models, "gla_naive", conv_mixer, input_seq_len, model_factory_kwargs,
        model_kwargs={"chunk_size": 64, "num_heads": 4, "gate_act": gate_act}
    )

# Test different number of heads
for num_heads in [2, 4, 8]:
    models = add_model(
        models, "gla_naive", conv_mixer, input_seq_len, model_factory_kwargs,
        model_kwargs={"chunk_size": 64, "num_heads": num_heads, "gate_act": "sigmoid"}
    )

# 3. Create training configurations
configs = []
for model in models:
    for lr in np.logspace(-3, -3, 1):
        run_id = f"{model.name}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            # logger=LoggerConfig(
            #     project_name="zoology_gla",
            #     entity="your_entity"
            # ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"./{run_id}",
            collect_predictions=True,
        )
        configs.append(config)
