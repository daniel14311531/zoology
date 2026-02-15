import uuid
import numpy as np
from zoology.config import TrainConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.config import ModelConfig, ModuleConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "011026-original-mqar-repro-v1" + sweep_id

VOCAB_SIZE = 8_192

# 1. First we are going to create the data configuration

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

input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size // 8),
    cache_dir="./cache"
)

# 2. Next, we are going to collect all the different model configs we want to sweep
models = []

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE,
}

# define this conv outside of if/else block because it is used in multiple models
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)

MIXERS = {
    "attention":"zoology.mixers.attention.MHA",
    "deltanet": "zoology.mixers.ogd.deltanet.DeltaNetLayer",
    "omd_deltanet": "zoology.mixers.ogd.omd_deltanet.OmdDeltaNetLayer",
    "conceptual_deltanet": "zoology.mixers.ogd.conceptual_deltanet.ConceptualDeltaNetLayer",
}

def add_model(models, model_name, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        seq_mixer = dict(
            name=MIXERS[model_name],
            kwargs={
                # "l_max": input_seq_len,
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

models = add_model(models, "attention", conv_mixer, input_seq_len, model_factory_kwargs, model_kwargs={"dropout": 0.1, "num_heads": 2})

for eta in [0.01, 0.1, 1.0]:
    models = add_model(models, "deltanet", conv_mixer, input_seq_len, model_factory_kwargs, model_kwargs={"eta": eta, "use_qk_activation": True})
    models = add_model(models, "omd_deltanet", conv_mixer, input_seq_len, model_factory_kwargs, model_kwargs={"eta": eta, "use_qk_activation": True})
    models = add_model(models, "conceptual_deltanet", conv_mixer, input_seq_len, model_factory_kwargs, model_kwargs={"eta": eta, "use_qk_activation": True})
    models = add_model(models, "deltanet", conv_mixer, input_seq_len, model_factory_kwargs, model_kwargs={"eta": eta, "use_qk_activation": False})
    models = add_model(models, "omd_deltanet", conv_mixer, input_seq_len, model_factory_kwargs, model_kwargs={"eta": eta, "use_qk_activation": False})
    models = add_model(models, "conceptual_deltanet", conv_mixer, input_seq_len, model_factory_kwargs, model_kwargs={"eta": eta, "use_qk_activation": False})

# 3. Finally we'll create a train config for each
configs = []
for model in models:
    # for lr in np.logspace(-3, -1.5, 4):
    for lr in np.logspace(-3, -3, 1):
        run_id = f"{model.name}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            # logger=LoggerConfig(
            #     project_name="zoology_ogd",
            #     entity="efficient-attention-nju"
            # ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"./{run_id}",
            collect_predictions=True,
        )
        configs.append(config)