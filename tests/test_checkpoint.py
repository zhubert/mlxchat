"""
Tests for checkpoint saving and loading.
"""
import os
import tempfile
import shutil

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxchat.gpt import GPT, GPTConfig
from mlxchat.muon import Muon
from mlxchat.checkpoint_manager import (
    save_checkpoint,
    load_checkpoint,
    find_last_step,
    flatten_dict,
    unflatten_dict,
)


def test_flatten_unflatten_dict():
    """Test dictionary flattening and unflattening."""
    nested = {
        "a": {"b": mx.array([1, 2, 3]), "c": mx.array([4, 5])},
        "d": mx.array([6, 7, 8, 9]),
    }

    # Flatten
    flat = dict(flatten_dict(nested))
    assert "a.b" in flat
    assert "a.c" in flat
    assert "d" in flat

    # Unflatten
    unflat = unflatten_dict(flat)
    assert "a" in unflat
    assert "b" in unflat["a"]
    assert "c" in unflat["a"]
    assert "d" in unflat

    # Check values match
    assert mx.array_equal(unflat["a"]["b"], nested["a"]["b"])
    assert mx.array_equal(unflat["a"]["c"], nested["a"]["c"])
    assert mx.array_equal(unflat["d"], nested["d"])


def test_save_and_load_checkpoint():
    """Test saving and loading a checkpoint."""
    # Create a small model
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
    )
    model = GPT(config)
    model.init_weights()

    # Create optimizers
    adam_opt = optim.Adam(learning_rate=0.001)
    muon_opt = Muon(learning_rate=0.01, momentum=0.95, nesterov=True)

    # Create some dummy optimizer states by doing a forward pass
    inputs = mx.random.randint(0, 1000, (2, 128))
    targets = mx.random.randint(0, 1000, (2, 128))

    def loss_fn(model, inputs, targets):
        return model(inputs, targets=targets)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, inputs, targets)

    # Split gradients for optimizers
    adam_grads = {}
    if "wte" in grads:
        adam_grads["wte"] = grads["wte"]
    if "lm_head" in grads:
        adam_grads["lm_head"] = grads["lm_head"]

    muon_grads = {}
    if "h" in grads:
        muon_grads["h"] = grads["h"]

    # Update optimizers to create state
    if adam_grads:
        adam_params = {}
        if "wte" in adam_grads:
            adam_params["wte"] = model.wte
        if "lm_head" in adam_grads:
            adam_params["lm_head"] = model.lm_head
        adam_opt.update(adam_params, adam_grads)

    if muon_grads:
        muon_params = {"h": model.h}
        muon_opt.update(muon_params, muon_grads)

    mx.eval(model.parameters(), adam_opt.state, muon_opt.state)

    # Save checkpoint
    temp_dir = tempfile.mkdtemp()
    try:
        step = 100
        meta_data = {
            "step": step,
            "loss": loss.item(),
            "model_config": {
                "sequence_len": config.sequence_len,
                "vocab_size": config.vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_kv_head": config.n_kv_head,
                "n_embd": config.n_embd,
            },
        }

        optimizer_data = {
            "adam": adam_opt.state,
            "muon": muon_opt.state,
        }

        save_checkpoint(temp_dir, step, model, optimizer_data, meta_data)

        # Verify files were created
        assert os.path.exists(os.path.join(temp_dir, f"model_{step:06d}.npz"))
        assert os.path.exists(os.path.join(temp_dir, f"optim_{step:06d}.npz"))
        assert os.path.exists(os.path.join(temp_dir, f"meta_{step:06d}.json"))

        # Load checkpoint
        model_data, loaded_optim_data, loaded_meta_data = load_checkpoint(
            temp_dir, step, load_optimizer=True
        )

        # Check metadata
        assert loaded_meta_data["step"] == step
        assert "model_config" in loaded_meta_data

        # Check model data
        assert "wte" in model_data
        assert "lm_head" in model_data
        assert "h" in model_data

        # Check optimizer data
        assert loaded_optim_data is not None
        assert "adam" in loaded_optim_data
        assert "muon" in loaded_optim_data

        # Create a new model and load the parameters
        new_model = GPT(config)
        new_model.init_weights()
        new_model.update(model_data)

        # Check that parameters match
        from mlx.utils import tree_flatten
        orig_params = tree_flatten(model.parameters())
        new_params = tree_flatten(new_model.parameters())

        assert len(orig_params) == len(new_params)
        for (orig_path, orig_value), (new_path, new_value) in zip(orig_params, new_params):
            assert orig_path == new_path
            assert mx.allclose(orig_value, new_value, atol=1e-6)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_find_last_step():
    """Test finding the last checkpoint step."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create some dummy checkpoint files
        for step in [10, 50, 100, 150]:
            path = os.path.join(temp_dir, f"model_{step:06d}.npz")
            mx.savez(path, dummy=mx.array([1, 2, 3]))

        # Find last step
        last_step = find_last_step(temp_dir)
        assert last_step == 150

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_save_checkpoint_without_optimizer():
    """Test saving a checkpoint without optimizer state."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
    )
    model = GPT(config)
    model.init_weights()

    temp_dir = tempfile.mkdtemp()
    try:
        step = 50
        meta_data = {
            "step": step,
            "model_config": {
                "sequence_len": config.sequence_len,
                "vocab_size": config.vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_kv_head": config.n_kv_head,
                "n_embd": config.n_embd,
            },
        }

        # Save without optimizer data
        save_checkpoint(temp_dir, step, model, None, meta_data)

        # Verify files
        assert os.path.exists(os.path.join(temp_dir, f"model_{step:06d}.npz"))
        assert not os.path.exists(os.path.join(temp_dir, f"optim_{step:06d}.npz"))
        assert os.path.exists(os.path.join(temp_dir, f"meta_{step:06d}.json"))

        # Load checkpoint
        model_data, optim_data, loaded_meta = load_checkpoint(
            temp_dir, step, load_optimizer=False
        )

        assert model_data is not None
        assert optim_data is None
        assert loaded_meta["step"] == step

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_flatten_unflatten_dict()
    print("✓ test_flatten_unflatten_dict passed")

    test_save_and_load_checkpoint()
    print("✓ test_save_and_load_checkpoint passed")

    test_find_last_step()
    print("✓ test_find_last_step passed")

    test_save_checkpoint_without_optimizer()
    print("✓ test_save_checkpoint_without_optimizer passed")

    print("\n✅ All checkpoint tests passed!")
