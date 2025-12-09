# Benchmarking


## Installation and running

```bash
# for torch
conda create -n torch python=3.11 -y
conda activate torch
bash scripts/install.sh
python test_gru_torch.py

# for jax
conda create -n jax python=3.12 -y
conda activate jax
bash scripts/install_jax.sh
python test_gru_jax.py
```

## Results

* Pytorch:
```
{'forward_compile_with_model_compile/sum': 3.206577326,
 'forward_compile_with_model_compile/min': 0.319381331,
 'forward_compile_with_model_compile/max': 0.322439361,
 'forward_compile_with_model_compile/avg': 0.3206577326,
 'forward_compile_with_model_compile/frac': 0.134429534646948,
 'forward_compile_with_model_compile/count': 10,
 'forward_eager/sum': 3.634695664,
 'forward_eager/min': 0.230637707,
 'forward_eager/max': 0.753163803,
 'forward_eager/avg': 0.3634695664,
 'forward_eager/frac': 0.15237756555345883,
 'forward_eager/count': 10,
 'forward_compile/sum': 2.480761282,
 'forward_compile/min': 0.017063785,
 'forward_compile/max': 0.324973694,
 'forward_compile/avg': 0.2480761282,
 'forward_compile/frac': 0.10400110485575927,
 'forward_compile/count': 10,
 'summary': '- 15% forward_eager\n- 13% forward_compile_with_model_compile\n- 10% forward_compile'}
```

* JAX:
```
{'jax_forward_compile/sum': 1.189535473,
 'jax_forward_compile/min': 0.000523988,
 'jax_forward_compile/max': 0.232672957,
 'jax_forward_compile/avg': 0.1189535473,
 'jax_forward_compile/frac': 0.18616297545885852,
 'jax_forward_compile/count': 10,
 'summary': '- 19% jax_forward_compile'}
```


## Analysis

For a model of about 411 millions parameters, JAX outperforms PyTorch about from 1.5x to 3x faster.

