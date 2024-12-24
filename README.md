## A Data-aware Probabilistic Client Sampling Scheme in Streaming Federated Learning

This is the code accompanying the  "A Data-aware Probabilistic Client Sampling Scheme in Streaming Federated Learning"

### Overview

---
In this paper, we propose a Data aware Probabilistic Client Sampling scheme (DPCS) for selecting appropriate clients to participate in model training in each round of federated learning

### Depdendencies

---
Tested stable depdencies:

* python 3.12.3 (conda)

* PyTorch 2.3.0

* torchvision 0.18.0

### Argument
---

| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `seed`                     | The random seed to use. |
| `dataset`      | Dataset to use. |
| `local_lr` | Learning rate that will be use. |
| `train_batch_size` | Batch size for the optimizers e.g. SGD or Adam. |
| `local_epoch` | Local re-training epochs. |
| `client_num` | Number of  clients. |
| `rounds`    | Number of communication rounds to use . |

#### Sample command

```python
python server.py 
```

### Citation

---
If you find this project helpful, please consider to cite the following paper:
```bibtex
@inproceedings{chaosong2024dataaware,
    title = {A Data-aware Probabilistic Client Sampling Scheme in Streaming Federated Learning},
    author = {Chao Song, Jianfeng Huang, Jie Wu and Li Lu},
    year =  {2024},
    booktitle = { {IEEE} Global Communications Conference ({GLOBECOM})},
}
```
