## A Data-aware Probabilistic Client Sampling Scheme in Streaming Federated Learning

This is the code accompanying the  "A Data-aware Probabilistic Client Sampling Scheme in Streaming Federated Learning"

### Overview

---
In this paper, to address these challenges, we propose a Data aware Probabilistic Client Sampling scheme (DPCS) for selecting appropriate clients to participate in model training in each round of federated learning

### Depdendencies

---
Tested stable depdencises:

* python 3.12.3 (conda)

* PyTorch 2.3.0

* torchvision 0.18.0

### Experients over Image Classification Task:

---
The main result related to the image classification task i.e. VGG-9 on CIFAR-10 can be reproduced via running `./run.sh`. The following arguments to the `./main.py` file control the important parameters of the experiment.

| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `seed`                     | The random seed to use. |
| `dataset`      | Dataset to use. |
| `local_lr` | Learning rate that will be use. |
| `train_batch_size` | Batch size for the optimizers e.g. SGD or Adam. |
| `epochs` | Locally training epochs. |
| `local_epoch` | Local re-training epochs. |
| `client_num` | Number of  clients. |
| `rounds`    | Number of communication rounds to use . |

#### Sample command

```python
python server.py 
```


