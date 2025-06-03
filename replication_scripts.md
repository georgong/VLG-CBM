Here are the list of scripts to reproduce some of the results in Tables 2 and 3:

CIFAR-10 training:
```bash
python train_cbm.py --backbone clip_RN50 --config configs/cifar10.json --annotation_dir annotations
```

CIFAR-100 training:
```bash
python train_cbm.py --backbone clip_RN50 --config configs/cifar100.json --annotation_dir annotations
```

CUB training:
```bash
# CLIP backbone:
python train_cbm.py --backbone clip_RN50 --config configs/cub.json --annotation_dir annotations

# Non-CLIP backbone:
python train_cbm.py --backbone resnet18_cub --config configs/cub.json --annotation_dir annotations
```

CIFAR-10 evaluation: 
```bash
python sparse_evaluation.py --load_path saved_models/cifar10/{replace with your directory name} --result_file result/cifar10/result.csv
```

CIFAR-100 evaluation:
```bash
python sparse_evaluation.py --load_path saved_models/cifar100/{replace with your directory name} --result_file result/cifar100/result.csv
```

CUB evaluation:
```bash
python sparse_evaluation.py --load_path saved_models/cub/{replace with your directory name} --result_file result/cub/result.csv
```

