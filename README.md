## Cloning the repository

To clone this repository with all its submodules:
```bash
git clone --recurse-submodules git@github.com:lexihaberl/DOPE.git
```

If you have already cloned the repository without the submodules, you can initialize and update them with:
```bash
git submodule update --init --recursive
```

## Downloading model weights
### yolov8 and gdrnpp
You can download the pretrained model weights with the following command:
```bash
bash scripts/download_ycbv_gdrnpp_weights.sh
```

## Building and running the containers
### Grounding DINO
```bash
cd ~/HSR_workspace/DOPE/ && docker-compose -f docker_compose/gsam2.yml up --build
```
```bash
cd ~/HSR_workspace/DOPE/ && docker-compose -f docker_compose/gsam2.yml up
```
