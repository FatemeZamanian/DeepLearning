# Face age regression
To use files:
- Clone repo

To run train file :
coppy kaggle.json file in main directory
```
python train.py --device cpu/cuda --dataset path of dataset folder
```
To get evaluate of network :
```
python test.py --device cpu/cuda --dataset path of dataset folder
```

To get inference :

- Download model from [Model](https://drive.google.com/file/d/12T6NcKvkremwbVNiYMiOlZtl5dbwbp-v/view?usp=sharing) and put them on root path

```
python inference.py --device cpu/cuda --kind camera/image  --image 'path of your image' 
```

