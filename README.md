# Temporal Action Location

## There are still many problems to be solved, it's not a mature version!!!

Faster rcnn architecture for Action Location


**Install**

```
1. git clone git@github.com:peterzpy/TAL.git
2. cd TAL
3. mkdir Pretrained
4. cd utils && python3 setyp.py nms/setup.py build_ext --inplace
```

> If you want to use pretrained resnet model [Resnet3D](https://github.com/kenshohara/3D-ResNets-PyTorch>)
>
> **Test**
>
> ```
> 1. cd model
> 2. python3 test_resnet_learn_nms_Net.py --image_path Your/Image/path --annotation_path Your/Annotation/Path --checkpoint_path Your/Checkpoint/Path
> ```
>
> **Train With Preprocessing**
>
> ``` 
> 1. cd model
> 2. python3 train_resnet_learn_nms_Net.py --use_resnet_pth 'True' --pth_path "../Pretrained/resnet-101-kinetics-ucf101_split1.pth" --preoprocess 'False' --image_path Your/Image/path --annotation_path Your/Annotation/Path --checkpoint_path Your/Checkpoint/Path --video_path Your/Video/path --video_annotation_path Your/VideoLabel/path --feature_preprocess True --feature_path Your/feature/path
> ```
>
> **Train Without Preprocessing**
>
> ``` 
> 1. cd model
> 2. python3 train_resnet_learn_nms_Net.py --image_path Your/Image/path --annotation_path Your/Annotation/Path --checkpoint_path Your/Checkpoint/Path --feature_path Your/feature/path
> ```

