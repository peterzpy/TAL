# TAL
Faster rcnn architecture for Action Location

**Install**

```
1. git clone git@github.com:peterzpy/TAL.git
2. cd TAL
3. mkdir Pretrained
```

> If you want to use pretrained resnet model [Resnet3D](<https://github.com/Tushar-N/pytorch-resnet3d>)
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
> 2. python3 train_resnet_learn_nms_Net.py --use_resnet_pth 'True' --pth_path "../Pretrained/i3d_r50_kinetics.pth" --preoprocess 'False' --image_path Your/Image/path --annotation_path Your/Annotation/Path --checkpoint_path Your/Checkpoint/Path --video_path Your/Video/path --video_annotation_path Your/VideoLabel/path
> ```
>
> **Train Without Preprocessing**
>
> ``` 
> 1. cd model
> 2. python3 train_resnet_learn_nms_Net.py --use_resnet_pth 'True' --pth_path "../Pretrained/i3d_r50_kinetics.pth" --image_path Your/Image/path --annotation_path Your/Annotation/Path --checkpoint_path Your/Checkpoint/Path
> ```
>
> **Train With fully net pretrained model (not only resnet part)**
>
> ``` 
> 1. cd model
> 2. python3 train_resnet_learn_nms_Net.py --image_path Your/Image/path --annotation_path Your/Annotation/Path --checkpoint_path Your/Checkpoint/Path
> ```

