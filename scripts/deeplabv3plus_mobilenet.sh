python main.py --model deeplabv3plus_mobilenet --gpu_id 0,1 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 
python main.py --model deeplabv3plus_mobilenet --gpu_id 0,1 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --test_only --save_val_results
