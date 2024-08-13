# fintune
python train_fintune.py -s data/bear/ -m output/bear_inpaint_finetune --port 7777 --start_checkpoint output/bear_incomplete/chkpnt30000.pth --iteration 31000 --is_finetune --save_iteration 30100 30200 30300 30400 30500 30600 30700 30800 30900

# train incomplete
python train.py -s data/bear/ -m output/bear_incomplete --port 9999 --mask_training
python train.py -s data/kitchen -m output/kitchen_incomplete --port 9999 --mask_training --images images_4 --sds_loss_weight 0


# train inpaint from scratch
python train.py -s data/bear/ -m output/bear_inpaint --mask_training --port 8888

# render image 
time python render.py -s data/bear/ -m output/bear_inpaint_finetune_1 --iteration 40000 --skip_mesh

# finetune only masked area with SDEdit result
python train_fintune.py -s data/bear_leftrefill/ -m output/bear_leftrefill_s0.25 --images leftrefill --start_checkpoint /home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete/chkpnt30000.pth --iteration 31200 --save_iterations 30150 30300 30600 30900 31200 --checkpoint_iteration 30150 30300 30600 30900 31200 --test_iterations 30150 30300 30600 30900 31200
python render.py -s data/bear_leftrefill/ -m output/bear_leftrefill_s0.25 --skip_mesh

# finetune only masked area with Lpips loss
python train_fintune.py -s data/bear_leftrefill/ -m output/bear_leftrefill_s0.99_lpips --images leftrefill --start_checkpoint /home_nfs/kkennethwu_nldap/2d-gaussian-splatting/output/bear_incomplete/chkpnt30000.pth --iteration 40000 --save_iterations 40000 --checkpoint_iteration 40000  --test_iterations 40000 --lambda_lpips 0.5