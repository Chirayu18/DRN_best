
folder=outfolder3
DRN/The_DRN_for_HGCAL/train $folder AToGG_pickles_testsamplev1 --nosemi --idx_name all --target trueE --in_layers 3 --mp_layers 2 --out_layers 2  --agg_layers 2 --valid_batch_size 250 --lr_sched Const --max_lr 0.0001 --pool mean --hidden_dim 64 --n_epochs 100 --train_batch_size 1
#DRN/The_DRN_for_HGCAL/train $folder AToGG_pickles_flatmass_tifr --nosemi --idx_name all --target trueE --in_layers 3 --mp_layers 2 --out_layers 2  --agg_layers 2 --valid_batch_size 250 --lr_sched Const --max_lr 0.0001 --hidden_dim 64 --predict_only
