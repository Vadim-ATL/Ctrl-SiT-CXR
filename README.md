## Training

To train the model, run the following command:

```bash
python train_rela_ctrl.py --data-path "C:\Users\HCI-4\Desktop\MIMIC_Counterfactual" --ckpt "C:\Users\HCI-4\Desktop\SiTXRay\SiT\results\best_ckpts\Base_003-SiT-XL-2-Linear-velocity-None\checkpoints\latest_checkpoint.pt" --path-type Linear --prediction velocity --global-batch-size 32 --num-workers 0 --log-every 10
