ECHO Train starting!

python main.py train ^
    --style_image_path="C:\Users\samet\Projects\fast_style_transfer\images\style\femme_nue_assise.jpg" ^
    --content_image_path="C:\Users\samet\Projects\fast_style_transfer\images\content\zurich.jpg" ^
    --train_dataset_path="D:\Datasets\COCO2014" ^
    --checkpoint_dir="C:\Users\samet\Projects\fast_style_transfer\weights\checkpoints" ^
    --weights_dir="C:\Users\samet\Projects\fast_style_transfer\weights" ^
    --sample_dir="C:\Users\samet\Projects\fast_style_transfer\images\samples" ^
    --epochs=2 ^
    --content_weight=1e-3 ^
    --style_weight=1.0 ^
    --sample_interval=1000

PAUSE