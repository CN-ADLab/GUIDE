bash ./tools/dist_test.sh \
    projects/configs/guide.py \
    ckpt/guide_gs32.pth \
    1 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/guide/results.pkl