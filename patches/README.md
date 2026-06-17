# YOLOP Local Patches

`YOLOP/` is kept out of this repository because it is a cloned upstream repo with large weights and generated outputs.

To reproduce the evaluation setup used in this project, clone or keep YOLOP at `./YOLOP`, then apply:

```powershell
cd YOLOP
git apply ..\patches\yolop_windows_eval.patch
```

The patch:

- points YOLOP to `../datasets/yolop_bdd100k`
- uses CPU-friendly test settings
- fixes `tools/test.py` checkpoint loading when `--weights` is parsed as a list
- fixes a PyTorch compatibility issue in `build_targets`
