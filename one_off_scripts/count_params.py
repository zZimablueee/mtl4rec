import sys, os
sys.path.insert(0, "/root/STEM/fuxictr")
os.chdir("/root/mtl4rec")
import torch, glob

for name, pattern in [("SharedBottom", "checkpoints/SharedBottom*"), ("MMoE", "checkpoints/MMoE*"), ("PLE", "checkpoints/PLE*")]:
    ckpts = glob.glob(pattern + "/*.ckpt")
    if ckpts:
        try:
            ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                total = sum(v.numel() for v in ckpt.values() if isinstance(v, torch.Tensor))
            else:
                total = sum(p.numel() for p in ckpt.parameters())
            print(f"{name}: {total:,} params")
        except Exception as e:
            print(f"{name}: error {e}")
    else:
        print(f"{name}: no checkpoint")
