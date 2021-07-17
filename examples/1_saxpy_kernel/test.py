import torch, main

# compute
x, y = torch.randn([2, 100], device="cuda")
z = main.saxpy(x, y, 1.5)

# measure error
z_gt = x * 1.5 + y
error = (z - z_gt).abs().max().item()

print(f"max error = {error:.3e}")
