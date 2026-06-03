import sys
print("Python:", sys.version)
pkgs = ["torch", "torchvision", "datasets", "scipy", "tqdm", "matplotlib", "PIL", "numpy"]
for name in pkgs:
    try:
        mod = __import__(name if name != "PIL" else "PIL")
        ver = getattr(mod, "__version__", "?")
        print(f"  OK  {name}=={ver}")
    except ImportError:
        print(f"  MISSING  {name}")
