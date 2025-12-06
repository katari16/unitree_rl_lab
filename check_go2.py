from pxr import Usd, UsdGeom

usd_path = "/home/ubuntu/unitree_model/Go2/usd/go2.usd"
stage = Usd.Stage.Open(usd_path)

print("=== Go2 USD Contents ===")
print(f"Valid: {stage is not None}")

if stage:
    print("\nPrims in USD:")
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()} - Type: {prim.GetTypeName()}")
        
    print(f"\nTotal prims: {len(list(stage.Traverse()))}")
else:
    print("ERROR: USD file is invalid!")
