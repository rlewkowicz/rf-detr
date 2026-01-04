import torch
from rfdetr import RFDETRBase
import argparse
from rfdetr.util.get_param_dicts import get_param_dict

def check():
    print("Initializing model...")
    rf = RFDETRBase()
    model = rf.model.model # This is the nn.Module (Joiner)
    
    # Mock args
    args = argparse.Namespace(
        lr=1e-4,
        lr_encoder=1.5e-4,
        lr_vit_layer_decay=0.8,
        lr_component_decay=1.0,
        weight_decay=1e-4,
        out_feature_indexes=[-1],
        num_queries=300,
        group_detr=13
    )
    
    print("Getting param dicts...")
    try:
        param_dicts = get_param_dict(args, model)
        print(f"Successfully got {len(param_dicts)} param groups")
        
        # Check for overlaps by object identity
        all_params = []
        for i, group in enumerate(param_dicts):
            params = group['params']
            if isinstance(params, torch.Tensor):
                params = [params]
            for p in params:
                all_params.append((id(p), p))
        
        ids = {}
        duplicates = []
        for i, group in enumerate(param_dicts):
            params = group['params']
            if isinstance(params, torch.Tensor):
                params = [params]
            for p in params:
                pid = id(p)
                if pid in ids:
                    duplicates.append((pid, ids[pid], i))
                else:
                    ids[pid] = i
        
        if duplicates:
            print(f"ERROR: {len(duplicates)} duplicate parameters found in param_dicts!")
            # To see names, we need to find which names they correspond to in the model
            param_to_name = {p: n for n, p in model.named_parameters()}
            for pid, first_group_idx, second_group_idx in duplicates:
                # Find the parameter object by ID (slow but okay for debug)
                found_p = None
                for p in model.parameters():
                    if id(p) == pid:
                        found_p = p
                        break
                name = param_to_name.get(found_p, "Unknown")
                print(f"Duplicate found: {name} (ID {pid}) in groups {first_group_idx} and {second_group_idx}")
        else:
            print("No duplicates found by object identity in return value.")

    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check()
