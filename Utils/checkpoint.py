# utils/checkpoint.py
import io, tarfile, torch
from collections import OrderedDict

def _strip_module_prefix(state_dict):
    # Handle DataParallel checkpoints with "module." prefixes
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    return new_sd

def load_state_dict_flex(obj, state_dict_key_candidates=("state_dict", "model", "net", "backbone"), strict=False):
    """
    Load a checkpoint dict into `obj`, trying several common keys and stripping 'module.' prefixes.
    """
    sd = None
    for k in state_dict_key_candidates:
        if k in obj:
            sd = obj[k]
            break
    if sd is None:
        # maybe the checkpoint itself is the state_dict
        sd = obj
    sd = _strip_module_prefix(sd)
    missing, unexpected = obj_target.load_state_dict(sd, strict=strict)  # obj_target defined by caller
    return missing, unexpected

def torch_load_any(path, map_location="cpu"):
    """
    Load either a raw PyTorch checkpoint (.pth/.pt/.pth.tar) or a real .tar containing such a file.
    Returns the loaded Python object (usually a dict).
    """
    # If it's really a tar archive, extract the first *.pth/*.pt inside to memory and torch.load it
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tf:
            # choose the first plausible weight file or the largest member
            members = [m for m in tf.getmembers() if m.isfile()]
            # heuristic: prefer .pth/.pt, otherwise pick largest file
            prefer = [m for m in members if m.name.endswith((".pth", ".pt", ".pth.tar", ".ckpt"))]
            member = prefer[0] if prefer else max(members, key=lambda m: m.size)
            f = tf.extractfile(member)
            byts = f.read()
            buf = io.BytesIO(byts)
            return torch.load(buf, map_location=map_location)
    else:
        return torch.load(path, map_location=map_location)
