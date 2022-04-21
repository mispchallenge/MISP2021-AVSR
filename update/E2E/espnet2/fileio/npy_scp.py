import collections.abc
from pathlib import Path
from typing import Union
import torch
import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text


class NpyScpWriter:
    """Writer class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

        >>> writer = NpyScpWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = numpy_array
        >>> writer['bb'] = numpy_array

    """

    def __init__(self, outdir: Union[Path, str], scpfile: Union[Path, str]):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")

        self.data = {}

    def get_path(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        assert isinstance(value, np.ndarray), type(value)
        p = self.dir / f"{key}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), value)
        self.fscp.write(f"{key} {p}\n")

        # Store the file path
        self.data[key] = str(p)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


class NpyScpReader(collections.abc.Mapping):
    """Reader class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

        >>> reader = NpyScpReader('npy.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2column_text(fname)

    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        return np.load(p)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class VideoScpReader(collections.abc.Mapping):
    """Reader class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

        >>> reader = NpyScpReader('npy.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2column_text(fname)

    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        return video_load(key,self.data[key])
      

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()

class Video_nointerpolation_ScpReader(collections.abc.Mapping):
    """Reader class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

        >>> reader = NpyScpReader('npy.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2column_text(fname)
     

    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        return nointerpolation_video_load(self.data[key])
      

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()



def nointerpolation_video_load(value):
    #load file as nump
    if "npz" in value:
            output = np.load(value)["data"]
    elif ".pt" in value:
            output = torch.load(value).numpy()
    return output.astype(np.float32)

def video_load(uid,value):
    if "npz" in value:
            output = np.load(value)["data"]
    elif ".pt" in value:
            output = torch.load(value).numpy()

    if "sp0.8" in uid:
        output = output.repeat(5,axis=0)
    elif "sp1.3" in uid:
        output = output.repeat(3,axis=0)
    else:
        output = output.repeat(4,axis=0)
    return output.astype(np.float32)
            
       
        


