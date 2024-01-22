from typing import Callable, List, Dict
import os


# NOTE (Sam): this torch processor appears to be 10% faster than standard multiprocessing - perhaps this is overkill
class Processor:
    def __init__(
        self,
        function_: Callable,
        source_paths: List[str],
        target_paths: List[
            str
        ],  # NOTE (Sam): this is target_folders in certain versions of the code since for example we want to save pitch at f0.pt and pitch mask as f0f.pt.  Have to think of a solution.
        recompute: bool = True,
    ):
        self.source_paths = source_paths
        self.function_ = function_
        self.target_paths = target_paths
        self.recompute = recompute

    def _get_data(self, source_path, target_path):
        # NOTE (Sam): we need caching to debug training issues in dev and for speed!
        # NOTE (Sam): won't catch issues with recomputation using different parameters but ssame name
        # TODO (Sam): add hashing
        if self.recompute or not os.path.exists(target_path):
            self.function_(source_path, target_path)
        else:
            pass

    def __getitem__(self, idx):
        try:
            self._get_data(
                source_path=self.source_paths[idx],
                target_path=self.target_paths[idx],
            )

        except Exception as e:
            print(f"Error while getting data: index = {idx}")
            print(e)
            raise
        return None

    def __len__(self):
        nfiles = len(self.source_paths)

        return nfiles
