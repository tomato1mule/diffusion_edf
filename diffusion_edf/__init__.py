from typing import List, Tuple, Optional, Union
import torch

GNN_OUTPUT_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], torch.Tensor, torch.Tensor]
EXTRACTOR_INFO_TYPE = Tuple[torch.Tensor, torch.Tensor]
QUERY_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
EDF_INFO_TYPE = Tuple[EXTRACTOR_INFO_TYPE, GNN_OUTPUT_TYPE]