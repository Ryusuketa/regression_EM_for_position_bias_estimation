from src.model import RegressionBasedEM
from src.trainer import Trainer
import numpy as np
from typing import Dict, Optional, Any


def get_model(clicks: np.array,
              query_document_features: np.array,
              item_exposure: np.array,
              gbdt_params: Optional[Dict[str, Any]] = None) -> RegressionBasedEM:
    position, query, document = clicks.shape
    params = dict(position=position,
                  query=query,
                  document=document,
                  clicks=clicks,
                  item_exposure=item_exposure,
                  query_document_features=query_document_features)
    if gbdt_params:
        params['gbdt_params'] = gbdt_params
    return RegressionBasedEM(**params)


def get_trainer(model: RegressionBasedEM, epoch: int = 100):
    return Trainer(model, epoch)
