import numpy as np
import tqdm
from src.model import RegressionBasedEM


class Trainer:
    def __init__(self, model: RegressionBasedEM, epoch: int = 100) -> None:
        self._model = model
        self._epoch = epoch

    def train(self, relevance: np.array = None, features: np.array = None):
        t = tqdm.trange(self._epoch)
        t = tqdm.trange(self._epoch)
        t = tqdm.trange(self._epoch)
        if len(relevance):
            print('random squared error mean %f' % np.sqrt(np.mean((relevance - np.random.rand(*relevance.shape))**2)))
            print('random squared error median %f' % np.sqrt(np.median((relevance - np.random.rand(*relevance.shape))**2)))
            print('random squared error std %f' % np.sqrt(np.std((relevance - np.random.rand(*relevance.shape))**2)))
        for _ in t:
            self._model.do_e_step()
            self._model.do_m_step()
            if len(relevance):
                rel = self._model.predict_relevance(features)
                desc = ''
                desc += ("squared error - median %f, " % np.sqrt(np.median((relevance - rel)**2)))
                desc += ("squared error - mean %f, " % np.sqrt(np.mean((relevance - rel)**2)))
                desc += ("squared error - std %f, " % np.sqrt(np.std((relevance - rel)**2)))
                desc += ("mean %f, " % np.mean(rel))
                desc += ("std %f, " % np.std(rel))
                desc += ("max %f, " % np.max(rel))
                desc += ("min %f, " % np.min(rel))
                t.set_description(desc)

    def get_model(self):
        return self._model
