import numpy as np
import compress_fasttext
import joblib
from catboost import CatBoostClassifier


class Classifiter():
    def __init__(self, path_catboost: str, path_fasttext_compressed: str, path_pipeline_meta: str):
        self.model = CatBoostClassifier()
        self.model.load_model(path_catboost)
        self.embedding_dim = 300

        self.pipeline_meta = joblib.load(path_pipeline_meta)
        self.ft_model = (
            compress_fasttext.models.CompressedFastTextKeyedVectors.load(path_fasttext_compressed)
        )
    def vectorize(self, text: str):
        words = [w for w in str(text).lower().split() if w.isalpha()]
        if not words:
            return np.zeros(self.embedding_dim)
        vectors = [self.ft_model[w] for w in words]
        return np.mean(vectors, axis=0)

    def predict(self, employment_type: str = '', contract: str = '', salary: float = np.nan, vectors: np.array = None):
        emp = str(employment_type) if employment_type else ""
        cntr = str(contract) if contract else ""
        sal = float(salary) if (salary is not None and not np.isnan(salary)) else np.nan
    
       
        if vectors is None:
            row = [sal, emp, cntr] + [0.0] * self.embedding_dim
        else:
            row = [sal, emp, cntr] + vectors.tolist()
        return float(self.model.predict_proba([row])[0, 1])

    