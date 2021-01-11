from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


class AdversarialValidator:
    def __init__(self, cols_to_drop=[], scoring=["roc_auc"]):
        self.cols_to_drop = cols_to_drop
        self.scoring = scoring
        self.cv = None
        self.model = RandomForestClassifier()
        self.results = None
        self.adv_X = None
        self.adv_y = None

    def validate(self, train, test):
        adv_train = train.assign(test=lambda x: 0)
        adv_test = test.assign(test=lambda x: 1)
        adv = pd.concat([adv_train, adv_test])
        adv_y = adv["test"]
        adv_X = adv.drop(self.cols_to_drop + ["test"], axis=1)
        adv_cv = cross_validate(
            estimator=self.model,
            X=adv_X,
            y=adv_y,
            scoring=self.scoring,
            return_estimator=True,
        )
        self.results = adv_cv
        self.adv_X = adv_X
        self.adv_y = adv_y
        return adv_cv["test_roc_auc"].mean()

    def feature_importances_(self):
        if self.results is None:
            raise Exception(
                "validate needs to be called before feature importances can be calculated."
            )

        feature_imps = list(
            map(lambda x: x.feature_importances_, self.results["estimator"])
        )
        avg_feature_imp = np.vstack(feature_imps).mean(axis=0)

        return pd.Series(avg_feature_imp, index=self.adv_X.columns).sort_values(
            ascending=False
        )
