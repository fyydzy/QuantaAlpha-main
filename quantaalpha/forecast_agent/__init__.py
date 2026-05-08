import matplotlib

# Use a non-GUI backend to avoid tkinter/TkAgg teardown crashes in CLI/batch runs.
matplotlib.use("Agg")

from quantaalpha.forecast_agent.framework import (
    ForecastAgent,
    ForecastEvaluator,
    ForecastFeedback,
    ForecastStep,
    ForecastStrategy,
    ForecastSubjects,
    ForecastTask,
)
from quantaalpha.forecast_agent.timesfm_agent import (
    AutoTimesFmForecastAgent,
    GridTimesFmStrategy,
    TimesFmEvaluator,
    TimesFmHyperParams,
)
from quantaalpha.forecast_agent.sarimax_agent import (
    AutoSarimaxForecastAgent,
    FixedSarimaxStrategy,
    SarimaxEvaluator,
    SarimaxHyperParams,
)
from quantaalpha.forecast_agent.lasso_agent import (
    AutoLassoForecastAgent,
    FixedLassoStrategy,
    LassoEvaluator,
    LassoHyperParams,
)
from quantaalpha.forecast_agent.elasticnet_agent import (
    AutoElasticnetForecastAgent,
    ElasticnetEvaluator,
    ElasticnetHyperParams,
    FixedElasticnetStrategy,
)
from quantaalpha.forecast_agent.lstm_agent import (
    AutoLstmForecastAgent,
    FixedLstmStrategy,
    LstmEvaluator,
    LstmHyperParams,
)
from quantaalpha.forecast_agent.ridge_agent import (
    AutoRidgeForecastAgent,
    FixedRidgeStrategy,
    RidgeEvaluator,
    RidgeHyperParams,
)
from quantaalpha.forecast_agent.random_forest_agent import (
    AutoRandomForestForecastAgent,
    FixedRandomForestStrategy,
    RandomForestEvaluator,
    RandomForestHyperParams,
)
from quantaalpha.forecast_agent.xgboost_agent import (
    AutoXgboostForecastAgent,
    RandomGridXgboostStrategy,
    XgboostEvaluator,
    XgboostHyperParams,
)
from quantaalpha.forecast_agent.lightgbm_agent import (
    AutoLightgbmForecastAgent,
    RandomGridLightgbmStrategy,
    LightgbmEvaluator,
    LightgbmHyperParams,
)
from quantaalpha.forecast_agent.catboost_agent import (
    AutoCatboostForecastAgent,
    RandomGridCatboostStrategy,
    CatboostEvaluator,
    CatboostHyperParams,
)

__all__ = [
    "ForecastAgent",
    "ForecastEvaluator",
    "ForecastFeedback",
    "ForecastStep",
    "ForecastStrategy",
    "ForecastSubjects",
    "ForecastTask",
    "TimesFmHyperParams",
    "TimesFmEvaluator",
    "GridTimesFmStrategy",
    "AutoTimesFmForecastAgent",
    "SarimaxHyperParams",
    "SarimaxEvaluator",
    "FixedSarimaxStrategy",
    "AutoSarimaxForecastAgent",
    "LassoHyperParams",
    "LassoEvaluator",
    "FixedLassoStrategy",
    "AutoLassoForecastAgent",
    "ElasticnetHyperParams",
    "ElasticnetEvaluator",
    "FixedElasticnetStrategy",
    "AutoElasticnetForecastAgent",
    "LstmHyperParams",
    "LstmEvaluator",
    "FixedLstmStrategy",
    "AutoLstmForecastAgent",
    "RidgeHyperParams",
    "RidgeEvaluator",
    "FixedRidgeStrategy",
    "AutoRidgeForecastAgent",
    "RandomForestHyperParams",
    "RandomForestEvaluator",
    "FixedRandomForestStrategy",
    "AutoRandomForestForecastAgent",
    "XgboostHyperParams",
    "XgboostEvaluator",
    "RandomGridXgboostStrategy",
    "AutoXgboostForecastAgent",
    "LightgbmHyperParams",
    "LightgbmEvaluator",
    "RandomGridLightgbmStrategy",
    "AutoLightgbmForecastAgent",
    "CatboostHyperParams",
    "CatboostEvaluator",
    "RandomGridCatboostStrategy",
    "AutoCatboostForecastAgent",
]
