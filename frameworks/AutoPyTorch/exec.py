import logging
import math
import os
import tempfile as tmp
import warnings

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sklearn.metrics
# TODO: Add AutoNetEnsemble
from autoPyTorch import AutoNetEnsemble, AutoNetClassification, AutoNetRegression

from frameworks.shared.callee import call_run, result, Timer, touch

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** AutoPyTorch ****\n")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    is_classification = config.type == 'classification'

    # Mapping of benchmark metrics to autosklearn metrics
    metrics_mapping = dict(
        # acc=sklearn.metrics.accuracy,
        # auc=sklearn.metrics.roc_auc,
        # f1=sklearn.metrics.f1,
        # logloss=sklearn.metrics.log_loss,
        # mae=sklearn.metrics.mean_absolute_error,
        # mse=sklearn.metrics.mean_squared_error,
        # rmse=sklearn.metrics.mean_squared_error,  # autosklearn can optimize on mse, and we compute rmse independently on predictions
        # r2=sklearn.metrics.r2
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    log.info("Running AutoPyTorch with a maximum time of %ss on %s cores with %sMB.",
             config.max_runtime_seconds, config.cores, config.max_mem_size_mb)
    log.info("Environment: %s", os.environ)

    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc
    predictors_type = dataset.predictors_type
    log.debug("predictors_type=%s", predictors_type)
    # log.info("finite=%s", np.isfinite(X_train))

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    log.warning("Using meta-learned initialization, which might be bad (leakage).")
    estimator = AutoNetClassification if is_classification else AutoNetRegression
    # TODO: Commented out because autoPyTorch fails to load the correct file after training when using AutoNetEnsemble
    # autoPyTorch = AutoNetEnsemble(estimator,
    #     log_level='info',
    #     max_runtime=config.max_runtime_seconds,
    #     min_budget=config.max_runtime_seconds/40,
    #     max_budget=config.max_runtime_seconds/5
    # )
    autoPyTorch = estimator(
        log_level='info',
        max_runtime=config.max_runtime_seconds,
        min_budget=config.max_runtime_seconds / 40,
        max_budget=config.max_runtime_seconds / 5
    )

    with Timer() as training:
        autoPyTorch.fit(X_train, y_train, validation_split=0.33)

    # Convert output to strings for classification
    log.info("Predicting on the test set.")
    X_test = dataset.test.X_enc
    y_test = dataset.test.y_enc
    with Timer() as predict:
        predictions = autoPyTorch.predict(X_test)
    probabilities = autoPyTorch.predict(X_test, return_probabilities=True)[1] if is_classification else None

    print(probabilities)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  training_duration=training.duration,
                  predict_duration=predict.duration
    )


def make_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    touch(subdir, as_dir=True)
    return subdir


if __name__ == '__main__':
    call_run(run)
