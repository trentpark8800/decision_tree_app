from typing import TypeVar, NamedTuple, List, Tuple
from collections import namedtuple

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score
from sklearn import tree

DataFrame = TypeVar("pandas.core.DataFrame")
Series = TypeVar("pandas.core.Series")
Figure = TypeVar("matplotlib.pyplot.figure")


def create_decision_tree(
    max_depth: int, info_purity_measure: str, train_df: DataFrame, features: List[str], target: str
) -> DecisionTreeClassifier:
    """_summary_

    Args:
        max_depth (int): _description_
        info_purity_measure (str): _description_
        train_df (DataFrame): _description_
        features (List[str]): _description_
        target (str): _description_

    Returns:
        DecisionTreeClassifier: _description_
    """
    classifier = DecisionTreeClassifier(
        max_depth=max_depth, criterion=info_purity_measure, random_state=42
    )

    classifier.fit(train_df[features], train_df[target])

    return classifier


def plot_decision_tree(
    classifier: DecisionTreeClassifier, features: List[str], class_names: List[str]
) -> Figure:
    """_summary_

    Args:
        classifier (DecisionTreeClassifier): _description_
        features (List[str]): _description_
        class_names (List[str]): _description_

    Returns:
        _type_: _description_
    """

    fig, ax = plt.subplots(figsize=(18, 16))

    _ = tree.plot_tree(
        classifier,
        precision=1,
        fontsize=12,
        feature_names=features,
        class_names=class_names,
        ax=ax,
        filled=True,
    )

    return fig


def evaluate_decision_tree(
    classifier: DecisionTreeClassifier,
    input_df: DataFrame,
    features: List[str],
    target: str,
    class_names: List[str],
) -> Tuple[Figure, NamedTuple]:
    """_summary_

    Args:
        classifier (DecisionTreeClassifier): _description_
        input_df (DataFrame): _description_
        features (List[str]): _description_
        target (str): _description_

    Returns:
        Tuple[Figure, NamedTuple]: _description_
    """
    predicted_output = classifier.predict(input_df[features])

    # Get accuracy, precision and f1 scores
    EvaluationMetrics: NamedTuple = namedtuple("EvaluationMetrics", "accuracy precision f1_score")
    evaluation_metrics = EvaluationMetrics(
        accuracy_score(y_true=input_df[target], y_pred=predicted_output),
        precision_score(y_true=input_df[target], y_pred=predicted_output, average="weighted"),
        f1_score(y_true=input_df[target], y_pred=predicted_output, average="weighted"),
    )

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    _ = ConfusionMatrixDisplay.from_estimator(
        estimator=classifier,
        X=input_df[features],
        y=input_df[target],
        display_labels=class_names,
        ax=ax,
    )

    return fig, evaluation_metrics
