## Imports

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from .core.models import get_evaluator

def get_evaluation_results(query, actual_output, context):
    scores = {}
    faithfullness = FaithfulnessMetric(
    threshold=0.5,
    model=get_evaluator(),
    include_reason=False
    )

    answer_relevancy = AnswerRelevancyMetric(threshold=0.5,
    model=get_evaluator(),
    include_reason=False
    )

    test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    retrieval_context=list(context.to_list())
    )

    results = evaluate(test_cases=[test_case],metrics=[faithfullness,answer_relevancy])
    for _, test_results in results:
        if test_results:
            for t in test_results:
                for m in t.metrics_data:
                    scores[m.name] = m.score

    return scores["Faithfulness"], scores["Answer Relevancy"]