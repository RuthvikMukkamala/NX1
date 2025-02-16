import dspy
import warnings
from opik.integrations.dspy.callback import OpikCallback
from .ChainOfThought import ChainOfThoughtEventClassifier

warnings.filterwarnings("ignore")


class SelfConsistencyModule(dspy.Module):
    def __init__(self, num_samples=3, model_name="ollama_chat/llama3.2"):
        super().__init__()
        self.classifier = ChainOfThoughtEventClassifier(model_name=model_name)
        self.num_samples = num_samples
        self.opik_callback = OpikCallback(project_name="NX1")

    def forward(self, input_text):
        self.opik_callback.on_module_start("self_consistency_task", self.classifier, {"input_text": input_text})

        results = [self.classifier.cot_event_classifier_task(task="Self-Consistency", question=input_text) for _ in
                   range(self.num_samples)]

        categories = [result["category"] for result in results]
        significances = [result["significance"] for result in results]

        final_category = max(set(categories), key=categories.count)
        final_significance = max(set(significances), key=significances.count)

        self.opik_callback.on_module_end("self_consistency_task",
                                         {"category": final_category, "significance": final_significance})
        self.opik_callback.flush()

        return final_category, final_significance
