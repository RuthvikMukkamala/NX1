import dspy
import warnings
from opik.integrations.dspy.callback import OpikCallback

warnings.filterwarnings("ignore")


class FewShotPrompt(dspy.Signature):
    examples = dspy.InputField()
    input_text = dspy.InputField()
    category = dspy.OutputField()
    significance = dspy.OutputField()


class FewShotModule(dspy.Module):
    def __init__(self, examples):
        super().__init__()
        self.examples = examples
        self.opik_callback = OpikCallback(project_name="NX1")

    def forward(self, input_text):
        self.opik_callback.on_module_start("few_shot_task", None, {"input_text": input_text, "examples": self.examples})

        predictor = dspy.Predict(FewShotPrompt)
        result = predictor(examples=self.examples, input_text=input_text)

        self.opik_callback.on_module_end("few_shot_task",
                                         {"category": result.category, "significance": result.significance})
        self.opik_callback.flush()

        return result.category, result.significance
