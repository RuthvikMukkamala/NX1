import dspy
import warnings
from opik.integrations.dspy.callback import OpikCallback

warnings.filterwarnings("ignore")

class ToTPrompt(dspy.Signature):
    input_text = dspy.InputField()
    thoughts = dspy.OutputField()
    evaluations = dspy.OutputField()
    category = dspy.OutputField()
    significance = dspy.OutputField()

class ToTModule(dspy.Module):
    def __init__(self, beam_width=3, max_depth=3):
        super().__init__()
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.opik_callback = OpikCallback(project_name="NX1")

    def forward(self, input_text):
        self.opik_callback.on_module_start("tree_of_thought_task", None, {"input_text": input_text})

        predictor = dspy.Predict(ToTPrompt)
        result = predictor(input_text=input_text)

        self.opik_callback.on_module_end("tree_of_thought_task", {
            "thoughts": result.thoughts,
            "evaluations": result.evaluations,
            "category": result.category,
            "significance": result.significance
        })
        self.opik_callback.flush()

        return result.thoughts, result.evaluations, result.category, result.significance
