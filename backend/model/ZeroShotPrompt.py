import dspy
import warnings
from opik.integrations.dspy.callback import OpikCallback

warnings.filterwarnings("ignore")

class ZeroShotPrompt(dspy.Signature):
    input_text = dspy.InputField()
    category = dspy.OutputField()
    significance = dspy.OutputField()

class ZeroShotModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.opik_callback = OpikCallback(project_name="NX1")

    def forward(self, input_text):
        self.opik_callback.on_module_start("zero_shot_task", None, {"input_text": input_text})
        predictor = dspy.Predict(ZeroShotPrompt)
        result = predictor(input_text=input_text)
        self.opik_callback.on_module_end("zero_shot_task", {"category": result.category, "significance": result.significance})
        self.opik_callback.flush()
        return result.category, result.significance
