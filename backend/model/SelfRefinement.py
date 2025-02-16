import dspy
import warnings
from opik.integrations.dspy.callback import OpikCallback

warnings.filterwarnings("ignore")

class SelfRefinementPrompt(dspy.Signature):
    input_text = dspy.InputField()
    initial_category = dspy.OutputField()
    initial_significance = dspy.OutputField()
    refined_category = dspy.OutputField()
    refined_significance = dspy.OutputField()

class SelfRefinementModule(dspy.Module):
    def __init__(self, num_iterations=2):
        super().__init__()
        self.num_iterations = num_iterations
        self.opik_callback = OpikCallback(project_name="NX1")

    def forward(self, input_text):
        self.opik_callback.on_module_start("self_refinement_task", None, {"input_text": input_text})

        predictor = dspy.Predict(SelfRefinementPrompt)
        result = predictor(input_text=input_text)

        current_category = result.initial_category
        current_significance = result.initial_significance

        for _ in range(self.num_iterations):
            result = predictor(
                input_text=f"Refine category: {current_category}, significance: {current_significance}"
            )
            current_category = result.refined_category
            current_significance = result.refined_significance

        self.opik_callback.on_module_end("self_refinement_task", {"category": current_category, "significance": current_significance})
        self.opik_callback.flush()

        return current_category, current_significance
