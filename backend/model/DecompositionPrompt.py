import dspy
import warnings
from opik.integrations.dspy.callback import OpikCallback

warnings.filterwarnings("ignore")


class SubtaskPrompt(dspy.Signature):
    subtask = dspy.InputField()
    category = dspy.OutputField()
    significance = dspy.OutputField()


class DecompositionPrompt(dspy.Signature):
    input_text = dspy.InputField()
    subtasks = dspy.OutputField()


class DecompositionModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.opik_callback = OpikCallback(project_name="NX1")

    def forward(self, input_text):
        self.opik_callback.on_module_start(
            call_id="decomposition_task",
            instance=self,
            inputs={"input_text": input_text},
        )

        try:
            decomposer = dspy.Predict(DecompositionPrompt)
            decomp_result = decomposer(input_text=input_text)

            # Debug: Print the raw output from the decomposer
            print(f"Decomposition Output: {decomp_result}")

            # Validate subtasks
            if not hasattr(decomp_result, "subtasks"):
                raise ValueError("The 'subtasks' field is missing in the output.")

            subtasks = decomp_result.subtasks

            if not isinstance(subtasks, list):
                raise ValueError("Subtasks must be a list.")

            if not subtasks:
                print("No subtasks generated.")
                return None, None

            subtask_solver = dspy.Predict(SubtaskPrompt)
            final_categories = []
            final_significances = []

            for subtask in subtasks:
                subtask_result = subtask_solver(subtask=subtask)
                final_categories.append(subtask_result.category)
                final_significances.append(subtask_result.significance)

            combined_category = max(set(final_categories), key=final_categories.count)
            combined_significance = max(set(final_significances), key=final_significances.count)

            self.opik_callback.on_module_end(
                call_id="decomposition_task",
                outputs={
                    "category": combined_category,
                    "significance": combined_significance,
                    "subtasks": subtasks,
                },
            )
            self.opik_callback.flush()

            return combined_category, combined_significance

        except Exception as e:
            self.opik_callback.on_module_end(
                call_id="decomposition_task",
                outputs=None,
                exception=e,
            )
            self.opik_callback.flush()
            print(f"Error during decomposition: {e}")
            return None, None