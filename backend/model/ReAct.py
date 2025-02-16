import dspy
import warnings
from opik.integrations.dspy.callback import OpikCallback

warnings.filterwarnings("ignore")


class ReActPrompt(dspy.Signature):
    system_content = dspy.InputField()
    user_content = dspy.InputField()
    response = dspy.OutputField()


class ReActFramework(dspy.Module):
    def __init__(self, model_name="ollama_chat/llama3.2"):
        super().__init__()
        self.model_name = model_name
        self.predictor = dspy.Predict(ReActPrompt)
        self.opik_callback = OpikCallback(project_name="NX1")

    def get_completion(self, system_content: str, user_content: str) -> str:
        self.opik_callback.on_module_start(
            call_id="react_task",
            instance=self.predictor,
            inputs={"system_content": system_content, "user_content": user_content},
        )

        try:
            result = self.predictor(system_content=system_content, user_content=user_content)
            self.opik_callback.on_module_end(
                call_id="react_task",
                outputs={"response": result.response},
            )
            self.opik_callback.flush()
            return result.response.strip()
        except Exception as e:
            self.opik_callback.on_module_end(
                call_id="react_task",
                outputs=None,
                exception=e,
            )
            self.opik_callback.flush()
            print(f"Error during ReAct completion: {e}")
            return None