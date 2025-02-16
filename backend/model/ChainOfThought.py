import dspy
import warnings
from opik.integrations.dspy.callback import OpikCallback

warnings.filterwarnings("ignore")


class CoT(dspy.Signature):
    problem_text = dspy.InputField()
    reasoning = dspy.OutputField()
    category = dspy.OutputField()
    significance = dspy.OutputField()


class GuidedCoT(dspy.Signature):
    problem_text = dspy.InputField()
    identification = dspy.OutputField(desc="Identify the key aspects of the event.")
    context = dspy.OutputField(desc="Provide background or relevant context.")
    analysis = dspy.OutputField(desc="Analyze and classify the event type.")
    conclusion = dspy.OutputField(desc="Summarize the final event classification.")


class ChainOfThoughtEventClassifier(dspy.Module):
    def __init__(self, model_name="ollama_chat/llama3.2", api_base="http://localhost:11434", api_key=""):
        super().__init__()
        self.lm = dspy.LM(model_name, api_base=api_base, api_key=api_key)
        self.opik_callback = OpikCallback(project_name="NX1")
        dspy.settings.configure(lm=self.lm, callbacks=[self.opik_callback])

    def cot_event_classifier_task(self, task, question=None, history=None):
        self._log_task_start(task, question)

        cot = dspy.Predict(CoT)
        response = cot(problem_text=question)

        result = {
            "category": response.category,
            "significance": response.significance,
            "reasoning": response.reasoning
        }

        self._log_task_end(task, cot, result)
        self._display_result(result, history)

        return result

    def guided_cot_event_classifier_task(self, task, question=None, history=None):
        self._log_task_start(task, question)

        guided_cot = dspy.Predict(GuidedCoT)
        response = guided_cot(problem_text=question)

        result = {
            "identification": response.identification,
            "context": response.context,
            "analysis": response.analysis,
            "conclusion": response.conclusion
        }

        self._log_task_end(task, guided_cot, result)
        self._display_guided_result(result, history)

        return result

    def _log_task_start(self, task, question):
        print(f"Starting {task} Classification")
        print(f"Problem: {question}")

    def _log_task_end(self, task, instance, result):
        self.opik_callback.on_module_start(call_id=f"{task}_task", instance=instance, inputs={"problem_text": result})
        self.opik_callback.on_module_end(call_id=f"{task}_task", outputs=result)
        self.opik_callback.flush()

    def _display_result(self, result, history):
        print(f"Result: {result['category']} | Significance: {result['significance']}")
        print(f"Reasoning: {result['reasoning']}")
        self._display_history(history)

    def _display_guided_result(self, result, history):
        print(f"**Step 1 - Identification:** {result['identification']}")
        print(f"**Step 2 - Context:** {result['context']}")
        print(f"**Step 3 - Analysis:** {result['analysis']}")
        print(f"**Step 4 - Conclusion:** {result['conclusion']}")
        self._display_history(history)

    def _display_history(self, history):
        if history:
            print("Prompt History:")
            print(self.lm.inspect_history(n=history))
            print("===========================\n")
