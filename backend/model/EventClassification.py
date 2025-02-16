import dspy
import warnings
import os
from .ChainOfThought import ChainOfThoughtEventClassifier
from .ZeroShotPrompt import ZeroShotModule
from .FewShotPrompt import FewShotModule
from .SelfConsistency import SelfConsistencyModule
from .TreeOfThought import ToTModule
from .SelfRefinement import SelfRefinementModule
from .DecompositionPrompt import DecompositionModule
from .ReAct import ReActFramework

warnings.filterwarnings("ignore")


class LocalLLMEventClassification:
    def __init__(self, path_to_txt, model="ollama_chat/llama3.2", event_categories=None):
        self.path = path_to_txt
        self.model = model
        self.event_categories = event_categories or [
            "Acquisition",
            "Customer Event",
            "Personnel Change",
            "Scheduling Event"
        ]

        self.chain_of_thought_classifier = ChainOfThoughtEventClassifier(model_name=self.model)
        self.zero_shot_classifier = ZeroShotModule()
        self.few_shot_classifier = FewShotModule(examples=[
            {"input_text": "CEO resignation", "category": "Personnel Change", "significance": "Significant"},
            {"input_text": "Small acquisition", "category": "Acquisition", "significance": "Not Significant"}
        ])
        self.self_consistency_classifier = SelfConsistencyModule(num_samples=5)
        self.tree_of_thought_classifier = ToTModule(beam_width=3, max_depth=3)
        self.self_refinement_classifier = SelfRefinementModule(num_iterations=2)
        self.decomposition_classifier = DecompositionModule()
        self.react_framework = ReActFramework(model_name=self.model)

    def read_text(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as file:
            return file.read()

    def classify_events(self, method="cot", use_guided=False, history=None):
        text_content = self.read_text()
        task = "8-K Event Classification"

        prompt = f"""
        You are a financial analyst specializing in event classification for stocks. 
        Given the following event description, classify it into one of the predefined categories:
        {', '.join(self.event_categories)}.

        Additionally, determine whether the event is "significant" or "not significant" based on its 
        potential impact on the stock price. Consider the scale and importance of the event.

        Event description: "{text_content}"

        Provide your response in JSON format as:
        {{"category": "<category>", "significance": "<significance>"}}
        """

        if method == "guided":
            self.chain_of_thought_classifier.guided_cot_event_classifier_task(task, question=prompt, history=history)
        elif method == "zero_shot":
            category, significance = self.zero_shot_classifier.forward(text_content)
            print(f"Zero-Shot Classification: Category={category}, Significance={significance}")
        elif method == "few_shot":
            category, significance = self.few_shot_classifier.forward(text_content)
            print(f"Few-Shot Classification: Category={category}, Significance={significance}")
        elif method == "self_consistency":
            category, significance = self.self_consistency_classifier.forward(text_content)
            print(f"Self-Consistency Classification: Category={category}, Significance={significance}")
        elif method == "tree_of_thought":
            _, _, category, significance = self.tree_of_thought_classifier.forward(text_content)
            print(f"Tree-of-Thought Classification: Category={category}, Significance={significance}")
        elif method == "self_refinement":
            category, significance = self.self_refinement_classifier.forward(text_content)
            print(f"Self-Refinement Classification: Category={category}, Significance={significance}")
        elif method == "decomposition":
            category, significance = self.decomposition_classifier.forward(text_content)
            print(f"Decomposition Classification: Category={category}, Significance={significance}")
        elif method == "react":
            system_content = f"""
            You are a financial analyst specializing in event classification for stocks. 
            Given the following event description, classify it into one of the predefined categories:
            {', '.join(self.event_categories)}.

            Additionally, determine whether the event is "significant" or "not significant" based on its 
            potential impact on the stock price. Consider the scale and importance of the event.
            """
            response = self.react_framework.get_completion(system_content, text_content)
            print(f"ReAct Framework Classification: {response}")
        else:
            self.chain_of_thought_classifier.cot_event_classifier_task(task, question=prompt, history=history)