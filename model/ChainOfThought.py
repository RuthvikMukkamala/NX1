import dspy



class ChainOfThought():

    def __init__(self):
        self.lm = dspy.LM('ollama_chat/llama3.2x', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=self.lm)
