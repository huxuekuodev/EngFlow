class HypoQuestionResult(BaseModel):
    Question: List[str] = Field(description="问题集合，每个元素是一个问题")


class HypoQuestionGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = HYPO_QUESTION_PROMPT
