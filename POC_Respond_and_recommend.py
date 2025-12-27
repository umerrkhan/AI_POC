import json
with open("./docs/feedback.json","r", encoding="utf-8") as file:
    data = json.load(file)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableLambda,RunnableParallel
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()

class FeedbackCategorization(BaseModel):
    overall_response:Optional[Literal['positive','negative']] = Field(description='Given Text is positive or negative')
    isrefundRequested:Optional[Literal['Yes','No']]= Field(description='Given Text is refund requested or not')  
    TroublewithSupport:Optional[Literal['Yes','No']]= Field(description='Given Text is refund requested or not')  
    
prompt1 = PromptTemplate(template="Generate a one-line supportive response to the following feedback while avoiding any admission of error or responsibility:\n {customer_feedback}", input_variables=['customer_feedback'])
prompt2 = PromptTemplate(template="From the customer feedback, determine the the name of the course being referred to:\n {customer_feedback}", input_variables=['customer_feedback'])
prompt3 = PromptTemplate(template="recommendation courses based on the following {topic} from udemy.com", input_variables=['topic'])   
parser = StrOutputParser()




for entry in data["customer_feedbacks"]:    
    feedback = entry["feedback"]
    loginid = entry["loginid"]
    print("\n"*10)
    print("-"*40)
    print(f"\ncustomer id: {loginid}")
    print(f"\nFeedback: {feedback}")
    structured_model = model.with_structured_output(FeedbackCategorization)    
    result = structured_model.invoke(feedback)
    print(f"overall_response: {result.overall_response}")
    print(f"isrefundRequested: {result.isrefundRequested}")
    print(f"TroublewithSupport: {result.TroublewithSupport}")

    parallel_chain = RunnableParallel({
        'response_for_customer': RunnableSequence(prompt1,model,parser),        
        'topic': RunnableSequence(prompt2,model,parser),
        })
    result = parallel_chain.invoke({'customer_feedback': feedback})

    
    print(f"\nTopic: {result['topic']}")
    print(f"\nresponse_for_customer: {result['response_for_customer']}")

    final_chain = RunnableSequence(prompt3,model,parser)
    result = final_chain.invoke(result['topic'])
    print(f"\n{result}")
    structured_model.get_graph().print_ascii()

