import os
import json
from datetime import datetime

from promptflow.client import PFClient
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluate import evaluate
from promptflow.evals.evaluators import RelevanceEvaluator, FluencyEvaluator, GroundednessEvaluator, CoherenceEvaluator

from azure_config import AzureConfig 

def main():

    # Read configuration
    azure_config = AzureConfig()

    # Set required environment variables
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_config.aoai_endpoint
    os.environ['AZURE_OPENAI_API_KEY'] = azure_config.aoai_api_key    

    ##################################
    ## Base Run
    ##################################

    pf = PFClient()
    flow = "./src"
    data = "./evaluations/test-dataset.jsonl"

    if not os.path.exists(data):
        print(f"❌ Data file not found: {data}")
        return

    try:
        base_run = pf.run(
            flow=flow,
            data=data,
            column_mapping={
                "question": "${data.question}",
                "chat_history": []
            },
            stream=True,
        )

        responses = pf.get_details(base_run)
        print(responses.head(10))

        # Convert to jsonl
        relevant_columns = responses[['inputs.question', 'inputs.chat_history', 'outputs.answer', 'outputs.context']]
        relevant_columns.columns = ['question', 'chat_history', 'answer', 'context']
        data_list = relevant_columns.to_dict(orient='records')
        with open('responses.jsonl', 'w') as f:
            for item in data_list:
                f.write(json.dumps(item) + '\n')

    except Exception as e:
        print(f"❌ Error during base run or processing responses: {e}")
        return

    ##################################
    ## Evaluation
    ##################################

    # Initialize Azure OpenAI connection
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=azure_config.aoai_endpoint,
        api_key=azure_config.aoai_api_key,
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version=azure_config.aoai_api_version,
    )

    azure_ai_project = {
        "subscription_id": azure_config.subscription_id,
        "resource_group_name": azure_config.resource_group,
        "project_name": azure_config.workspace_name,
    }

    fluency_evaluator = FluencyEvaluator(model_config=model_config)
    groundedness_evaluator = GroundednessEvaluator(model_config=model_config)
    relevance_evaluator = RelevanceEvaluator(model_config=model_config)
    coherence_evaluator = CoherenceEvaluator(model_config=model_config)

    data = "./responses.jsonl"
    if not os.path.exists(data):
        print(f"❌ Response data file not found: {data}")
        return

    prefix = os.getenv("PREFIX", datetime.now().strftime("%y%m%d%H%M%S"))[:14] 
    evaluation_name = f"{prefix} Quality Evaluation"

    print(f"🚀 Executing evaluation: {evaluation_name}")

    try:
        result = evaluate(
            evaluation_name=evaluation_name,
            data=data,
            evaluators={
                "Fluency": fluency_evaluator,
                "Groundedness": groundedness_evaluator,
                "Relevance": relevance_evaluator,
                "Coherence": coherence_evaluator
            },
            azure_ai_project=azure_ai_project,
            output_path="./qa_flow_quality_eval.json"
        )
    except Exception as e:
        print(f"⚠️ Evaluation failed with Azure project. Retrying without Azure project: {e}")
        result = evaluate(
            evaluation_name=evaluation_name,
            data=data,
            evaluators={
                "Fluency": fluency_evaluator,
                "Groundedness": groundedness_evaluator,
                "Relevance": relevance_evaluator,
                "Coherence": coherence_evaluator
            },
            output_path="./qa_flow_quality_eval.json"
        )

    print(f"✅ Check QA evaluation result {evaluation_name} in the 'Evaluation' section of your project: {azure_config.workspace_name}")

if __name__ == '__main__':
    import promptflow as pf
    main()
