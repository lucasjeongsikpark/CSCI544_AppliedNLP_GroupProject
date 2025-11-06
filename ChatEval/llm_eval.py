import os
# os.environ["OPENAI_API_KEY"] = "***"
# os.environ["OPENAI_BASE_URL"] = "***"

# always remember to put these lines at the top of your code if you are using clash
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["all_proxy"] = "socks5://127.0.0.1:7890"


import json
from eval_helper.get_evaluation import get_evaluation

from agentverse.agentverse import AgentVerse
from argparse import ArgumentParser

import time

parser = ArgumentParser()

parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--reverse_input", default=False, action="store_true")


args = parser.parse_args()

agentverse, args_data_path, args_output_dir = AgentVerse.from_task(args.config)

print(args)

os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

# uncomment this line if you don't want to overwrite your output_dir
# if os.path.exists(args_output_dir) and len(os.listdir(args_output_dir)) > 1 :
#
#     raise ValueError("the output_dir is not empty, check if is expected.")

def _parsing_evaluation_response_to_score(evaluation_response: list):
            flag = False
            # score1 example:
            # score1 = {"Correctness": 4, "Reasoning": 4, "Completeness": 5, "Accuracy": 4}
            score1 = {}
            score2 = {}
            evaluation_response = evaluation_response[1]['evaluation']
            print('-'*50)
            print("**** evaluation_response *****:", evaluation_response)
            print('-'*50)
            if "The score of Assistant 1:" not in evaluation_response or "The score of Assistant 2:" not in evaluation_response:
                return None, None

            scores1_lines = evaluation_response.split("The score of Assistant 1:")[1].split("The score of Assistant 2:")[0].strip().split("\n")
            scores2_lines = evaluation_response.split("The score of Assistant 2:")[1].strip().split("\n")
            # print("scores1_lines:", scores1_lines)
            # print("scores2_lines:", scores2_lines)
            try:
                for line in scores1_lines:
                    if ':' in line:
                        # score1 = {"Correctness": 4, "Reasoning": 4, "Completeness": 5, "Accuracy": 4}
                        key, value = line.split(":")
                        score1[key.strip()] = int(value.strip())
                for line in scores2_lines:
                    if ':' in line:
                        key, value = line.split(":")
                        score2[key.strip()] = int(value.strip())
            except:
                return None, None
                    
            return score1, score2

with open(args_data_path) as f:
    data = json.load(f)

if "faireval" in args_data_path:
    # pair_comparison_output = []
    
    for num, ins in enumerate(data[176:]):
        start_time = time.time()
        attempt = 0
        max_attempts = 50
        while True:
            attempt += 1
            if attempt > max_attempts:
                print("Max attempts reached. Skipping this instance.")
                score1, score2 = None, None
                break
            pair_comparison_output = []
            print(f"================================instance {num}====================================")
            # print(ins)
            # reassign the text to agents, and set final_prompt to null for debate at first round
            for agent_id in range(len(agentverse.agents)):
                agentverse.agents[agent_id].input_text = ins["input"]
                agentverse.agents[agent_id].output_text = ins["output"]

                agentverse.agents[agent_id].llama_text = ins["llama_output"]
                agentverse.agents[agent_id].distill_llama_text = ins["distill_llama_output"]

                if "document" in ins and ins["document"]:
                    agentverse.agents[agent_id].document = ins["document"]

                agentverse.agents[agent_id].final_prompt = ""

            agentverse.run()
            
            evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages, agent_nums=len(agentverse.agents))
            score1, score2 = _parsing_evaluation_response_to_score(evaluation)
            if score1 is not None and score2 is not None:
                break
            else:
                print("Re-evaluating due to parsing error...")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
                
        if "document" in ins and ins["document"]:
            pair_comparison_output.append({"input": ins["input"],
                                            "output": ins["output"],
                                            "document": ins["document"],
                                        "response": {"llama": ins["llama_output"],
                                                        "distill_llama": ins["distill_llama_output"]},
                                        "evaluation": evaluation,
                                        "score1": score1,
                                        "score2": score2,
                                        "attempt": attempt,
                                        "elapsed_time": elapsed_time
                                        })
        elif "system_prompt" in ins and ins["system_prompt"]:
            pair_comparison_output.append({"input": ins["input"],
                                        "output": ins["output"],
                                        "system_prompt": ins["system_prompt"],
                                       "response": {"llama": ins["llama_output"],
                                                    "distill_llama": ins["distill_llama_output"]},
                                       "evaluation": evaluation,
                                       "score1": score1,
                                       "score2": score2,
                                       "attempt": attempt,
                                        "elapsed_time": elapsed_time
                                       })
        else:
            pair_comparison_output.append({"input": ins["input"],
                                        "output": ins["output"],
                                       "response": {"llama": ins["llama_output"],
                                                    "distill_llama": ins["distill_llama_output"]},
                                       "evaluation": evaluation,
                                       "score1": score1,
                                       "score2": score2,
                                       "attempt": attempt,
                                        "elapsed_time": elapsed_time
                                       })
        
        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "pair_comparison_results.jsonl"), "a") as f:
            json.dump(pair_comparison_output, f, indent=4)
            f.write("\n")
    # with open(os.path.join(args_output_dir, "gt_origin_results.json"), "w") as f:
    #     json.dump(gt_origin_output, f, indent=4)

elif "adversarial" in args_data_path:

    pair_comparison_output = []

    for num, ins in enumerate(data):

        print(f"================================instance {num}====================================")

        # reassign the text to agents, and set final_prompt to null for debate at first round
        for agent_id in range(len(agentverse.agents)):
            agentverse.agents[agent_id].source_text = ins["question"]

            if args.reverse_input:
                agentverse.agents[agent_id].compared_text_one = ins["response"]["output_2"]
                agentverse.agents[agent_id].compared_text_two = ins["response"]["output_1"]
            else:
                agentverse.agents[agent_id].compared_text_one = ins["response"]["output_1"]
                agentverse.agents[agent_id].compared_text_two = ins["response"]["output_2"]

            agentverse.agents[agent_id].final_prompt = ""

        agentverse.run()

        evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages,
                                    agent_nums=len(agentverse.agents))

        pair_comparison_output.append({"question": ins["question"],
                                       "response": {"output_1": ins["response"]["output_1"],
                                                    "output_2": ins["response"]["output_2"]},
                                       "evaluation": evaluation})
 
        os.makedirs(args_output_dir, exist_ok=True)
        with open(os.path.join(args_output_dir, "pair_comparison_results.json"), "w") as f:
            json.dump(pair_comparison_output, f, indent=4)